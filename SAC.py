import gym
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
import copy
import matplotlib.pyplot as plt

print("gym version:", gym.__version__)
print("tensorflow version:", tf.__version__)

LOG_STD_MAX = 2
LOG_STD_MIN = -20
tf.random.set_seed(0)

class Replayer:
    """
    经验回放池
    """

    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity),
                                   columns=['observation', 'action', 'reward',
                                            'next_observation', 'done'])
        self.i = 0
        self.count = 0
        self.capacity = capacity

    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in
                self.memory.columns)


class OrnsteinUhlenbeckProcess:
    """
    用于给输出的action添加OrnsteinUhlenbeck噪声
    """

    def __init__(self, size, mu=0., sigma=1., theta=.15, dt=.01):
        self.size = size
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x = 0

    def __call__(self):
        n = np.random.normal(size=self.size)
        self.x += (self.theta * (self.mu - self.x) * self.dt +
                   self.sigma * np.sqrt(self.dt) * n)
        return self.x

    def reset(self, x=0.):
        self.x = x * np.ones(self.size)


class SACAgent:
    def __init__(self, env, q_nets, q_target_nets, policy_net,
                 q_optimizers, policy_optimizer, alpha_optimizer, auto_alpha=False, alpha=0.2,
                 replayer_capacity=100000, gamma=0.99, batches=1, tau=0.005,
                 batch_size=128, sample_threshold=1000, update_target_interval=1):
        """

        :param env: 环境
        :param q_nets: q网络
        :param q_target_nets:q目标网络
        :param policy_net: 策略网络
        :param q_optimizers: q网络优化器
        :param policy_optimizer: 策略网络优化器
        :param alpha_optimizer: 自动调节温度参数的优化器
        :param auto_alpha: 是否自动调节温度参数
        :param alpha: 温度参数
        :param replayer_capacity: 经验池最大容量
        :param gamma: 折扣系数
        :param batches:
        :param tau: 更新目标网络的系数
        :param batch_size:
        :param sample_threshold: 采样多少次后开始更新
        :param update_target_interval: 目标网络更新的间隔
        """
        self.env = env
        self.replayer = Replayer(replayer_capacity)
        self.gamma = gamma
        self.auto_alpha = auto_alpha
        self.h0 = -1. * self.env.action_space.shape[0]
        self.log_alpha = tf.Variable(0.)  # 使得alpha的值域为实数空间
        self.alpha = tfp.util.DeferredTensor(self.log_alpha, tf.exp) if auto_alpha else alpha
        self.alpha_optimizer = alpha_optimizer
        self.batches = batches
        self.tau = tau
        self.batch_size = batch_size
        self.action_low = self.env.action_space.low[0]
        self.action_high = self.env.action_space.high[0]

        self.noise = OrnsteinUhlenbeckProcess(env.action_space.shape)
        self.noise.reset()
        self.q_nets = q_nets
        self.q_target_nets = q_target_nets
        self.q_optimizers = q_optimizers
        self.policy_net = policy_net
        self.policy_optimizer = policy_optimizer
        self.update_target_interval = update_target_interval
        self.sample_threshold = sample_threshold
        self.iteration_times = 0

    def update_target_net(self, target_net, evaluate_net, learning_rate=0.005):
        """更新目标网络"""
        target_weights = target_net.get_weights()
        evaluate_weights = evaluate_net.get_weights()
        average_weights = [(1. - learning_rate) * t + learning_rate * e
                           for t, e in zip(target_weights, evaluate_weights)]
        target_net.set_weights(average_weights)

    def decide(self, observation, add_noise=True, deterministic=False):
        """与环境交互的动作选择"""
        action, _ = self.policy_net(observation[np.newaxis],
                                    deterministic=deterministic,
                                    training=False)
        action = action.numpy()
        if add_noise:
            noise = self.noise()[:, np.newaxis]
            assert action.shape == noise.shape
            action = noise + action

        return action[0]

    def learn(self, observation, action, reward, next_observation, done):
        self.iteration_times += 1
        self.replayer.store(observation, action, reward, next_observation, done)
        if done and self.sample_threshold < self.iteration_times:
            self.noise.reset()
            for i in range(self.batches):
                observations, actions, rewards, \
                next_observations, dones = self.replayer.sample(self.batch_size)
                next_actions, next_action_log_probs = tf.stop_gradient(
                    self.policy_net(next_observations,
                                    with_log_probs=True,
                                    training=False)
                )
                # 计算next value
                next_v_values = tuple(q_target(next_observations, next_actions)
                                      for q_target in self.q_target_nets)
                next_v_values = tf.reduce_min(next_v_values, axis=0)
                next_v_values = tf.stop_gradient((1 - dones)[:, np.newaxis] * \
                                                 (next_v_values - self.alpha * next_action_log_probs))
                q_target_values = tf.stop_gradient(rewards[:, np.newaxis] +
                                                   self.gamma * next_v_values)
                # 更新q网络，依据论文此处是设置的两个q网络
                for q_net, q_optimizer in zip(self.q_nets, self.q_optimizers):
                    with tf.GradientTape() as tape:
                        q_values = q_net(observations, actions)
                        q_losses = 0.5 * tf.pow((q_values -
                                                 q_target_values), 2)
                        q_loss = tf.nn.compute_average_loss(q_losses)
                    grad1 = tape.gradient(q_loss, q_net.trainable_variables)
                    q_optimizer.apply_gradients(zip(grad1, q_net.trainable_variables))
                # 更新策略网络
                with tf.GradientTape() as tape:
                    pi_actions, action_log_probs = self.policy_net(observations,
                                                                   with_log_probs=True,
                                                                   training=True
                                                                   )
                    pi_q_values = tuple(q_net(observations, pi_actions) for q_net in self.q_nets)
                    pi_q_values = tf.reduce_min(pi_q_values, axis=0)
                    policy_losses = self.alpha * action_log_probs - pi_q_values
                    policy_loss = tf.nn.compute_average_loss(policy_losses)
                grad2 = tape.gradient(policy_loss, self.policy_net.trainable_variables)
                self.policy_optimizer.apply_gradients(zip(grad2,
                                                          self.policy_net.trainable_variables))
                # 如果是自动化调节温度参数，更新温度参数
                if self.auto_alpha:
                    _, log_porbs = tf.stop_gradient(self.policy_net(observations,
                                                                    with_log_probs=True,
                                                                    training=False))
                    with tf.GradientTape() as tape:
                        alpha_losses = -1. * self.alpha * tf.stop_gradient(log_porbs + self.h0)
                        alpha_loss = tf.nn.compute_average_loss(alpha_losses)
                    grad3 = tape.gradient(alpha_loss, [self.log_alpha])
                    self.alpha_optimizer.apply_gradients(zip(grad3, [self.log_alpha]))
                # 更新目标网络
                if self.iteration_times % self.update_target_interval == 0:
                    for target_net, eval_net in zip(self.q_target_nets, self.q_nets):
                        self.update_target_net(target_net, eval_net, self.tau)
        return


class QNet(keras.Model):
    def __init__(self, ):
        super(QNet, self).__init__()
        self.layer1_observation = keras.layers.Dense(100, 'relu')
        self.layer1_action = keras.layers.Dense(100, 'relu')
        self.layer2 = keras.layers.Dense(100, 'relu')
        self.layer3 = keras.layers.Dense(1)

    def call(self, observations, actions):
        x1 = self.layer1_observation(observations)
        x2 = self.layer1_action(actions)
        x = self.layer2(x1 + x2)
        out = self.layer3(x)
        return out


class PolicyNet(keras.Model):
    def __init__(self, output_size, activation='relu'):
        super(PolicyNet, self).__init__()
        self.layer1 = keras.layers.Dense(100, activation)
        self.layer2 = keras.layers.Dense(100, activation)
        self.layer3_u = keras.layers.Dense(output_size)
        self.layer3_log_std = keras.layers.Dense(output_size)
        self.output_size = output_size

    def call(self, observations, with_log_probs=False, deterministic=False, training=False):
        x = self.layer1(observations)
        x = self.layer2(x)
        u = self.layer3_u(x)
        log_std = self.layer3_log_std(x)
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = tf.exp(log_std)
        pi_distribution = tfp.distributions.MultivariateNormalDiag(u, std)
        if deterministic:
            pi_action = u
        elif not training:
            pi_action = pi_distribution.sample()
        else:
            noise_distribution = tfp.distributions.Normal(tf.zeros(u.shape),
                                                          tf.ones(std.shape))
            noise = tf.stop_gradient(noise_distribution.sample())
            pi_action = u + noise * std
        if with_log_probs:
            logp_pi = tf.reduce_sum(pi_distribution.log_prob(pi_action), axis=-1, keepdims=True)
            logp_pi -= tf.reduce_sum(2 * (np.log(2) - pi_action -
                                          tf.math.softplus(-2 * pi_action)), axis=1, keepdims=True)
        else:
            logp_pi = None

        pi_action = 2. * tf.tanh(pi_action)

        return pi_action, logp_pi


class PolicyNet1(keras.Model):
    def __init__(self, output_size, activation='relu'):
        super(PolicyNet1, self).__init__()
        self.layer1_observation = keras.layers.Dense(100, activation)
        self.layer1_kesi = keras.layers.Dense(100, activation)
        self.layer2 = keras.layers.Dense(100, activation)
        self.layer3_u = keras.layers.Dense(output_size)
        self.layer3_log_std = keras.layers.Dense(output_size)
        self.output_size = output_size

    def call(self, observations, with_log_probs=False, deterministic=False, training=False):
        kesi = tf.random.normal((observations.shape[0], self.output_size))
        x1 = self.layer1_observation(observations)
        x2 = self.layer1_kesi(kesi)
        x = self.layer2(x1 + x2)
        u = self.layer3_u(x)
        log_std = self.layer3_log_std(x)
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = tf.exp(log_std)
        pi_distribution = tfp.distributions.Normal(u, std)
        if deterministic:
            pi_action = u
        elif not training:
            pi_action = u + std
        else:
            pi_action = u + std
        if with_log_probs:
            logp_pi = tf.reduce_sum(pi_distribution.log_prob(pi_action), axis=-1, keepdims=True)
            logp_pi -= tf.reduce_sum(2 * (np.log(2) - pi_action -
                                          tf.math.softplus(-2 * pi_action)), axis=1, keepdims=True)
        else:
            logp_pi = None

        pi_action = tf.tanh(pi_action)

        return pi_action, logp_pi


def main():
    env = gym.make('MountainCarContinuous-v0')
    # env = gym.make("Pendulum-v0")
    q_nets = [QNet() for i in range(2)]  # 定义q网络
    q_target_nets = [copy.deepcopy(i) for i in q_nets]  # 定义q目标网络
    policy_net = PolicyNet1(env.action_space.shape[0])  # 定义策略网络
    q_optimizers = [tf.optimizers.Adam(1e-3) for i in range(2)]
    policy_optimizer = tf.optimizers.Adam(3e-4)
    alpha_optimizer = tf.optimizers.Adam(3e-1)  # 定义优化器

    agent = SACAgent(env, q_nets, q_target_nets, policy_net,
                     q_optimizers, policy_optimizer, alpha_optimizer,
                     tau=0.005, batch_size=64, batches=1, update_target_interval=1,
                     auto_alpha=True, alpha=0.2
                     )
    rewards_list = []
    for i in range(100):
        print("episode: ", i)
        rewards = 0
        n = 0
        observation = env.reset()
        while True:
            n += 1
            action = agent.decide(observation, add_noise=False)
            next_observation, reward, done, info = env.step(action)
            rewards += reward
            agent.learn(observation, action, reward, next_observation, done)
            if done:
                print("rewards:{} n_steps:{} ".format(rewards, n, ))
                rewards_list.append(rewards / n)
                break
    plt.plot(rewards_list)
    plt.title("average reward per step")
    plt.xlabel("n episode")
    plt.ylabel("reward")


if __name__ == '__main__':
    main()