import gym
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import copy
from distutils.version import LooseVersion
import matplotlib.pyplot as plt
EPS = 1e-6
print("gym version:", gym.__version__)
print("tensorflow version:", tf.__version__)


class Replayer:
    """
    经验存储和回放
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

    def __call__(self):
        n = np.random.normal(size=self.size)
        self.x += (self.theta * (self.mu - self.x) * self.dt +
                   self.sigma * np.sqrt(self.dt) * n)
        return self.x

    def reset(self, x=0.):
        self.x = x * np.ones(self.size)


def adaptive_isotropic_gaussian_kernel(xs, ys, h_min=1e-3):
    """
    用于SVGD的核函数
    :param xs:
    :param ys:
    :param h_min:
    :return:
    """
    Kx, D = xs.get_shape().as_list()[-2:]
    Ky, D2 = ys.get_shape().as_list()[-2:]
    assert D == D2

    leading_shape = tf.shape(input=xs)[:-2]

    # 计算pairwise distances
    diff = tf.expand_dims(xs, -2) - tf.expand_dims(ys, -3)

    if LooseVersion(tf.__version__) <= LooseVersion('1.5.0'):
        dist_sq = tf.reduce_sum(input_tensor=diff ** 2, axis=-1, keepdims=False)
    else:
        dist_sq = tf.reduce_sum(input_tensor=diff ** 2, axis=-1, keepdims=False)

    # 计算 median.
    input_shape = tf.concat((leading_shape, [Kx * Ky]), axis=0)
    values, _ = tf.nn.top_k(
        input=tf.reshape(dist_sq, input_shape),
        k=(Kx * Ky // 2 + 1),
        sorted=True)

    medians_sq = values[..., -1]

    h = medians_sq / np.log(Kx)
    h = tf.maximum(h, h_min)
    h = tf.stop_gradient(h)
    h_expanded_twice = tf.expand_dims(tf.expand_dims(h, -1), -1)

    kappa = tf.exp(-dist_sq / h_expanded_twice)
    # 重构梯度的维度
    h_expanded_thrice = tf.expand_dims(h_expanded_twice, -1)
    kappa_expanded = tf.expand_dims(kappa, -1)
    kappa_grad = -2 * diff / h_expanded_thrice * kappa_expanded

    return {"output": kappa, "gradient": kappa_grad}


class SQLAgent:
    def __init__(self, env, q_net, q_target_net, policy_net, q_optimizer, policy_optimizer,
                 replayer_capacity=10000, gamma=0.99, tau=0.005,
                 batches=1, batch_size=64, K=30, M=30, value_n_particles=16):
        """
        :param env: 环境
        :param q_net: Q网络
        :param q_target_net: Q网络的目标网络
        :param policy_net: 策略网络
        :param q_optimizer: Q网络优化器
        :param policy_optimizer: 策略网络优化器
        :param replayer_capacity: Replayer最大存储量
        :param gamma: 折扣系数
        :param tau:更新目标网络时的系数
        :param batches:
        :param batch_size:
        :param K:fixed action的个数
        :param M:updated action的个数
        :param value_n_particles:重要性采样计算V函数积分时的采样数
        """
        self.env = env
        self.replayer = Replayer(replayer_capacity)
        self.gamma = gamma
        self.batches = batches
        self.batch_size = batch_size

        self.q_net = q_net
        self.q_target_net = q_target_net
        self.policy_net = policy_net
        self.q_optimizer = q_optimizer
        self.policy_optimizer = policy_optimizer

        self.noise = OrnsteinUhlenbeckProcess(self.env.action_space.shape)
        self.noise.reset()

        self.value_n_particles = value_n_particles
        self.K = K
        self.M = M
        self.tau = tau
        self.kernel_fn = adaptive_isotropic_gaussian_kernel

    def update_target_net(self, target_net, evaluate_net, learning_rate=1.):
        """更新目标网络"""
        target_weights = target_net.get_weights()
        evaluate_weights = evaluate_net.get_weights()
        average_weights = [(1. - learning_rate) * t + learning_rate * e
                           for t, e in zip(target_weights, evaluate_weights)]
        target_net.set_weights(average_weights)

    def decide(self, observation, add_noise):
        """选取动作和环境交互"""
        action = self.policy_net(observation[np.newaxis])
        action = action.numpy()
        if add_noise:
            noise = self.noise()[:, np.newaxis]
            assert action.shape == noise.shape
            action = np.clip(noise + action,self.env.action_space.low,self.env.action_space.high)

        return action[0]

    def learn(self, observation, action, reward, net_observation, done):
        """更新agent"""
        self.replayer.store(observation, action, reward, net_observation, done)
        if done:
            self.noise.reset()
            for i in range(self.batches):
                observations, actions, rewards, \
                next_observations, dones = self.replayer.sample(self.batch_size)
                # 利用均匀分布采样计算V值的积分形式
                action_uniform_distribution = np.random.uniform(-1., 1.,  # 从均匀分布采样，计算V值
                                                                (1,
                                                                 self.value_n_particles,
                                                                 np.prod(self.env.action_space.shape)))
                next_q_values = self.q_target_net(next_observations[:, None, :],
                                                  action_uniform_distribution)
                next_v_values = tf.reduce_logsumexp(next_q_values, axis=1)
                next_v_values -= tf.math.log(tf.cast(self.value_n_particles, tf.float32))
                next_v_values += np.log(2) * np.prod(self.env.action_space.shape)
                # 利用mse loss计算Q网络的梯度并更新
                q_target = rewards[:, np.newaxis] + dones[:, np.newaxis] * self.gamma * next_v_values
                with tf.GradientTape() as tape:
                    q_values = self.q_net(observations, actions)
                    q_losses = tf.losses.mse(q_target, q_values)
                    q_loss = tf.nn.compute_average_loss(q_losses)
                grad = tape.gradient(q_loss, self.q_net.trainable_variables)
                self.q_optimizer.apply_gradients(zip(grad, self.q_net.trainable_variables))
                # 使用SVGD技术更新策略网络
                with tf.GradientTape() as tape1:
                    actions = self.policy_net(observations, self.K + self.M)
                    fixed_actions, updated_actions = tf.split(
                        actions, [self.K, self.M], axis=1)     # 采样K+M个action，并划分样本
                fixed_actions = tf.stop_gradient(fixed_actions)

                with tf.GradientTape() as tape2:
                    tape2.watch(fixed_actions)
                    svgd_target_values = self.q_net(observations[:, None, :], fixed_actions, )
                    squash_correction = tf.reduce_sum(
                        tf.math.log(1 - fixed_actions ** 2 + EPS), axis=-1, keepdims=True)
                    log_p = svgd_target_values + squash_correction  # 计算Q函数对fixed_actions的梯度
                grad_log_p = tape2.gradient(log_p, fixed_actions)[0]
                grad_log_p = tf.expand_dims(grad_log_p, axis=2)
                grad_log_p = tf.stop_gradient(grad_log_p)

                kernel_dict = self.kernel_fn(xs=fixed_actions, ys=updated_actions) #计算kernel函数

                kappa = tf.expand_dims(kernel_dict["output"], axis=3)
                action_gradients = tf.reduce_mean(
                    kappa * grad_log_p + kernel_dict["gradient"], axis=1)

                gradients = tape1.gradient(updated_actions, self.policy_net.trainable_variables,
                                           action_gradients)    # updated_actions对策略网络的梯度
                with tf.GradientTape() as tape3:                # 根据SVGD得到的更新方向更新策略网络
                    surrogate_loss = tf.reduce_sum([
                        tf.reduce_sum(w * tf.stop_gradient(g))
                        for w, g in zip(self.policy_net.trainable_variables, gradients)
                    ])
                loss_gradients = tape3.gradient(surrogate_loss, self.policy_net.trainable_variables)
                self.policy_optimizer.apply_gradients(zip(loss_gradients,
                                                          self.policy_net.trainable_variables))
                # 更新target_net
                self.update_target_net(self.q_target_net, self.q_net, self.tau)


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
    def __init__(self, output_size, squash=True):
        super(PolicyNet, self).__init__()
        self.layer1_observation = keras.layers.Dense(100, 'relu')
        self.layer1_kesi = keras.layers.Dense(100, 'relu')
        self.layer2 = keras.layers.Dense(100, 'relu')
        if squash:
            self.layer3 = keras.layers.Dense(output_size, 'tanh')
        else:
            self.layer3 = keras.layers.Dense(output_size)
        self.output_size = output_size

    def call(self, observations, n_samples=1):
        batch_size = observations.shape[0]
        if n_samples > 1:           # 如果采样数大于1，对每个observation生成kesi采样n_samples样本
            kesi_shape = [batch_size, n_samples, self.output_size]
            observations = observations[:, None, :]
        else:
            kesi_shape = [batch_size, self.output_size]
        kesi = tf.random.normal(kesi_shape)
        x1 = self.layer1_observation(observations)
        x2 = self.layer1_kesi(kesi)
        x = self.layer2(x1 + x2)
        actions = self.layer3(x)

        return actions


def main():
    env = gym.make('MountainCarContinuous-v0')
    # env = gym.make("Pendulum-v0")
    q_net = QNet()     # 实例化Q网络
    q_target_net = copy.deepcopy(q_net)  # 实例化Q目标网络
    policy_net = PolicyNet(env.action_space.shape[0])  # 实例化策略网络
    q_optimizer = tf.optimizers.Adam(1e-4)          # 定义优化器
    policy_optimizer = tf.optimizers.Adam(1e-4)
    agent = SQLAgent(env=env,
                     q_net=q_net, q_target_net=q_target_net, policy_net=policy_net,
                     q_optimizer=q_optimizer, policy_optimizer=policy_optimizer,
                     tau=0.005, batches=1, batch_size=64)
    # agent和环境交互，并且进行学习
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