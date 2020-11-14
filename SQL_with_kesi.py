import gym
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import copy
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

EPS = 1e-6
tf.random.set_seed(0)
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
    :param xs:fixed_actions
    :param ys:updated_actions
    :param h_min:最小pairwise distances
    :return:
    """
    Kx, D = xs.get_shape().as_list()[-2:]
    Ky, D2 = ys.get_shape().as_list()[-2:]
    assert D == D2

    leading_shape = tf.shape(input=xs)[:-2]

    # 计算pairwise distances
    diff = tf.expand_dims(xs, -2) - tf.expand_dims(ys, -3)
    dist_sq = tf.reduce_sum(input_tensor=diff ** 2, axis=-1, keepdims=False)

    # 获取pairwise distances的中间值.
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
    # 计算kernel值
    kappa = tf.exp(-dist_sq / h_expanded_twice)
    # 计算kernel的梯度
    h_expanded_thrice = tf.expand_dims(h_expanded_twice, -1)
    kappa_expanded = tf.expand_dims(kappa, -1)
    kappa_grad = -2 * diff / h_expanded_thrice * kappa_expanded

    return {"output": kappa, "gradient": kappa_grad}


class SQLAgent:
    def __init__(self,
                 env,
                 q_net,
                 q_target_net,
                 policy_net,
                 q_optimizer,
                 policy_optimizer,
                 replayer_capacity=10000,
                 gamma=0.99,
                 tau=0.005,
                 alpha=0.3,
                 batches=1,
                 batch_size=64,
                 K=30,
                 M=30,
                 value_n_particles=16,
                 sample_threshold=10000):
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
        :param sample_threshold:启动训练的最小样本数
        """
        self.env = env
        self.replayer = Replayer(replayer_capacity)
        self.gamma = gamma
        self.alpha = alpha
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

        self.iterations = 0
        self.sample_threshold = sample_threshold

    def update_target_net(self, target_net, evaluate_net, learning_rate=1.):
        """更新目标网络"""
        target_weights = target_net.get_weights()
        evaluate_weights = evaluate_net.get_weights()
        average_weights = [(1. - learning_rate) * t + learning_rate * e
                           for t, e in zip(target_weights, evaluate_weights)]
        target_net.set_weights(average_weights)

    def decide(self, observation, add_noise=False):
        """选取动作和环境交互"""
        action = self.policy_net(observation[np.newaxis, :])
        action = action.numpy()
        if add_noise:
            noise = self.noise()
            action = np.clip(action + noise, self.env.action_space.low, self.env.action_space.high)
        return action[0]

    def learn(self, observation, action, reward, next_observation, done):
        """更新agent"""
        self.iterations += 1
        self.replayer.store(observation, action, reward, next_observation, done)
        if done:
            self.noise.reset()
        if self.iterations > self.sample_threshold:
            for i in range(self.batches):
                observations, actions, rewards, \
                next_observations, dones = self.replayer.sample(self.batch_size)
                # 利用均匀分布采样计算V值的积分形式
                action_from_distribution = np.random.uniform(-1., 1.,  # 从均匀分布采样，计算V值
                                                             (next_observations.shape[0],
                                                              self.value_n_particles,
                                                              np.prod(self.env.action_space.shape))
                                                             )
                next_observations = np.tile(np.expand_dims(next_observations, axis=1),
                                            (1, self.value_n_particles, 1)
                                            )
                next_q_values = 1. / self.alpha * self.q_target_net(next_observations,
                                                                    action_from_distribution
                                                                    )
                next_v_values = tf.reduce_logsumexp(next_q_values, axis=1)
                next_v_values -= tf.math.log(tf.cast(self.value_n_particles, tf.float32))
                next_v_values += np.log(2.) * np.prod(self.env.action_space.shape)
                next_v_values = self.alpha * next_v_values
                # 利用mse loss计算Q网络的梯度并更新
                q_target = rewards[:, np.newaxis] + (1 - dones[:, np.newaxis]) * self.gamma * next_v_values
                with tf.GradientTape() as tape:
                    q_values = self.q_net(observations, actions)
                    q_losses = tf.losses.mse(q_target, q_values)
                    q_loss = tf.nn.compute_average_loss(q_losses)
                grad = tape.gradient(q_loss, self.q_net.trainable_variables)
                self.q_optimizer.apply_gradients(zip(grad, self.q_net.trainable_variables))

                # 使用SVGD更新策略网络
                pi_observations = np.tile(np.expand_dims(observations, axis=1),
                                          (1, self.K + self.M, 1)
                                          )
                with tf.GradientTape() as tape1:
                    actions = self.policy_net(pi_observations)
                    fixed_actions, updated_actions = tf.split(
                        actions, [self.K, self.M], axis=1)  # 采样K+M个action，并划分样本
                fixed_actions = tf.stop_gradient(fixed_actions)

                q_observations = np.tile(np.expand_dims(observations, axis=1),
                                         (1, self.K, 1))
                with tf.GradientTape() as tape2:
                    tape2.watch(fixed_actions)
                    svgd_target_values = self.q_net(q_observations, fixed_actions, )
                    squash_correction = tf.reduce_sum(
                        tf.math.log(1 - fixed_actions ** 2 + EPS), axis=-1, keepdims=True)
                    log_p = svgd_target_values + squash_correction  # 计算Q函数对fixed_actions的梯度
                grad_log_p = tape2.gradient(log_p, fixed_actions)
                grad_log_p = tf.expand_dims(grad_log_p, axis=2)
                grad_log_p = tf.stop_gradient(grad_log_p)

                kernel_dict = self.kernel_fn(xs=fixed_actions, ys=updated_actions)  # 计算kernel

                kappa = tf.expand_dims(kernel_dict["output"], axis=3)
                action_gradients = tf.reduce_mean(
                    kappa * grad_log_p + kernel_dict["gradient"], axis=1)

                gradients = tape1.gradient(updated_actions, self.policy_net.trainable_variables,
                                           action_gradients)  # updated_actions对策略网络的梯度
                with tf.GradientTape() as tape3:            # 根据SVGD得到的更新方向更新策略网络
                    surrogate_loss = tf.reduce_sum([
                        -1. * tf.reduce_sum(w * tf.stop_gradient(g))
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
        self.layer1_observation = keras.layers.Dense(64, 'relu')
        self.layer1_action = keras.layers.Dense(64, 'relu')
        self.layer2 = keras.layers.Dense(64, 'relu')
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
        self.layer1 = keras.layers.Dense(64, activation)
        self.layer1_kesi = keras.layers.Dense(64, activation)
        self.layer2 = keras.layers.Dense(64, activation)
        self.layer3_u = keras.layers.Dense(output_size)
        self.layer3_log_std = keras.layers.Dense(output_size)
        self.output_size = output_size

    def call(self, observations):
        shape = observations.shape
        kesi = tf.random.normal((*shape[0:-1], self.output_size))
        x_obs = self.layer1(observations)
        x_kesi = self.layer1_kesi(kesi)
        x = self.layer2(x_obs+x_kesi)
        u = self.layer3_u(x)
        log_std = self.layer3_log_std(x)
        log_std = tf.clip_by_value(log_std, -20., 2)
        std = tf.exp(log_std)
        noise = tf.random.normal(std.shape)
        action = tf.tanh(u+noise*std)

        return action



def main():
    env = gym.make('MountainCarContinuous-v0')
    q_net = QNet()  # Q网络
    q_target_net = copy.deepcopy(q_net)  # Q目标网络
    policy_net = PolicyNet(env.action_space.shape[0])  # 策略网络
    q_optimizer = tf.optimizers.Adam(1e-3)  # 定义优化器
    policy_optimizer = tf.optimizers.Adam(1e-3)

    rewards_list = []
    random_explore_times = 10000
    iter_times = 0
    skip_frame = 4
    act_prob = 0.3
    agent = SQLAgent(env=env,
                     q_net=q_net,
                     q_target_net=q_target_net,
                     policy_net=policy_net,
                     q_optimizer=q_optimizer,
                     policy_optimizer=policy_optimizer,
                     tau=0.01,
                     batches=1,
                     batch_size=32,
                     sample_threshold=random_explore_times
                     )
    # agent和环境交互，并且进行学习
    print("Begin to train ")
    for i in range(100):
        print("episode: ", i)
        rewards = 0
        observation = env.reset()
        while True:
            iter_times += 1
            action = agent.decide(observation, add_noise=False)
            # action = env.action_space.sample()
            if iter_times < random_explore_times and np.random.random() > act_prob:
                for j in range(skip_frame):  # 积累样本
                    env.step(action)
            next_observation, reward, done, info = env.step(action)
            rewards += reward
            agent.learn(observation, action, reward, next_observation, done)
            observation = next_observation
            if done:
                print("episode rewards:{}".format(rewards))
                if iter_times > random_explore_times:
                    rewards_list.append(rewards)
                break
        if iter_times > random_explore_times and \
                np.mean(rewards_list[-10:]) > 90. and len(rewards_list) > 10:
            break
    # plt.plot(rewards_list)
    # plt.title("episode rewards")
    # plt.xlabel("n episodes")
    # plt.ylabel("rewards")
    # 开始测试
    print("Begin to test ")
    test_rewards_list = []
    for i in range(100):
        observation = env.reset()
        rewards = 0
        while True:
            action = agent.decide(observation)
            observation, reward, done, info = env.step(action)
            rewards += reward
            if done:
                test_rewards_list.append(rewards)
                break
    print("The average reward for 100 episodes: ", np.mean(test_rewards_list))


if __name__ == '__main__':
    main()
