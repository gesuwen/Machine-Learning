# implementation of DQN with experience replay and target networks to play Atari breakout

import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gym.core import ObservationWrapper
from gym.spaces import Box
import cv2
import gym
from framebuffer import FrameBuffer
from replay_buffer import ReplayBuffer
import tensorflow as tf
from keras.layers import Conv2D, Dense, Flatten
import keras
from tqdm import trange
from pandas import DataFrame


# Processing game image
class PreprocessAtari(ObservationWrapper):
    def __init__(self, env):
        """A gym wrapper that crops, scales image into the desired shapes and optionally grayscales it."""
        ObservationWrapper.__init__(self, env)

        self.img_size = (64, 64)
        self.observation_space = Box(0.0, 1.0, (self.img_size[0], self.img_size[1], 1))

    def _observation(self, img):
        """what happens to each observation"""
        # crop image (top and bottom, top from 34, bottom remove last 16)
        img = img[34:-16, :, :]
        # resize image
        img = cv2.resize(img, self.img_size)
        # grayscale
        img = img.mean(-1, keepdims=True)
        # convert pixels to range (0,1)
        img = img.astype('float32') / 255.

        return img


class DQNAgent:
    def __init__(self, name, state_shape, n_actions, epsilon=0, reuse=False):
        """A simple DQN agent"""
        with tf.variable_scope(name, reuse=reuse):
            self.network = keras.models.Sequential()
            self.network.add(Conv2D(16, (3, 3), strides=2, activation='relu', input_shape=state_shape))
            self.network.add(Conv2D(32, (3, 3), strides=2, activation='relu'))
            self.network.add(Conv2D(64, (3, 3), strides=2, activation='relu'))
            self.network.add(Flatten())
            self.network.add(Dense(256, activation='relu'))
            self.network.add(Dense(n_actions, activation='linear'))

            # prepare a graph for agent step
            self.state_t = tf.placeholder('float32', [None, ] + list(state_shape))
            self.qvalues_t = self.get_symbolic_qvalues(self.state_t)

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        self.epsilon = epsilon

    def get_symbolic_qvalues(self, state_t):
        """takes agent's observation, returns qvalues. Both are tf Tensors"""
        qvalues = self.network(state_t)

        assert tf.is_numeric_tensor(qvalues) and qvalues.shape.ndims == 2, \
            "please return 2d tf tensor of qvalues [you got %s]" % repr(qvalues)
        assert int(qvalues.shape[1]) == n_actions

        return qvalues

    def get_qvalues(self, state_t):
        """Same as symbolic step except it operates on numpy arrays"""
        sess = tf.get_default_session()
        return sess.run(self.qvalues_t, {self.state_t: state_t})

    def sample_actions(self, qvalues):
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape
        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)
        should_explore = np.random.choice([0, 1], batch_size, p=[1 - epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)


def make_env():
    env = gym.make("BreakoutDeterministic-v4")
    env = PreprocessAtari(env)
    env = FrameBuffer(env, n_frames=4, dim_order='tensorflow')
    return env


def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards = []
    for _ in range(n_games):
        s = env.reset()
        reward = 0
        for _ in range(t_max):
            qvalues = agent.get_qvalues([s])
            action = qvalues.argmax(axis=-1)[0] if greedy else agent.sample_actions(qvalues)[0]
            s, r, done, _ = env.step(action)
            reward += r
            if done: break

        rewards.append(reward)
    return np.mean(rewards)


def play_and_record(agent, env, exp_replay, n_steps=1):
    """
    Play the game for exactly n steps, record every (s,a,r,s', done) to replay buffer.
    Whenever game ends, add record with done=True and reset the game.
    :returns: return sum of rewards over time
    """
    # State at the beginning of rollout
    s = env.framebuffer

    # Play the game for n_steps as per instructions above
    reward = 0.0
    for t in range(n_steps):
        # get agent to pick action given state s
        qvalues = agent.get_qvalues([s])
        action = agent.sample_actions(qvalues)[0]
        next_s, r, done, _ = env.step(action)

        # add to replay buffer
        exp_replay.add(s, action, r, next_s, done)
        reward += r
        if done:
            s = env.reset()
        else:
            s = next_s
    return reward


def load_weigths_into_target_network(agent, target_network):
    """ assign target_network.weights variables to their respective agent.weights values. """
    assigns = []
    for w_agent, w_target in zip(agent.weights, target_network.weights):
        assigns.append(tf.assign(w_target, w_agent, validate_shape=True))
    tf.get_default_session().run(assigns)


def sample_batch(exp_replay, batch_size):
    obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = exp_replay.sample(batch_size)
    return {
        obs_ph:obs_batch, actions_ph:act_batch, rewards_ph:reward_batch,
        next_obs_ph:next_obs_batch, is_done_ph:is_done_batch
    }


if __name__ == "__main__":
    # a buffer that stores 4 last images
    print("\na buffer that stores 4 last images")
    env = make_env()
    env.reset()
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape

    for _ in range(50):
        obs, _, _, _ = env.step(env.action_space.sample())

    plt.title("Game image")
    plt.imshow(env.render("rgb_array"))
    plt.show()
    plt.title("Agent observation (4 frames left to right)")
    plt.imshow(obs.transpose([0,2,1]).reshape([state_dim[0],-1]))

    # Building a network
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    agent = DQNAgent("dqn_agent", state_dim, n_actions, epsilon=0.5)
    sess.run(tf.global_variables_initializer())

    target_network = DQNAgent("target_network", state_dim, n_actions)

    # placeholders that will be fed with exp_replay.sample(batch_size)
    obs_ph = tf.placeholder(tf.float32, shape=(None,) + state_dim)
    actions_ph = tf.placeholder(tf.int32, shape=[None])
    rewards_ph = tf.placeholder(tf.float32, shape=[None])
    next_obs_ph = tf.placeholder(tf.float32, shape=(None,) + state_dim)
    is_done_ph = tf.placeholder(tf.float32, shape=[None])

    is_not_done = 1 - is_done_ph
    gamma = 0.99

    # Take q-values for actions agent
    current_qvalues = agent.get_symbolic_qvalues(obs_ph)
    current_action_qvalues = tf.reduce_sum(tf.one_hot(actions_ph, n_actions) * current_qvalues, axis=1)

    # compute q-values for NEXT states with target network
    next_qvalues_target = target_network.get_symbolic_qvalues(next_obs_ph)

    # compute state values by taking max over next_qvalues_target for all actions
    next_state_values_target = tf.reduce_max(next_qvalues_target, axis=-1)

    # compute Q_reference(s,a) as per formula above.
    reference_qvalues = rewards_ph + gamma * next_state_values_target * is_not_done

    # Define loss function for sgd.
    td_loss = (current_action_qvalues - reference_qvalues) ** 2
    td_loss = tf.reduce_mean(td_loss)

    train_step = tf.train.AdamOptimizer(1e-3).minimize(td_loss, var_list=agent.weights)

    sess.run(tf.global_variables_initializer())

    # Main loop
    moving_average = lambda x, span, **kw: DataFrame({'x':np.asarray(x)}).x.ewm(span=span, **kw).mean().values

    mean_rw_history = []
    td_loss_history = []

    exp_replay = ReplayBuffer(10**5)
    play_and_record(agent, env, exp_replay, n_steps=10000)

    for i in trange(10 ** 5):
        # play
        play_and_record(agent, env, exp_replay, 10)
        # train
        _, loss_t = sess.run([train_step, td_loss], sample_batch(exp_replay, batch_size=64))
        td_loss_history.append(loss_t)
        # adjust agent parameters
        if i % 500 == 0:
            load_weigths_into_target_network(agent, target_network)
            agent.epsilon = max(agent.epsilon * 0.99, 0.01)
            mean_rw_history.append(evaluate(make_env(), agent, n_games=3))

        if i % 100 == 0:
            print("buffer size = %i, epsilon = %.5f" % (len(exp_replay), agent.epsilon))

            plt.subplot(1, 2, 1)
            plt.title("mean reward per game")
            plt.plot(mean_rw_history)
            plt.grid()

            assert not np.isnan(loss_t)
            plt.figure(figsize=[12, 4])
            plt.subplot(1, 2, 2)
            plt.title("TD loss history (moving average)")
            plt.plot(moving_average(np.array(td_loss_history), span=100, min_periods=100))
            plt.grid()
            plt.show()

    assert np.mean(mean_rw_history[-10:]) > 10.
    print("\nlast 10 steps of mean reward > 10. Program finished. ")
