# Build a neural network for approximate Q learning

import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import keras.layers as L


def get_action(state, epsilon=0):
    """
    sample actions with epsilon-greedy policy
    recap: with p = epsilon pick random action, else pick action with highest Q(s,a)
    """

    q_values = network.predict(state[None])[0]
    exploration = np.random.random()
    if exploration < epsilon:
        action = np.random.choice(n_actions, 1)[0]
    else:
        action = np.argmax(q_values)

    return action


def generate_session(t_max=1000, epsilon=0, train=False):
    """play env with approximate q-learning agent and train it at the same time"""
    total_reward = 0
    s = env.reset()

    for t in range(t_max):
        a = get_action(s, epsilon=epsilon)
        next_s, r, done, _ = env.step(a)

        if train:
            sess.run(train_step, {
                states_ph: [s], actions_ph: [a], rewards_ph: [r],
                next_states_ph: [next_s], is_done_ph: [done]
            })

        total_reward += r
        s = next_s
        if done: break

    return total_reward


if __name__ == "__main__":
    # OpenAI gym CartPole-v0 example
    env = gym.make("CartPole-v0").env
    env.reset()
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape

    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    keras.backend.set_session(sess)

    network = keras.models.Sequential()
    network.add(L.InputLayer(state_dim))

    # create a network for approximate q-learning following guidelines above
    network.add(L.Dense(100, activation='relu'))
    network.add(L.Dense(100, activation='relu'))
    network.add(L.Dense(n_actions, activation='linear'))

    # Q-learning via gradient descent - train the agent's Q-function by minimizing the TD loss
    # Create placeholders for the <s, a, r, s'> tuple and a special indicator for game end (is_done = True)
    states_ph = keras.backend.placeholder(dtype='float32', shape=(None,) + state_dim)
    actions_ph = keras.backend.placeholder(dtype='int32', shape=[None])
    rewards_ph = keras.backend.placeholder(dtype='float32', shape=[None])
    next_states_ph = keras.backend.placeholder(dtype='float32', shape=(None,) + state_dim)
    is_done_ph = keras.backend.placeholder(dtype='bool', shape=[None])

    # get q-values for all actions in current states
    predicted_qvalues = network(states_ph)

    # select q-values for chosen actions
    predicted_qvalues_for_actions = tf.reduce_sum(predicted_qvalues * tf.one_hot(actions_ph, n_actions), axis=1)

    gamma = 0.99

    # compute q-values for all actions in next states
    predicted_next_qvalues = network(next_states_ph)

    # compute V * (next_states) using predicted next q-values
    next_state_values = tf.reduce_max(predicted_next_qvalues, axis=1)

    # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
    target_qvalues_for_actions = rewards_ph + gamma * next_state_values

    # at the last state use simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
    target_qvalues_for_actions = tf.where(is_done_ph, rewards_ph, target_qvalues_for_actions)

    # mean squared error loss to minimize
    loss = (predicted_qvalues_for_actions - tf.stop_gradient(target_qvalues_for_actions)) ** 2
    loss = tf.reduce_mean(loss)

    # training function that resembles agent.update(state, action, reward, next_state) from tabular agent
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    epsilon = 0.5

    for i in range(1000):
        session_rewards = [generate_session(epsilon=epsilon, train=True) for _ in range(100)]
        print("epoch #{}\tmean reward = {:.3f}\tepsilon = {:.3f}".format(i, np.mean(session_rewards), epsilon))

        epsilon *= 0.99
        assert epsilon >= 1e-4, "Make sure epsilon is always nonzero during training"

        if np.mean(session_rewards) > 300:
            print("Iteration ended, session mean reward > 300")
            break

