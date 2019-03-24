# Implementation of a basic reinforce algorithm a.k.a. policy gradient for OpenAI gym CartPole env
import gym
import numpy as np
import tensorflow as tf
import keras
import keras.layers as L
from collections import deque


# Computing cumulative rewards
def get_cumulative_rewards(rewards,  # rewards at each step
                           gamma=0.99  # discount for reward
                           ):
    """
    take a list of immediate rewards r(s,a) for the whole session
    compute cumulative rewards R(s,a) (a.k.a. G(s,a) in Sutton '16)
    R_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...

    The simple way to compute cumulative rewards is to iterate from last to first time tick
    and compute R_t = r_t + gamma*R_{t+1} recurrently
    """

    cumulative_rewards = deque([rewards[-1]])
    for i in range(len(rewards) - 2, -1, -1):
        cumulative_rewards.appendleft(rewards[i] + gamma * cumulative_rewards[0])
    return cumulative_rewards


def train_step(_states,_actions,_rewards):
    """given full session, trains agent with policy gradient"""
    _cumulative_rewards = get_cumulative_rewards(_rewards)
    update.run({states: _states, actions: _actions, cumulative_rewards: _cumulative_rewards})


def generate_session(t_max=1000):
    """play env with REINFORCE agent and train at the session end"""
    # arrays to record session
    states, actions, rewards = [], [], []
    s = env.reset()
    for t in range(t_max):
        # action probabilities array aka pi(a|s)
        action_probas = get_action_proba(s)
        a = np.random.choice(n_actions, 1, p=action_probas)[0]
        new_s, r, done, info = env.step(a)

        # record session history to train later
        states.append(s)
        actions.append(a)
        rewards.append(r)

        s = new_s
        if done:
            break
    train_step(states, actions, rewards)

    return sum(rewards)


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env.reset()
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape

    # Building the policy network

    # create input variables. only need <s,a,R> for REINFORCE
    states = tf.placeholder('float32',(None,)+state_dim, name="states")
    actions = tf.placeholder('int32', name="action_ids")
    cumulative_rewards = tf.placeholder('float32', name="cumulative_returns")

    # network architecture
    network = keras.models.Sequential()
    network.add(L.Dense(32, activation='relu', input_shape=state_dim))
    network.add(L.Dense(32, activation='relu'))
    network.add(L.Dense(n_actions, activation='linear'))

    logits = network(states)

    policy = tf.nn.softmax(logits)
    log_policy = tf.nn.log_softmax(logits)

    # utility function to pick action in one given state
    get_action_proba = lambda s: policy.eval({states: [s]})[0]

    # Loss function and updates
    indices = tf.stack([tf.range(tf.shape(log_policy)[0]),actions],axis=-1)
    log_policy_for_actions = tf.gather_nd(log_policy,indices)

    # policy objective function of the REINFORCE algorithm
    J = tf.reduce_mean(log_policy_for_actions * cumulative_rewards)

    # regularize with entropy
    entropy = -tf.reduce_mean(policy * log_policy, 1, name="entropy")

    # select all network weights for training
    all_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    # weight updates. maximizing J is same as minimizing -J. Adding negative entropy.
    loss = -J - 0.1 * entropy
    update = tf.train.AdamOptimizer().minimize(loss,var_list=all_weights)

    print("\nTraining starts")
    s = tf.InteractiveSession()
    s.run(tf.global_variables_initializer())

    for i in range(100):
        rewards = [generate_session() for _ in range(100)]  # generate new sessions
        print("mean reward:%.3f" % (np.mean(rewards)))

        if np.mean(rewards) > 300:
            print("Mean reward larger than 300, iteration terminates. ")
            break

