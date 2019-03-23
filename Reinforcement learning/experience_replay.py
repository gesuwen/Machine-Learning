# Use experience replay to train Q-learning and EV-SARSA

import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import gym
from qlearning import *
from pandas import DataFrame


class ReplayBuffer(object):
    def __init__(self, size):
        """
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = deque([])
        self._maxsize = size


    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        # add data to storage
        self._storage.append(data)
        if self.__len__() > self._maxsize:
            self._storage.popleft()

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.choice(range(self.__len__())) for _ in range(batch_size)]

        # collect <s,a,r,s',done> for each index
        obs_batch = []
        act_batch = []
        rew_batch = []
        next_obs_batch = []
        done_mask = []

        for idx in idxes:
            obs_batch.append(self._storage[idx][0])
            act_batch.append(self._storage[idx][1])
            rew_batch.append(self._storage[idx][2])
            next_obs_batch.append(self._storage[idx][3])
            done_mask.append(self._storage[idx][4])

        return np.array(obs_batch), np.array(act_batch), np.array(rew_batch), np.array(next_obs_batch), np.array(
            done_mask)


def play_and_train_with_replay(env, agent, replay=None,
                               t_max=10 ** 4, replay_batch_size=32):
    """
    :param replay: ReplayBuffer where agent can store and sample (s,a,r,s',done) tuples.
        If None, do not use experience replay
    """
    total_reward = 0.0
    s = env.reset()

    for t in range(t_max):
        # get agent to pick action given state s
        a = agent.get_action(s)

        next_s, r, done, _ = env.step(a)

        # update agent on current transition. Use agent.update
        agent.update(s, a, r, next_s)

        if replay is not None:
            # store current <s,a,r,s'> transition in buffer
            replay.add(s, a, r, next_s, done)

            # sample replay_batch_size random transitions from replay,
            # then update agent on each of them in a loop
            s_batch, a_batch, r_batch, next_s_batch, done_batch = replay.sample(replay_batch_size)
            for i in range(replay_batch_size):
                agent.update(s_batch[i], a_batch[i], r_batch[i], next_s_batch[i])

        s = next_s
        total_reward += r
        if done:
            break

    return total_reward


# OpenAI gym Taxi-v2 example
print("\nOpenAI gym Taxi-v2 example")
env = gym.make("Taxi-v2")
n_actions = env.action_space.n

# Create two agents: first will use experience replay, second will not.
print("\nTwo agents: first will use experience replay, second will not.")
agent_baseline = QLearningAgent(alpha=0.5, epsilon=0.25, discount=0.99,
                       get_legal_actions = lambda s: range(n_actions))

agent_replay = QLearningAgent(alpha=0.5, epsilon=0.25, discount=0.99,
                       get_legal_actions = lambda s: range(n_actions))

print("\n Replay buffer has size 1000")
replay = ReplayBuffer(1000)

moving_average = lambda x, span=100: DataFrame({'x': np.asarray(x)}).x.ewm(span=span).mean().values
rewards_replay, rewards_baseline = [], []

for i in range(1000):
    rewards_replay.append(play_and_train_with_replay(env, agent_replay, replay))
    rewards_baseline.append(play_and_train_with_replay(env, agent_baseline, replay=None))

    agent_replay.epsilon *= 0.99
    agent_baseline.epsilon *= 0.99

    if i % 100 == 0:
        print('Baseline : eps =', agent_replay.epsilon, 'mean reward =', np.mean(rewards_baseline[-10:]))
        print('ExpReplay: eps =', agent_baseline.epsilon, 'mean reward =', np.mean(rewards_replay[-10:]))
        plt.plot(moving_average(rewards_replay), label='exp. replay')
        plt.plot(moving_average(rewards_baseline), label='baseline')
        plt.grid()
        plt.legend()
        plt.show()
