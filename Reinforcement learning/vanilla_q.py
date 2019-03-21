# implementation of vanilla Q-learning algorithm

import matplotlib.pyplot as plt
from qlearning import *
import gym
from gym.core import ObservationWrapper


def play_and_train(env, agent, t_max=10 ** 4):
    """
    - run a full game, actions given by agent's e-greedy policy
    - train agent using agent.update(...) whenever it is possible
    - return total reward
    """
    total_reward = 0.0
    s = env.reset()

    for t in range(t_max):
        # get agent to pick action given state s.
        a = agent.get_action(s)

        next_s, r, done, _ = env.step(a)

        # train (update) agent for state s
        agent.update(s, a, r, next_s)

        s = next_s
        total_reward += r
        if done:
            break

    return total_reward


class Binarizer(ObservationWrapper):

    def _observation(self, state):
        state[0] = np.round(state[0], 0)
        state[0] = np.round(state[1], 1)
        state[0] = np.round(state[2], 2)
        state[0] = np.round(state[3], 1)
        return tuple(state)


if __name__ == '__main__':
    # Taxi-v2 environment from OpenAI gym
    print("\nTaxi-v2 environment from OpenAI gym")
    env = gym.make("Taxi-v2")
    n_actions = env.action_space.n

    agent = QLearningAgent(alpha=0.5, epsilon=0.25, discount=0.99,
                           get_legal_actions = lambda s: range(n_actions))

    rewards = []
    for i in range(1000):
        rewards.append(play_and_train(env, agent))
        agent.epsilon *= 0.99

        if i % 100 == 0:
            print("iter ", i, 'eps =', agent.epsilon, 'mean reward =', np.mean(rewards[-10:]))
            plt.plot(rewards)
            plt.show()

    # Binarized state spaces - OpenAI gym CartPole-v0 env. This environment has a continuous set of possible states, so
    # need to group them into bins.
    print("\n OpenAI gym CartPole-v0")
    env = gym.make("CartPole-v0")
    n_actions = env.action_space.n

    print("first state:%s" % (env.reset()))

    # play a few games and record all states
    print("\n play a few games and record all states")
    all_states = []
    for _ in range(1000):
        all_states.append(env.reset())
        done = False
        while not done:
            s, r, done, _ = env.step(env.action_space.sample())
            all_states.append(s)
            if done:
                break

    all_states = np.array(all_states)

    for obs_i in range(env.observation_space.shape[0]):
        plt.hist(all_states[:, obs_i], bins=20)
        plt.show()

    # Binarize environment
    env = Binarizer(gym.make("CartPole-v0"))

    all_states = []
    for _ in range(1000):
        all_states.append(env.reset())
        done = False
        while not done:
            s, r, done, _ = env.step(env.action_space.sample())
            all_states.append(s)
            if done:
                break

    all_states = np.array(all_states)

    for obs_i in range(env.observation_space.shape[0]):
        plt.hist(all_states[:, obs_i], bins=20)
        plt.show()

    # Learn binarized policy
    print("\n Learn binarized policy")
    agent = QLearningAgent(alpha=0.5, epsilon=0.25, discount=0.99,
                           get_legal_actions = lambda s: range(n_actions))

    rewards = []
    for i in range(1000):
        rewards.append(play_and_train(env, agent))
        agent.epsilon *= 0.99
        if i % 100 == 0:
            print("iter ", i, 'eps =', agent.epsilon, 'mean reward =', np.mean(rewards[-10:]))
            plt.plot(rewards)
            plt.show()

