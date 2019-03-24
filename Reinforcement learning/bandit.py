from abc import ABCMeta, abstractmethod, abstractproperty
from matplotlib import pyplot as plt
from collections import OrderedDict
import numpy as np
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


# Bernoulli Bandit - The bandit has K actions. Action produce 1.0 reward r with probability 0≤θk≤1 unknown to agent, but
# fixed over time.
class BernoulliBandit:
    def __init__(self, n_actions=5):
        self._probs = np.random.random(n_actions)

    @property
    def action_count(self):
        return len(self._probs)

    def pull(self, action):
        if np.any(np.random.random() > self._probs[action]):
            return 0.0
        return 1.0

    def optimal_reward(self):
        """ Used for regret calculation
        """
        return np.max(self._probs)

    def step(self):
        """ Used in nonstationary version
        """
        pass

    def reset(self):
        """ Used in nonstationary version
        """


class AbstractAgent(metaclass=ABCMeta):
    def init_actions(self, n_actions):
        self._successes = np.zeros(n_actions)
        self._failures = np.zeros(n_actions)
        self._total_pulls = 0

    @abstractmethod
    def get_action(self):
        """
        Get current best action
        :rtype: int
        """
        pass

    def update(self, action, reward):
        """
        Observe reward from action and update agent's internal parameters
        :type action: int
        :type reward: int
        """
        self._total_pulls += 1
        if reward == 1:
            self._successes[action] += 1
        else:
            self._failures[action] += 1

    @property
    def name(self):
        return self.__class__.__name__


class RandomAgent(AbstractAgent):
    def get_action(self):
        return np.random.randint(0, len(self._successes))


# Epsilon-greedy agent
class EpsilonGreedyAgent(AbstractAgent):
    def __init__(self, epsilon=0.01):
        self._epsilon = epsilon

    def get_action(self):
        n_actions = self._successes + self._failures
        p = self._successes / n_actions
        # explore
        if np.random.random() < self._epsilon:
            return np.random.randint(0, len(self._successes))
        # exploit
        else:
            return np.argmax(p)

    @property
    def name(self):
        return self.__class__.__name__ + "(epsilon={})".format(self._epsilon)


# UCB Agent - select among actions that are uncertain or have potential to be optimal
class UCBAgent(AbstractAgent):
    def get_action(self):
        n_actions = self._successes + self._failures
        ucb = np.sqrt(2 * np.log10(self._total_pulls) / n_actions)
        p = self._successes / (n_actions) + ucb
        return np.argmax(p)
    @property
    def name(self):
        return self.__class__.__name__


# Thompson sampling - take into account actual distribution of rewards
class ThompsonSamplingAgent(AbstractAgent):
    def get_action(self):
        p = np.random.beta(self._successes + 1, self._failures + 1)
        return np.argmax(p)

    @property
    def name(self):
        return self.__class__.__name__


def get_regret(env, agents, n_steps=5000, n_trials=50):
    scores = OrderedDict({
        agent.name: [0.0 for step in range(n_steps)] for agent in agents
    })

    for trial in range(n_trials):
        env.reset()
        for a in agents:
            a.init_actions(env.action_count)
        for i in range(n_steps):
            optimal_reward = env.optimal_reward()
            for agent in agents:
                action = agent.get_action()
                reward = env.pull(action)
                agent.update(action, reward)
                scores[agent.name][i] += optimal_reward - reward
            env.step()  # change bandit's state if it is unstationary
    for agent in agents:
        scores[agent.name] = np.cumsum(scores[agent.name]) / n_trials

    return scores


def plot_regret(agents, scores):
    for agent in agents:
        plt.plot(scores[agent.name])

    plt.legend([agent.name for agent in agents])
    plt.ylabel("regret")
    plt.xlabel("steps")
    plt.show()


if __name__ == "__main__":
    agents = [
        EpsilonGreedyAgent(),
        UCBAgent(),
        ThompsonSamplingAgent()
    ]
    regret = get_regret(BernoulliBandit(), agents, n_steps=10000, n_trials=10)
    plot_regret(agents, regret)

