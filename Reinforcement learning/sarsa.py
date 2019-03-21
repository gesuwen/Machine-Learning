# Expected Value SARSA with epsilon-greedy policy

import numpy as np
import matplotlib.pyplot as plt
from qlearning import *
import gym, gym.envs.toy_text
from pandas import DataFrame


class EVSarsaAgent(QLearningAgent):

    def get_value(self, state):
        """
        Returns Vpi for current state under epsilon-greedy policy:
          V_{pi}(s) = sum _{over a_i} {pi(a_i | s) * Q(s, a_i)}

        """
        epsilon = self.epsilon
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0

        state_value = 0
        for action in possible_actions:
            if action == self.get_best_action(state):
                state_value += ((1 - epsilon) + epsilon / len(possible_actions)) * self.get_qvalue(state, action)
            else:
                state_value += epsilon / len(possible_actions) * self.get_qvalue(state, action)

        return state_value


def play_and_train(env, agent, t_max=10 ** 4):

    total_reward = 0.0
    s = env.reset()

    for t in range(t_max):
        a = agent.get_action(s)

        next_s, r, done, _ = env.step(a)
        agent.update(s, a, r, next_s)

        s = next_s
        total_reward += r
        if done: break

    return total_reward


def draw_policy(env, agent):
    """ Prints CliffWalkingEnv policy with arrows. Hard-coded. """
    n_rows, n_cols = env._cliff.shape

    actions = '^>v<'

    for yi in range(n_rows):
        for xi in range(n_cols):
            if env._cliff[yi, xi]:
                print(" C ", end='')
            elif (yi * n_cols + xi) == env.start_state_index:
                print(" X ", end='')
            elif (yi * n_cols + xi) == n_rows * n_cols - 1:
                print(" T ", end='')
            else:
                print(" %s " % actions[agent.get_best_action(yi * n_cols + xi)], end='')
        print()


if __name__ == '__main__':
    # Cliff World environment
    env = gym.envs.toy_text.CliffWalkingEnv()
    n_actions = env.action_space.n

    print(env.__doc__)
    print("\n Visualize cliff wall")
    env.render()


    agent_sarsa = EVSarsaAgent(alpha=0.25, epsilon=0.2, discount=0.99,
                           get_legal_actions = lambda s: range(n_actions))

    agent_ql = QLearningAgent(alpha=0.25, epsilon=0.2, discount=0.99,
                           get_legal_actions = lambda s: range(n_actions))

    moving_average = lambda x, span=100: DataFrame({'x': np.asarray(x)}).x.ewm(span=span).mean().values

    rewards_sarsa, rewards_ql = [], []

    print("Comparison between Q learning and SARSA")
    for i in range(5000):
        rewards_sarsa.append(play_and_train(env, agent_sarsa))
        rewards_ql.append(play_and_train(env, agent_ql))
        # Note: agent.epsilon stays constant

        if i % 100 == 0:
            print('EVSARSA mean reward =', np.mean(rewards_sarsa[-100:]))
            print('QLEARNING mean reward =', np.mean(rewards_ql[-100:]))
            plt.title("epsilon = %s" % agent_ql.epsilon)
            plt.plot(moving_average(rewards_sarsa), label='ev_sarsa')
            plt.plot(moving_average(rewards_ql), label='qlearning')
            plt.grid()
            plt.legend()
            plt.ylim(-500, 0)
            plt.show()

    print("\n Visualize the final policy:")
    print("Q-Learning")
    draw_policy(env, agent_ql)

    print("SARSA")
    draw_policy(env, agent_sarsa)
