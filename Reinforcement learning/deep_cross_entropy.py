# Approximate cross entropy with deep neural network
import gym
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier


def generate_session(t_max=1000):
    states, actions = [], []
    total_reward = 0

    s = env.reset()

    for t in range(t_max):

        # a vector of action probabilities in current state
        probs = agent.predict_proba([s])[0]

        a = np.random.choice(2, 1, p=probs)[0]

        new_s, r, done, info = env.step(a)

        # record sessions like you did before
        states.append(s)
        actions.append(a)
        total_reward += r

        s = new_s
        if done: break
    return states, actions, total_reward


def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    """
    Select states and actions from games that have rewards >= percentile
    :param states_batch: list of lists of states, states_batch[session_i][t]
    :param actions_batch: list of lists of actions, actions_batch[session_i][t]
    :param rewards_batch: list of rewards, rewards_batch[session_i][t]

    :returns: elite_states,elite_actions, both 1D lists of states and respective actions from elite sessions
    """

    reward_threshold = np.percentile(rewards_batch, percentile)

    elite_states = [s for i in range(len(states_batch)) if rewards_batch[i] >= reward_threshold for s in
                    states_batch[i]]
    elite_actions = [a for i in range(len(actions_batch)) if rewards_batch[i] >= reward_threshold for a in
                     actions_batch[i]]

    return elite_states, elite_actions


def show_progress(batch_rewards, log, percentile, reward_range=[-990, +10]):
    """
    A convenience function that displays training progress.
    No cool math here, just charts.
    """

    mean_reward, threshold = np.mean(batch_rewards), np.percentile(batch_rewards, percentile)
    log.append([mean_reward, threshold])

    print("mean reward = %.3f, threshold=%.3f" % (mean_reward, threshold))
    plt.figure(figsize=[8, 4])
    plt.subplot(1, 2, 1)
    plt.plot(list(zip(*log))[0], label='Mean rewards')
    plt.plot(list(zip(*log))[1], label='Reward thresholds')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.hist(batch_rewards, range=reward_range);
    plt.vlines([np.percentile(batch_rewards, percentile)], [0], [100], label="percentile", color='red')
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    # Initialize the environment
    env = gym.make("CartPole-v0").env
    env.reset()
    n_actions = env.action_space.n

    # create agent
    agent = MLPClassifier(hidden_layer_sizes=(20,20),
                          activation='tanh',
                          warm_start=True, # keep progress between .fit(...) calls
                          max_iter=1 # make only 1 iteration on each .fit(...)
                         )
    # initialize agent to the dimension of state an amount of actions
    agent.fit([env.reset()]*n_actions, list(range(n_actions)));

    # Training loop
    n_sessions = 100
    percentile = 70
    log = []

    for i in range(100):
        # generate new sessions
        sessions = [generate_session() for _ in range(n_sessions)]

        batch_states, batch_actions, batch_rewards = map(np.array, zip(*sessions))

        elite_states, elite_actions = select_elites(batch_states, batch_actions, batch_rewards, percentile)

        agent.fit(elite_states, elite_actions)
        if i % 25 == 0:
            print("\n Iteration" + str(i))
            show_progress(batch_rewards, log, percentile, reward_range=[0, np.max(batch_rewards)])

        if np.mean(batch_rewards) > 190:
            print("batch reward is larger than 190, program terminate.")
            break

