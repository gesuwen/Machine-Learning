import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def generate_session(policy, t_max=10 ** 4):
    """
    Play game until end or for t_max ticks.
    :param policy: an array of shape [n_states,n_actions] with action probabilities
    :returns: list of states, list of actions and sum of rewards
    """
    states, actions = [], []
    total_reward = 0.

    s = env.reset()

    for t in range(t_max):

        a = np.random.choice(n_actions, 1, p=policy[s])[0]

        new_s, r, done, info = env.step(a)

        # Record state, action and add up reward to states,actions and total_reward accordingly.
        states.append(s)
        actions.append(a)
        total_reward += r

        s = new_s
        if done:
            break
    return states, actions, total_reward


# Cross entropy method
def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    """
    Select states and actions from games that have rewards >= percentile
    :param states_batch: list of lists of states, states_batch[session_i][t]
    :param actions_batch: list of lists of actions, actions_batch[session_i][t]
    :param rewards_batch: list of rewards, rewards_batch[session_i][t]

    :returns: elite_states,elite_actions, both 1D lists of states and respective actions from elite sessions

    Please return elite states and actions in their original order
    [i.e. sorted by session number and timestep within session]

    If you're confused, see examples below. Please don't assume that states are integers (they'll get different later).
    """
    reward_threshold = np.percentile(rewards_batch, percentile)
    elite_states = [s for i in range(len(states_batch)) if rewards_batch[i] >= reward_threshold for s in
                    states_batch[i]]
    elite_actions = [a for i in range(len(actions_batch)) if rewards_batch[i] >= reward_threshold for a in
                     actions_batch[i]]

    return elite_states, elite_actions


def update_policy(elite_states, elite_actions):
    """
    Given old policy and a list of elite states/actions from select_elites,
    return new updated policy where each action probability is proportional to

    policy[s_i,a_i] ~ #[occurences of si and ai in elite states/actions]

    :param elite_states: 1D list of states from elite sessions
    :param elite_actions: 1D list of actions from elite sessions

    """

    new_policy = np.zeros([n_states, n_actions])

    state_action_dict = defaultdict(lambda: [0] * n_actions)
    for i in range(len(elite_states)):
        state_action_dict[elite_states[i]][elite_actions[i]] += 1
    for i in range(n_states):
        if i not in state_action_dict:
            new_policy[i] = [1 / n_actions] * n_actions
        else:
            new_policy[i] = [p / sum(state_action_dict[i]) for p in state_action_dict[i]]

    return new_policy


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
    # test gym
    env = gym.make("Taxi-v2")
    env.reset()
    env.render()

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    print("n_states=%i, n_actions=%i"%(n_states, n_actions))

    # Create stochastic policy
    # initialize policy uniformly
    policy = np.ones((n_states, n_actions)) / n_actions


    # Play the game with uniform policy
    s,a,r = generate_session(policy)

    # initial reward distribution for uniform policy
    sample_rewards = [generate_session(policy,t_max=1000)[-1] for _ in range(200)]
    print("\nInitial reward distribution for uniform policy")
    plt.hist(sample_rewards,bins=20)
    plt.vlines([np.percentile(sample_rewards, 50)], [0], [100], label="50'th percentile", color='green')
    plt.vlines([np.percentile(sample_rewards, 90)], [0], [100], label="90'th percentile", color='red')
    plt.legend()

    n_sessions = 250  # sample this many sessions
    percentile = 50  # take this percent of session with highest rewards
    learning_rate = 0.5  # add this thing to all counts for stability

    log = []

    for i in range(100):
        sessions = [generate_session(policy) for _ in range(n_sessions)]
        batch_states, batch_actions, batch_rewards = zip(*sessions)
        elite_states, elite_actions = select_elites(batch_states, batch_actions, batch_rewards, percentile)
        new_policy = update_policy(elite_states, elite_actions)
        policy = learning_rate * new_policy + (1 - learning_rate) * policy

        if i % 25 == 0:
            print("Iteration "+str(i))
            # display results on chart
            show_progress(batch_rewards, log, percentile)
