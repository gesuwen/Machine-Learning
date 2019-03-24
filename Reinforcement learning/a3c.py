# Use advantage actor-critic algorithm to play atari KungFuMaster
import matplotlib.pyplot as plt
import numpy as np
import gym
from atari_util import PreprocessAtari
import tensorflow as tf
from keras.layers import Conv2D, Dense, Flatten, Input
from keras.models import Model, Sequential
from tqdm import trange
from pandas import DataFrame


def make_env():
    env = gym.make("KungFuMasterDeterministic-v0")
    env = PreprocessAtari(
        env, height=42, width=42,
        crop=lambda img: img[60:-30, 5:],
        dim_order='tensorflow',
        color=False, n_frames=4,
        reward_scale=reward_scale)
    return env


class Agent:
    def __init__(self, name, state_shape, n_actions, reuse=False):
        """A simple actor-critic agent"""

        with tf.variable_scope(name, reuse=reuse):
            # Prepare neural network architecture
            inputs = Input(shape=state_shape)
            x = Conv2D(32, (3, 3), strides=2, activation='relu')(inputs)
            x = Conv2D(32, (3, 3), strides=2, activation='relu')(x)
            x = Conv2D(32, (3, 3), strides=2, activation='relu')(x)
            x = Flatten()(x)
            x = Dense(128, activation='relu')(x)

            logits = Dense(n_actions, activation='linear')(x)
            state_value = Dense(1, activation='linear')(x)
            self.network = Model(inputs=inputs, output=[logits, state_value])
            # prepare a graph for agent step
            self.state_t = tf.placeholder('float32', [None, ] + list(state_shape))
            self.agent_outputs = self.symbolic_step(self.state_t)

    def symbolic_step(self, state_t):
        """Takes agent's previous step and observation, returns next state and whatever it needs to learn (tf tensors)"""
        # Apply neural network
        logits, state_value = self.network(state_t)
        state_value = state_value[:, 0]

        return (logits, state_value)

    def step(self, state_t):
        """Same as symbolic step except it operates on numpy arrays"""
        sess = tf.get_default_session()
        return sess.run(self.agent_outputs, {self.state_t: state_t})

    def sample_actions(self, agent_outputs):
        """pick actions given numeric agent outputs (np arrays)"""
        logits, state_values = agent_outputs
        policy = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        return np.array([np.random.choice(len(p), p=p) for p in policy])


def evaluate(agent, env, n_games=1):
    """Plays an a game from start till done, returns per-game rewards """
    game_rewards = []
    for _ in range(n_games):
        state = env.reset()
        total_reward = 0
        while True:
            action = agent.sample_actions(agent.step([state]))[0]
            state, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        # We rescale the reward back to ensure compatibility
        # with other evaluations.
        game_rewards.append(total_reward / reward_scale)
    return game_rewards


class EnvBatch:
    def __init__(self, n_envs=10):
        """ Creates n_envs environments and babysits them for ya' """
        self.envs = [make_env() for _ in range(n_envs)]

    def reset(self):
        """ Reset all games and return [n_envs, *obs_shape] observations """
        return np.array([env.reset() for env in self.envs])

    def step(self, actions):
        """
        Send a vector[batch_size] of actions into respective environments
        :returns: observations[n_envs, *obs_shape], rewards[n_envs], done[n_envs,], info[n_envs]
        """
        results = [env.step(a) for env, a in zip(self.envs, actions)]
        new_obs, rewards, done, infos = map(np.array, zip(*results))

        # reset environments automatically
        for i in range(len(self.envs)):
            if done[i]:
                new_obs[i] = self.envs[i].reset()

        return new_obs, rewards, done, infos


if __name__ == "__main__":
    # scale rewards to avoid exploding gradients during optimization.
    reward_scale = 0.01

    env = make_env()

    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    print("Observation shape:", obs_shape)
    print("Num actions:", n_actions)
    print("Action names:", env.env.env.get_action_meanings())

    s = env.reset()
    for _ in range(100):
        s, _, _, _ = env.step(env.action_space.sample())

    plt.title('Game image')
    plt.imshow(env.render('rgb_array'))
    plt.show()

    plt.title('Agent observation (4-frame buffer)')
    plt.imshow(s.transpose([0,2,1]).reshape([42,-1]))
    plt.show()

    # Build an agent
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    agent = Agent("agent", obs_shape, n_actions)
    sess.run(tf.global_variables_initializer())
    state = [env.reset()]
    logits, value = agent.step(state)
    print("\nexample action logits:\n", logits)
    print("\nexample state values:\n", value)

    env_monitor = gym.wrappers.Monitor(env, directory="kungfu_videos", force=True)
    rw = evaluate(agent, env_monitor, n_games=3,)
    env_monitor.close()
    print ("\n reward of 3 games: ", rw)

    # Training on parallel games
    env_batch = EnvBatch(10)
    batch_states = env_batch.reset()
    batch_actions = agent.sample_actions(agent.step(batch_states))
    batch_next_states, batch_rewards, batch_done, _ = env_batch.step(batch_actions)

    print("batch State shape:", batch_states.shape)
    print("batch Actions:", batch_actions[:3])
    print("batch Rewards:", batch_rewards[:3])
    print("batch Done:", batch_done[:3])

    # Actor-critic
    # define a loss functions and learning algorithms
    states_ph = tf.placeholder('float32', [None,] + list(obs_shape))
    next_states_ph = tf.placeholder('float32', [None,] + list(obs_shape))
    actions_ph = tf.placeholder('int32', (None,))
    rewards_ph = tf.placeholder('float32', (None,))
    is_done_ph = tf.placeholder('float32', (None,))

    # logits[n_envs, n_actions] and state_values[n_envs, n_actions]
    logits, state_values = agent.symbolic_step(states_ph)
    next_logits, next_state_values = agent.symbolic_step(next_states_ph)
    next_state_values = next_state_values * (1 - is_done_ph)

    # probabilities and log-probabilities for all actions
    probs = tf.nn.softmax(logits)            # [n_envs, n_actions]
    logprobs = tf.nn.log_softmax(logits)     # [n_envs, n_actions]

    # log-probabilities only for agent's chosen actions
    logp_actions = tf.reduce_sum(logprobs * tf.one_hot(actions_ph, n_actions), axis=-1) # [n_envs,]

    # compute advantage using rewards_ph, state_values and next_state_values
    gamma = 0.99
    advantage = rewards_ph + gamma * next_state_values - state_values

    # compute policy entropy given logits_seq. Mind the "-" sign!
    entropy = -tf.reduce_sum(probs * logprobs, 1, name="entropy")

    actor_loss = - tf.reduce_mean(logp_actions * tf.stop_gradient(advantage)) - 0.001 * tf.reduce_mean(entropy)

    # compute target state values using temporal difference formula. Use rewards_ph and next_step_values
    target_state_values = rewards_ph + gamma * next_state_values

    critic_loss = tf.reduce_mean((state_values - tf.stop_gradient(target_state_values))**2 )

    train_step = tf.train.AdamOptimizer(1e-4).minimize(actor_loss + critic_loss)
    sess.run(tf.global_variables_initializer())

    ewma = lambda x, span=100: DataFrame({'x':np.asarray(x)}).x.ewm(span=span).mean().values

    env_batch = EnvBatch(10)
    batch_states = env_batch.reset()

    rewards_history = []
    entropy_history = []

    for i in trange(100000):
        batch_actions = agent.sample_actions(agent.step(batch_states))
        batch_next_states, batch_rewards, batch_done, _ = env_batch.step(batch_actions)

        feed_dict = {
            states_ph: batch_states,
            actions_ph: batch_actions,
            next_states_ph: batch_next_states,
            rewards_ph: batch_rewards,
            is_done_ph: batch_done,
        }
        batch_states = batch_next_states

        _, ent_t = sess.run([train_step, entropy], feed_dict)
        entropy_history.append(np.mean(ent_t))

        if i % 500 == 0:
            if i % 2500 == 0:
                rewards_history.append(np.mean(evaluate(agent, env, n_games=3)))
                if rewards_history[-1] >= 50:
                    print("Your agent has earned the yellow belt")

            plt.figure(figsize=[8, 4])
            plt.subplot(1, 2, 1)
            plt.plot(rewards_history, label='rewards')
            plt.plot(ewma(np.array(rewards_history), span=10), marker='.', label='rewards ewma@10')
            plt.title("Session rewards")
            plt.grid()
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(entropy_history, label='entropy')
            plt.plot(ewma(np.array(entropy_history), span=1000), label='entropy ewma@1000')
            plt.title("Policy entropy")
            plt.grid()
            plt.legend()
            plt.show()
