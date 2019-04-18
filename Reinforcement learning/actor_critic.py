# Actor Critic algorithm to play OpenAI gym CartPole-v1
import tensorflow as tf
import numpy as np
import gym
tf.set_random_seed(42)


# Policy Gradient
class Policy(object):
    def __init__(self, obssize, actsize, sess, optimizer):
        """
        obssize: size of the states
        actsize: size of the actions
        """
        # BUILD PREDICTION GRAPH
        # build the input
        state = tf.placeholder(tf.float32, [None, obssize])
        hidden_layer = tf.layers.dense(inputs=state,
                                       units=16,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       activation=tf.nn.relu)
        hidden_layer = tf.layers.dense(inputs=hidden_layer,
                                       units=16,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       activation=tf.nn.relu)
        hidden_layer = tf.layers.dense(inputs=hidden_layer,
                                       units=16,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       activation=tf.nn.relu)
        hidden_layer = tf.layers.dense(inputs=hidden_layer,
                                       units=actsize, activation=None)  # prob is of shape [None, actsize]
        prob = tf.nn.softmax(hidden_layer)

        # BUILD LOSS
        Q_estimate = tf.placeholder(tf.float32, [None])
        actions = tf.placeholder(tf.int32, [None])

        # Loss and train op
        surrogate_loss = -tf.reduce_mean(tf.reduce_sum(prob * tf.one_hot(actions, actsize), axis=1) * Q_estimate)

        self.train_op = optimizer.minimize(surrogate_loss)

        # some bookkeeping
        self.state = state
        self.prob = prob
        self.actions = actions
        self.Q_estimate = Q_estimate
        self.loss = surrogate_loss
        self.optimizer = optimizer
        self.sess = sess

    def compute_prob(self, states):
        """
        compute prob over actions given states pi(a|s)
        states: numpy array of size [numsamples, obssize]
        return: numpy array of size [numsamples, actsize]
        """
        return self.sess.run(self.prob, feed_dict={self.state: states})

    def train(self, states, actions, Qs):
        """
        states: numpy array (states)
        actions: numpy array (actions)
        Qs: numpy array (Q values)
        """
        return self.sess.run(self.train_op, feed_dict={self.state: states, self.actions: actions, self.Q_estimate: Qs})


class ValueFunction(object):
    def __init__(self, obssize, sess, optimizer):
        """
        obssize: size of states
        """
        # need to implement both prediction and loss
        states = tf.placeholder(tf.float32, [None, obssize])
        targets = tf.placeholder(tf.float32, [None])
        hidden_layer = tf.layers.dense(inputs=states,
                                       units=16,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       activation=tf.nn.relu)
        hidden_layer = tf.layers.dense(inputs=hidden_layer,
                                       units=16,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       activation=tf.nn.relu)
        hidden_layer = tf.layers.dense(inputs=hidden_layer,
                                       units=16,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       activation=tf.nn.relu)
        value_estimate = tf.layers.dense(inputs=hidden_layer,
                                         units=1, activation=None)

        self.loss = tf.reduce_mean(tf.square(value_estimate - targets))
        self.train_op = optimizer.minimize(
            self.loss, global_step=tf.contrib.framework.get_global_step())
        self.sess = sess
        self.states = states
        self.targets = targets
        self.value_estimate = value_estimate

    def compute_values(self, states):
        """
        compute value function for given states
        states: numpy array of size [numsamples, obssize]
        return: numpy array of size [numsamples]
        """
        return self.sess.run(self.value_estimate, feed_dict={self.states: states})

    def train(self, states, targets):
        """
        states: numpy array
        targets: numpy array
        """
        return self.sess.run(self.train_op, feed_dict={self.states: states, self.targets: targets})


def discounted_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_sum = 0
    for i in reversed(range(0,len(r))):
        discounted_r[i] = running_sum * gamma + r[i]
        running_sum = discounted_r[i]
    return list(discounted_r)


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    print(env.observation_space)  # four observations: horizontal coordinate of car, horizontal velocity of car
                                  # angle of the pole to the vertical line, angular velocity of the pole
    print(env.action_space)  # two actions: push to the right/left

    # parameter initializations
    alpha = 10 ** (-2.5)#3e-2  # learning rate for PG
    beta = 10 ** (-2)#3e-2  # learning rate for baseline
    numtrajs = 20  # num of trajecories to collect at each iteration
    iterations = 2000  # total num of iterations
    envname = "CartPole-v1"  # environment name
    gamma = .999  # discount
    episodes = 1000

    # initialize environment
    env = gym.make(envname)
    obssize = env.observation_space.low.size
    actsize = env.action_space.n

    # sess
    sess = tf.Session()

    # optimizer
    optimizer_p = tf.train.AdamOptimizer(alpha)
    optimizer_v = tf.train.AdamOptimizer(beta)

    # initialize networks
    actor = Policy(obssize, actsize, sess, optimizer_p)  # policy initialization
    baseline = ValueFunction(obssize, sess, optimizer_v)  # baseline initialization

    # initialize tensorflow graphs
    sess.run(tf.global_variables_initializer())

    # main iteration
    for ite in range(episodes):

        # trajs records for batch update
        OBS = []  # observations
        ACTS = []  # actions
        ADS = []  # advantages (to update policy)
        VAL = []  # value functions (to update baseline)

        for num in range(numtrajs):
            # record for each episode
            obss = []  # observations
            acts = []   # actions
            rews = []  # instant rewards

            obs = env.reset()
            done = False

            while not done:

                prob = actor.compute_prob(np.expand_dims(obs,0))
                action = np.random.choice(actsize, p=prob.flatten(), size=1)
                newobs, reward, done, _ = env.step(action[0])

                # record
                obss.append(obs)
                acts.append(action[0])
                rews.append(reward)

                # update
                obs = newobs

            # compute returns from instant rewards
            returns = discounted_rewards(rews, gamma)

            # record for batch update
            VAL += returns
            OBS += obss
            ACTS += acts

        # update baseline
        VAL = np.array(VAL)
        OBS = np.array(OBS)
        ACTS = np.array(ACTS)

        baseline.train(OBS, VAL)  # update only one step

        # update policy
        BAS = baseline.compute_values(OBS)  # compute baseline for variance reduction
        ADS = VAL - np.squeeze(BAS,1)

        actor.train(OBS, ACTS, ADS)  # update only one step

    # after training, we will evaluate the performance of the agent
    # on a target environment
    eval_episodes = 100
    record = []
    env = gym.make('CartPole-v1')
    eval_mode = True
    for ite in range(eval_episodes):

        obs = env.reset()
        done = False
        rsum = 0

        while not done:

            # epsilon greedy for exploration
            if eval_mode:
                p = actor.compute_prob(np.expand_dims(obs,0)).ravel()
                action = np.random.choice(np.arange(2), size=1, p=p)[0]
            else:
                raise NotImplementedError

            newobs, r, done, _ = env.step(action)
            rsum += r
            obs = newobs
        record.append(rsum)
    print("eval performance of PG agent: {}".format(np.mean(record)))
