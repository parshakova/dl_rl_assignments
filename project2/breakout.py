
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import random
from breakout_env import *
from wait import *
import tensorflow as tf

tf.flags.DEFINE_boolean("restore", False, "Print scores performance")
FLAGS = tf.flags.FLAGS

env = breakout_environment(nx = 5, ny = 8, nb = 3, nt = 1, nf = 2)

max_steps = 200      # max number of steps to run in each episode
gamma = 0.99
batch_size = 30
replay_size = 1000
replay_start = 100
n_episodes = 2500 + replay_start       # number of episodes to run, 1 for continuing task
target_update = 100

class epsilon_profile: pass
epsilon_profile = epsilon_profile()
epsilon_profile.init = 1.    # initial epsilon in e-greedy
epsilon_profile.final = 0.1   # final epsilon in e-greedy
epsilon_profile.dec_episode = 1. / 500  # amount of decrement in each episode
epsilon_profile.dec_step = 0.   # amount of decrement in each step

sess = tf.InteractiveSession()


class DQN:
    def __init3__(self, env):
        # state size [ny, nx, nf]
        self.s = tf.placeholder(tf.float32, shape=[None, env.ny, env.nx, env.nf])
        self.a = tf.placeholder(tf.int32, shape=[None])
        self.batch_size = tf.placeholder(tf.int32, shape=[])
        self.bootstraped = tf.placeholder(tf.float32, shape=[None])
        self.ch1 = 80
        self.env = env
        # network parameters
        W_fc1 = tf.Variable(tf.truncated_normal([env.nx*env.ny*env.nf, self.ch1], stddev=0.1))
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[self.ch1]))
        W_fc2 = tf.Variable(tf.truncated_normal([self.ch1, env.na], stddev=0.1))
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[env.na]))
        # Q network 
        h_flat = tf.reshape(self.s, [self.batch_size, -1])
        fc1 = tf.matmul(h_flat, W_fc1) + b_fc1
        self.q = tf.matmul(tf.nn.relu(fc1), W_fc2) + b_fc2
        # find Q values for particular actions
        action_one_hot = tf.one_hot(self.a, env.na, 1.0, 0.0, name='action_one_hot')
        self.q_a = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')

        self.optim = tf.train.RMSPropOptimizer(0.0002, momentum=0.95, epsilon=0.01) 
        self.theta = [W_fc1, b_fc1, W_fc2, b_fc2]
        self.target_init()
        # minimize MSE between bootstraped value of target network and apporximated value of
        # behaviour network
        losses = tf.squared_difference(self.bootstraped, self.q_a)
        self.loss = tf.reduce_mean(losses)
        self.train_step = self.optim.minimize(self.loss)


    def __init2__(self, env):
        # state size [ny, nx, nf]
        self.s = tf.placeholder(tf.float32, shape=[None, env.ny, env.nx, env.nf])
        self.a = tf.placeholder(tf.int32, shape=[None])
        self.batch_size = tf.placeholder(tf.int32, shape=[])
        self.bootstraped = tf.placeholder(tf.float32, shape=[None])
        self.ch1, self.ch2 = 80, 150
        self.env = env

        W_conv1 = tf.Variable(tf.truncated_normal([2,1, env.nf, self.ch1], stddev=0.1))
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[self.ch1]))
        W_fc1 = tf.Variable(tf.truncated_normal([7*5*self.ch1, self.ch2], stddev=0.1))
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[self.ch2]))
        W_fc2 = tf.Variable(tf.truncated_normal([self.ch2, env.na], stddev=0.1))
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[env.na]))

        h_conv = tf.nn.conv2d(self.s, W_conv1, strides=[1, 1, 1, 1], padding='VALID')
        h_relu = tf.nn.relu(h_conv + b_conv1)
        h_flat = tf.reshape(h_relu, [self.batch_size, -1])
        fc1 = tf.matmul(h_flat, W_fc1) + b_fc1
        self.q = tf.matmul(tf.nn.relu(fc1), W_fc2) + b_fc2
        
        action_one_hot = tf.one_hot(self.a, env.na, 1.0, 0.0, name='action_one_hot')
        self.q_a = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')

        self.optim = tf.train.RMSPropOptimizer(0.00025, momentum=0.95, epsilon=0.01) #tf.train.AdamOptimizer(0.0001)
        self.theta = [W_conv1, b_conv1, W_fc1, b_fc1, W_fc2, b_fc2]
        self.target_init()

        losses = tf.squared_difference(self.bootstraped, self.q_a)
        self.loss = tf.reduce_mean(losses)
        self.train_step = self.optim.minimize(self.loss)

    def __init__(self, env):
        # state size [ny, nx, nf]
        self.s = tf.placeholder(tf.float32, shape=[None, env.ny, env.nx, env.nf])
        self.a = tf.placeholder(tf.int32, shape=[None])
        self.batch_size = tf.placeholder(tf.int32, shape=[])
        self.bootstraped = tf.placeholder(tf.float32, shape=[None])
        self.ch1, self.ch2, self.ch3 = 30, 30, 128
        self.env = env

        W_conv1 = tf.Variable(tf.truncated_normal([3, 3, env.nf, self.ch1], stddev=0.1))
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[self.ch1]))
        W_conv2 = tf.Variable(tf.truncated_normal([3, 3, self.ch1, self.ch2], stddev=0.1))
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[self.ch2]))
        W_fc1 = tf.Variable(tf.truncated_normal([env.ny * env.nx *self.ch2, self.ch3], stddev=0.1))
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[self.ch3]))
        W_fc2 = tf.Variable(tf.truncated_normal([self.ch3, env.na], stddev=0.1))
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[env.na]))
        h_conv = tf.nn.conv2d(self.s, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
        h_relu = tf.nn.relu(h_conv + b_conv1)
        h_conv = tf.nn.conv2d(h_relu, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        h_relu = tf.nn.relu(h_conv + b_conv2) 
        h_flat = tf.reshape(h_relu, [self.batch_size, -1])
        
        fc1 = tf.matmul(h_flat, W_fc1) + b_fc1
        self.q = tf.matmul(tf.nn.relu(fc1), W_fc2) + b_fc2
        
        action_one_hot = tf.one_hot(self.a, env.na, 1.0, 0.0, name='action_one_hot')
        self.q_a = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')

        self.optim = tf.train.AdamOptimizer(0.00025)  #tf.train.RMSPropOptimizer(0.00025, momentum=0.95, epsilon=0.01) 
        self.theta = [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2]
        self.target_init()

        losses = tf.squared_difference(self.bootstraped, self.q_a)
        self.loss = tf.reduce_mean(losses)
        self.train_step = self.optim.minimize(self.loss)

    def copy_parameters(self):
        for i in range(len(self.theta)):
            self.theta_[i].assign(self.theta[i]).eval()
        #print(self.theta_)

    def target_init(self):
        self.theta_ = []
        for param in self.theta:
            self.theta_ += [tf.Variable(tf.truncated_normal(tf.shape(param), stddev=0.1))]
        W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2 = self.theta_

        h_conv = tf.nn.conv2d(self.s, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
        h_relu = tf.nn.relu(h_conv + b_conv1)
        h_conv = tf.nn.conv2d(h_relu, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        h_relu = tf.nn.relu(h_conv + b_conv2) 
        h_flat = tf.reshape(h_relu, [self.batch_size, -1])
        fc1 = tf.matmul(h_flat, W_fc1) + b_fc1
        self.q_ = tf.matmul(tf.nn.relu(fc1), W_fc2) + b_fc2
        self.q_max_ = tf.reduce_max(self.q_, axis=1)

    def train_iter(self, s, a, bootstraped):
        loss = self.loss.eval(feed_dict={self.s: s, self.a: a, self.batch_size: s.shape[0], self.bootstraped:bootstraped})
        self.train_step.run(feed_dict={self.s: s, self.a: a, self.batch_size: s.shape[0], self.bootstraped:bootstraped})
        return loss

    def target_q(self, s):
        return self.q_max_.eval(feed_dict={self.s: s, self.batch_size: s.shape[0]})

    def behav_q(self, s):
        return self.q.eval(feed_dict={self.s: np.expand_dims(s, axis=0), self.batch_size: 1})


agent = DQN(env)
tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()).run()
agent.copy_parameters()

saver = tf.train.Saver()
epsilon = epsilon_profile.init
losses, ep_r, ep_st, replay = [], [], [], []

if not FLAGS.restore:
    for episode in range(n_episodes):
        env_s = env.reset()
        env_done = 0
        ep_r += [0]
        for t in range(max_steps):
            if np.random.rand() < epsilon:
                env_a = np.random.randint(env.na) 
            else:
                env_q = agent.behav_q(env_s)
                env_a = np.random.choice(np.where(env_q[0] == np.max(env_q))[0])

            env_sn, env_r, env_done, _,_,_,_,_,_,_,_ = env.run(env_a - 1)  # -1: left, 0: stay, 1: right
            
            if len(replay) == replay_size:
                replay.pop(0)

            ep_r[-1] += env_r
            replay += [(env_s, env_a, env_r, env_sn, env_done)]             
            if len(replay) >= replay_start:
                samples = random.sample(replay, batch_size)
                s, a, r, sn, done = map(np.array, zip(*samples))
                
                q_max_ = agent.target_q(sn)
                bootstraped = r + (1. - done)*gamma*q_max_
                loss = agent.train_iter(s, a, bootstraped)
                losses += [loss]
                epsilon = max(epsilon - epsilon_profile.dec_step, epsilon_profile.final) 

            if env_done == 1:
                ep_st += [t]
                break

            env_s = env_sn

        if env_done != 1:
            ep_st += [max_steps]
        
        if len(replay) >= replay_start:
            if episode % target_update == 0:
                agent.copy_parameters()
            print_rate =50
            if episode % print_rate == 0:
                env_s = env.reset()
                ep_r = []
                env_done = 0
                for t in range(max_steps):
                    env_q = agent.behav_q(env_s)
                    env_a = np.random.choice(np.where(env_q[0] == np.max(env_q))[0])
                    env_sn, env_r, env_done, _,_,_,_,_,_,_,_ = env.run(env_a - 1)  # -1: left, 0: stay, 1: right
                    ep_r += [env_r]

                    if env_done == 1:
                        break

                    env_s = env_sn

                print("test reward = %2d in %3d steps" % (np.sum(ep_r), len(ep_r)))
                print('loss = %5.4f  av_r = %2.4f  r = %2d  av_steps = %3.1f  episode = %d  epsilon=%1.2f' % (loss, np.mean(ep_r[:print_rate]), ep_r[-1], np.mean(ep_st[:print_rate]), episode, epsilon))
                ep_r, ep_st = [], []

            epsilon = max(epsilon - epsilon_profile.dec_episode, epsilon_profile.final)

    save_path = saver.save(sess, "./breakout.ckpt")

else:
    saver.restore(sess, "./breakout.ckpt")

print("\nTesting \n")
env_s = env.reset()
ep_r = []
env_done = 0
for t in range(max_steps):
    env_q = agent.behav_q(env_s)
    env_a = np.random.choice(np.where(env_q[0] == np.max(env_q))[0])
    env_sn, env_r, env_done, _,_,_,_,_,_,_,_ = env.run(env_a - 1)  # -1: left, 0: stay, 1: right
    ep_r += [env_r]

    if env_done == 1:
        break

    env_s = env_sn

print("obtained reward = %2d in %3d steps" % (np.sum(ep_r), len(ep_r)))

 
