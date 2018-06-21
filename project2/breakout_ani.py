# EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017
# Information Theory & Machine Learning Lab, School of EE, KAIST
# written by Sae-Young Chung, 11/19/2017

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import tensorflow as tf


class breakout_animation(animation.TimedAnimation):
    def __init__(self, env, max_steps, frames_per_step = 5):
        self.env = env
        self.agent = self.initialize_agent()
        self.max_steps = max_steps

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        self.objs = []

        # boundary
        w = 0.1
        ax.plot([-w,-w,env.nx+w,env.nx+w],[0,env.ny+w,env.ny+w,0],'k-',linewidth=5)

        # bricks
        wb = 0.05
        self.bricks = []
        self.brick_colors = [['red'], ['blue','red'], ['blue','green','red'], ['blue','green','yellow','red'], ['blue','green','yellow','orange','red'], \
            ['purple','blue','green','yellow','brown','orange','red'], ['purple','blue','green','yellow','brown','orange','red']]    # add more colors if needed
        for y in range(self.env.nb):
            b = []
            yp = y + (self.env.ny - self.env.nt - self.env.nb)
            for x in range(self.env.nx):
                b.append(patches.Rectangle((x + wb, yp + wb), 1-2*wb, 1-2*wb, edgecolor='none', facecolor=self.brick_colors[self.env.nb-1][y]))
                ax.add_patch(b[x])
                self.objs.append(b[x])
            self.bricks.append(b)
 
        # ball
        self.ball = patches.Circle(env.get_ball_pos(0.), radius = 0.15, color = 'red')
        ax.add_patch(self.ball)
        self.objs.append(self.ball)

        # score text
        self.text = ax.text(0.5 * env.nx, 0, '', ha='center')
        self.objs.append(self.text)

        # game over text
        self.gameover_text = ax.text(0.5 * env.nx, 0.5 * env.ny, '', ha='center')
        self.objs.append(self.gameover_text)

        self.frames_per_step = frames_per_step
        self.total_frames = self.frames_per_step * self.max_steps

        # paddle
        self.paddle = patches.Rectangle((env.p, 0.5), 1, 0.5, edgecolor='none', facecolor='red')
        ax.add_patch(self.paddle)

        # for early termination of animation
        self.iter_objs = []
        self.iter_obj_cnt = 0

        # interval = 50msec
        animation.TimedAnimation.__init__(self, fig, interval=50, repeat=False, blit=False)

    def initialize_agent(self):
        sess = tf.InteractiveSession()
        agent = DQN(env)
        tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()).run()
        agent.copy_parameters()
        saver = tf.train.Saver()
        saver.restore(sess, "./breakout.ckpt")
        return agent

    def _draw_frame(self, k):
        if self.terminal:
            return
        if k == 0:
            self.iter_obj_cnt -= 1
        if k % self.frames_per_step == 0:
            env_q = self.agent.behav_q(self.env.s)
            env_a = np.random.choice(np.where(env_q[0] == np.max(env_q))[0])
            self.a = env_a - 1
            self.p = self.env.p
            self.pn = min(max(self.p + self.a, 0), self.env.nx - 1)

        t = (k % self.frames_per_step) * 1. / self.frames_per_step
        self.ball.center = self.env.get_ball_pos(t)
        self.paddle.set_x(t * self.pn + (1-t) * self.p)

        if k % self.frames_per_step == self.frames_per_step - 1:
            sn, reward, terminal, p0, p, bx0, by0, vx0, vy0, rx, ry = self.env.run(self.a)
            self.sum_reward += reward
            if reward > 0.:
                self.bricks[ry][rx].set_facecolor('none')
                self.text.set_text('Score: %d' % self.sum_reward)
            if terminal:
                self.terminal = terminal
                self.gameover_text.set_text('Game Over')
                for _ in range(self.total_frames - k - 1):
                    self.iter_objs[self.iter_obj_cnt].next()     # for early termination of animation (latest iterator is used first)

        self._drawn_artists = self.objs

    def new_frame_seq(self):
        iter_obj = iter(range(self.total_frames))
        self.iter_objs.append(iter_obj)
        self.iter_obj_cnt += 1
        return iter_obj

    def _init_draw(self):
        _ = self.env.reset()
        self.sum_reward = 0.
        self.p = self.env.p    # current paddle position
        self.pn = self.p       # next paddle position
        self.a = 0             # action
        self.terminal = 0

        for y in range(self.env.nb):
            for x in range(self.env.nx):
                self.bricks[y][x].set_facecolor(self.brick_colors[self.env.nb-1][y])

        self.ball.center = self.env.get_ball_pos(0.)
        self.paddle.set_x(self.p)

        self.text.set_text('Score: 0')
        self.gameover_text.set_text('')


class DQN:
    def __init__(self, env):
        # state size [ny, nx, nf]
        self.s = tf.placeholder(tf.float32, shape=[None, env.ny, env.nx, env.nf])
        self.a = tf.placeholder(tf.int32, shape=[None])
        self.batch_size = tf.placeholder(tf.int32, shape=[])
        self.bootstraped = tf.placeholder(tf.float32, shape=[None])
        self.ch1 = 80
        self.env = env

        W_fc1 = tf.Variable(tf.truncated_normal([env.nx*env.ny*env.nf, self.ch1], stddev=0.1))
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[self.ch1]))
        W_fc2 = tf.Variable(tf.truncated_normal([self.ch1, env.na], stddev=0.1))
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[env.na]))

        h_flat = tf.reshape(self.s, [self.batch_size, -1])
        fc1 = tf.matmul(h_flat, W_fc1) + b_fc1
        self.q = tf.matmul(tf.nn.relu(fc1), W_fc2) + b_fc2        
        action_one_hot = tf.one_hot(self.a, env.na, 1.0, 0.0, name='action_one_hot')
        self.q_a = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')

        self.optim = tf.train.RMSPropOptimizer(0.0002, momentum=0.95, epsilon=0.01) 
        self.theta = [W_fc1, b_fc1, W_fc2, b_fc2]
        self.target_init()

        losses = tf.squared_difference(self.bootstraped, self.q_a)
        self.loss = tf.reduce_mean(losses)
        self.train_step = self.optim.minimize(self.loss)

    def copy_parameters(self):
        for i in range(len(self.theta)):
            self.theta_[i].assign(self.theta[i]).eval()

    def target_init(self):
        self.theta_ = []
        for param in self.theta:
            self.theta_ += [tf.Variable(tf.truncated_normal(tf.shape(param), stddev=0.1))]
        W_fc1, b_fc1, W_fc2, b_fc2 = self.theta_

        h_flat = tf.reshape(self.s, [self.batch_size, -1])
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



from breakout_env import *
from wait import *

env = breakout_environment(nx = 5, ny = 8, nb = 3, nt = 1, nf = 2)
ani = breakout_animation(env, 200)
ani.save('breakout.mp4', dpi=200)
plt.show(block=False)
wait('Press enter to quit')
