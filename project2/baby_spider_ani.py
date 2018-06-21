# EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017
# Information Theory & Machine Learning Lab, School of EE, KAIST
# written by Sae-Young Chung, 11/12/2017

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import axes3d

class baby_spider_animation(animation.TimedAnimation):
    def __init__(self, Q, env, max_steps, epsilon, frames_per_step = 5):
        self.Q = Q
        self.env = env
        self.max_steps = max_steps
        self.epsilon = epsilon

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('equal')
        ax.set_xlim(-1,2)
        ax.set_ylim(-1.5,1.5)
        ax.set_zlim(0,3)
    
        # body
        n_turns = 15
        u = np.linspace(0, 2 * n_turns * np.pi, 1000)
        self.body = np.zeros([3,len(u)])
        for i in range(len(u)):
            self.body[0,i] = 0.5 * np.cos(u[i] * 0.5 / n_turns)
            self.body[1,i] = 0.5 * np.sin(u[i])*np.sin(u[i] * 0.5 / n_turns)
            self.body[2,i] = 0.3 + 0.3 * np.cos(u[i])*np.sin(u[i] * 0.5 / n_turns)
    
        # hat
        n_turns = 7
        u = np.linspace(0, 2 * n_turns * np.pi, 1000)
        self.hat = np.zeros([3,len(u)])
        for i in range(len(u)):
            self.hat[0,i] = 0.1 + 0.23 * u[i] / 2 / n_turns / np.pi * np.cos(u[i])
            self.hat[1,i] = 0.23 * u[i] / 2 / n_turns / np.pi * np.sin(u[i])
            self.hat[2,i] = 0.6 + (1 - u[i] / 2 / n_turns / np.pi) * 0.5

        # ball on top
        n_turns = 4
        u = np.linspace(0, 2 * n_turns * np.pi, 1000)
        self.ball = np.zeros([3,len(u)])
        for i in range(len(u)):
            self.ball[0,i] = 0.1 + 0.07 * np.cos(u[i])*np.sin(u[i] * 0.5 / n_turns)
            self.ball[1,i] = 0.07 * np.sin(u[i])*np.sin(u[i] * 0.5 / n_turns)
            self.ball[2,i] = 1.1 + 0.07 * np.cos(u[i] * 0.5 / n_turns)

        # legs
        self.leg = np.zeros([3,4])
        self.leg[0] = np.array([0., 0., 0., 0.])
        self.leg[1] = np.array([0., 0.2, 0.4, 0.6])
        self.leg[2] = np.array([0., 0.7, -0.3, -0.3])
        self.leg_ofs = np.array([[0.], [0.5], [0.3]])   # for front left leg

        self.forward_angle = np.pi * (20. / 180)    # a leg can move forward/backward by +/- this angle
        self.upward_angle = np.pi * (30. / 180)     # a leg can move upward by this angle
        self.forward_dist = 0.5 * (self.leg[1,2] + self.leg[1,3]) * 2 * np.sin(self.forward_angle)    # distance traveled in one step

        self.body_obj, = ax.plot([], [], [], 'k', linewidth=1)
        self.hat_obj, = ax.plot([], [], [], 'purple', linewidth=1)
        self.ball_obj, = ax.plot([], [], [], 'darkgreen', linewidth=1)
        self.leg_FL_obj, = ax.plot([], [], [],'k', linewidth=3)
        self.leg_FR_obj, = ax.plot([], [], [],'k', linewidth=3)

        self.text_obj = ax.text(-1,1.5,2,'',zdir='x')

        self.line_objs = [self.body_obj, self.hat_obj, self.ball_obj, \
            self.leg_FL_obj, self.leg_FR_obj]
        self.objs = [self.body_obj, self.hat_obj, self.ball_obj, \
            self.leg_FL_obj, self.leg_FR_obj, self.text_obj]

        self.frames_per_step = frames_per_step

        # interval = 50msec
        animation.TimedAnimation.__init__(self, fig, interval=50, repeat=False, blit=False)

    def _draw_frame(self, k):
        if k % self.frames_per_step == 0:
            self.fl_up = self.new_fl_up; self.fl_fw = self.new_fl_fw 
            if np.random.rand() < self.epsilon:
                a = np.random.randint(self.env.n_actions)      # random action
            else:
                mx = np.max(self.Q[self.s])
                a = np.random.choice(np.where(self.Q[self.s]==mx)[0])     # greedy action with random tie break
            sn = self.env.next_state[self.s,a]
            r = self.env.reward[self.s,a]
            self.sum_reward += r
            self.current_speed = r * self.forward_dist
            self.s = sn
            self.new_fl_up = self.s & 1
            self.new_fl_fw = (self.s >> 1) & 1

        t = (k % self.frames_per_step) * 1. / self.frames_per_step
        FL = self.rotation_matrix(-(2 * (t * self.new_fl_fw + (1-t) * self.fl_fw) - 1) * self.forward_angle,\
            (t * self.new_fl_up + (1-t) * self.fl_up) * self.upward_angle)
        self.leg_FL, self.leg_FR = self.rotate_leg(FL)

        self.current_location += self.current_speed / self.frames_per_step

        self.body_obj.set_data(self.body[0] + self.current_location, self.body[1])
        self.body_obj.set_3d_properties(self.body[2])
        self.hat_obj.set_data(self.hat[0] + self.current_location, self.hat[1])
        self.hat_obj.set_3d_properties(self.hat[2])
        self.ball_obj.set_data(self.ball[0] + self.current_location, self.ball[1])
        self.ball_obj.set_3d_properties(self.ball[2])

        self.leg_FL_obj.set_data(self.leg_FL[0] + self.current_location, self.leg_FL[1])
        self.leg_FL_obj.set_3d_properties(self.leg_FL[2])
        self.leg_FR_obj.set_data(self.leg_FR[0] + self.current_location, self.leg_FR[1])
        self.leg_FR_obj.set_3d_properties(self.leg_FR[2])

        if self.fl_up == 0 and self.new_fl_up == 0:
            self.leg_FL_obj.set_color('red')
            self.leg_FR_obj.set_color('red')
        else:
            self.leg_FL_obj.set_color('black')
            self.leg_FR_obj.set_color('black')

        self.text_obj.set_text('step: %d(%d), score: %.2f' % (k // self.frames_per_step + 1, k + 1, self.sum_reward))

        self._drawn_artists = self.objs

    def new_frame_seq(self):
        return iter(range(self.frames_per_step * self.max_steps))

    def _init_draw(self):
        self.s = self.env.init_state

        self.new_fl_up = self.fl_up = self.s & 1
        self.new_fl_fw = self.fl_fw = (self.s >> 1) & 1

        FL = self.rotation_matrix(-(2 * self.fl_fw - 1) * self.forward_angle, self.fl_up * self.upward_angle)
        self.leg_FL, self.leg_FR = self.rotate_leg(FL)

        self.current_speed = 0.
        self.current_location = 0.
        self.sum_reward = 0.

        for obj in self.line_objs:
            obj.set_data([], [])
            obj.set_3d_properties([])
        self.text_obj.set_text('')

    # rotate by theta (z axis), then by phi (x axis)
    def rotation_matrix(self, theta, phi):
        R = np.array([[np.cos(theta), -np.sin(theta), 0.], [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
        R = np.matmul(np.array([[1., 0., 0.], [0., np.cos(phi), -np.sin(phi)], [0., np.sin(phi), np.cos(phi)]]), R)
        return R

    def rotate_leg(self, FL):
        leg_FL = np.matmul(FL, self.leg); leg_FL += self.leg_ofs
        leg_FR = np.matmul(FL, self.leg); leg_FR[1] *= -1; leg_FR += np.array([[self.leg_ofs[0,0]], [-self.leg_ofs[1,0]], [self.leg_ofs[2,0]]])
        return leg_FL, leg_FR

