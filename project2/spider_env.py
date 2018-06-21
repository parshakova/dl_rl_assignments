# EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017
# Information Theory & Machine Learning Lab, School of EE, KAIST
# written by Sae-Young Chung, 11/12/2017

import numpy as np

# baby spider environment
class spider_environment: 
    def __init__(self):
        self.n_states = pow(4,4)         # number of states: leg up/down, forward/backward
        self.n_actions = pow(4,4)        # number of actions 
        self.reward = np.zeros([self.n_states, self.n_actions])
        self.terminal = np.zeros(self.n_states, dtype=np.int)          # 1 if terminal state, 0 otherwise
        self.next_state = np.zeros([self.n_states, self.n_actions], dtype=np.int)        # next_state
        self.init_state = 0b00001010    # initial state
        n_legs = 4
        transition = [[1,0,2,0],[1,0,3,1],[3,2,2,0],[3,2,3,1]]
        class leg: pass
        for s in range(self.n_states):
            legs_state = []
            down_s = ""
            for l in range(n_legs):
                leg_i = leg()
                leg_i.up = (s >> 2*l) & 1
                leg_i.fw = (s >> (2*l +1)) & 1
                legs_state += [leg_i]
                down_s = bin(leg_i.up ^ 0b1)[2:] + down_s
            down_s = int("0b"+down_s, 2)
            for a in range(self.n_actions):
                legs_action = []
                for l in range(n_legs):
                    leg_i = leg()
                    leg_i.action_up = ((a >> 2*l) & 3) == 0
                    leg_i.action_dn = ((a >> 2*l) & 3) == 1
                    leg_i.action_fw = ((a >> 2*l) & 3) == 2
                    leg_i.action_bw = ((a >> 2*l) & 3) == 3
                    legs_action += [leg_i]
                next_s = 0
                down_sn = "0b"
                # start from MSB and move to LSB
                for l in reversed(range(n_legs)):
                    s_i = (s >> 2*l) & 3
                    a_i = (a >> 2*l) & 3
                    next_s = (next_s << 2) + transition[s_i][a_i] 
                    down_sn += bin((transition[s_i][a_i] & 1) ^ 0b1)[2:]
                down_sn = int(down_sn, 2)
                self.next_state[s,a] = next_s

                product = (down_s) & (down_sn)
                total_down = bin(product).count("1")
                total_force = 0
                for l in range(n_legs):
                    total_force += (legs_state[l].up == 0 and legs_state[l].fw == 1 and legs_action[l].action_bw == 1) - (legs_state[l].up == 0 and legs_state[l].fw == 0 and legs_action[l].action_fw == 1)
                
                diagonal = ((product & 1) & ((product >> 3)& 1)) | (((product >> 2) & 1) & ((product >> 1)& 1))

                if total_down == 0:
                    self.reward[s,a] = 0
                elif total_down >= 3:
                    self.reward[s,a] = 1.0 * total_force / total_down
                elif total_down == 2 and diagonal:
                    self.reward[s,a] = 1.0 * total_force / total_down
                else:
                    self.reward[s,a] = 0.25 * total_force / total_down


                


                 

