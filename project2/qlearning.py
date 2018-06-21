# EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017
# Information Theory & Machine Learning Lab, School of EE, KAIST
# written by Sae-Young Chung, 11/12/2017

import numpy as np

# train using Q-learning
def Q_learning_train(env, n_episodes, max_steps, alpha, gamma, epsilon_profile):
    Q = np.zeros([env.n_states, env.n_actions])
    n_steps = np.zeros(n_episodes) + max_steps
    sum_rewards = np.zeros(n_episodes)  # total reward for each episode
    epsilon = epsilon_profile.init
    for k in range(n_episodes):
        s = env.init_state
        for j in range(max_steps):
            if np.random.rand() < epsilon:
                a = np.random.randint(env.n_actions)      # random action
            else:
                mx = np.max(Q[s])
                a = np.random.choice(np.where(Q[s]==mx)[0])     # greedy action with random tie break
            sn = env.next_state[s,a]
            r = env.reward[s,a]
            sum_rewards[k] += r
            Q[s,a] = (1.-alpha)*Q[s,a]+alpha*(r+gamma*np.max(Q[sn]))
            if env.terminal[sn]:
                n_steps[k] = j+1  # number of steps taken
                break
            s = sn
            epsilon = max(epsilon - epsilon_profile.dec_step, epsilon_profile.final)
            #print(epsilon)
        epsilon = max(epsilon - epsilon_profile.dec_episode, epsilon_profile.final)
    return Q, n_steps, sum_rewards

# run tests using action-value function table Q assuming epsilon greedy
def Q_test(Q, env, n_episodes, max_steps, epsilon):
    n_steps = np.zeros(n_episodes) + max_steps  # number of steps taken for each episode
    sum_rewards = np.zeros(n_episodes)          # total rewards obtained for each episode
    state = np.zeros([n_episodes, max_steps], dtype=np.int)      
    action = np.zeros([n_episodes, max_steps], dtype=np.int)      
    next_state = np.zeros([n_episodes, max_steps], dtype=np.int)
    reward = np.zeros([n_episodes, max_steps])
    for k in range(n_episodes):
        s = env.init_state
        for j in range(max_steps):
            state[k,j] = s
            if np.random.rand() < epsilon:
                a = np.random.randint(env.n_actions)      # random action
            else:
                mx = np.max(Q[s])
                a = np.random.choice(np.where(Q[s]==mx)[0])     # greedy action with random tie break
            action[k,j] = a
            sn = env.next_state[s,a]
            r = env.reward[s,a]
            next_state[k,j] = sn
            reward[k,j] = r
            sum_rewards[k] += r
            if env.terminal[sn]:
                n_steps[k] = j+1
                break
            s = sn
    return n_steps, sum_rewards, state, action, next_state, reward

