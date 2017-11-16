# EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017
# Information Theory & Machine Learning Lab, School of EE, KAIST
# written by Sae-Young Chung on 11/15/2016

import matplotlib.pyplot as plt
import numpy as np

nx=12
ny=4
n_states=nx*ny     # number of states
n_actions=4        # number of actions, 0: up, 1: down, 2: left, 3: right
n_episodes=500     # number of episodes to run
max_steps=1000
alpha=0.1          # learning rate
gamma=0.99         # discount factor
reward=np.zeros([n_states,n_actions])-1
reward[1:nx-1,:]=-100
terminal=np.zeros(n_states,dtype=np.int)          # 1 if terminal state, 0 otherwise
terminal[nx-1]=1
next_state=np.zeros([n_states,n_actions],dtype=np.int)        # next_state
for i in range(nx):
    for j in range(ny):
        s=i + j * nx
        next_state[s,:]=s
        if j < ny - 1:
            next_state[s,0]=s + nx
        if j > 0:
            next_state[s,1]=s - nx
        if i < nx - 1:
            next_state[s,3]=s + 1
        if i > 0:
            next_state[s,2]=s - 1
next_state[1:nx-1,:]=0
init_state=0       # initial state

def Q_learning(init_state, n_states, n_actions, n_episodes, max_steps, alpha, gamma, reward, terminal, next_state):
    Q=np.zeros([n_states,n_actions])
    epsilon=0.1
    n_trials=np.zeros(n_episodes)
    sum_rewards=np.zeros(n_episodes)
    for k in range(n_episodes):
        s=init_state
        for j in range(max_steps):
            if np.random.rand()<epsilon:
                a=np.random.randint(n_actions)      # random action
            else:
                mx=np.max(Q[s])
                a=np.random.choice(np.argwhere(Q[s]==mx).flatten())     # greedy action with random tie break
            sn=next_state[s][a]
            r=reward[s][a]
            sum_rewards[k]+=r
            Q[s,a]=(1.-alpha)*Q[s,a]+alpha*(r+gamma*np.max(Q[sn]))
            if terminal[sn]:
                n_trials[k]=j+1
                break
            s=sn
    return Q,n_trials,sum_rewards

def sarsa(init_state, n_states, n_actions, n_episodes, max_steps, alpha, gamma, reward, terminal, next_state):
    Q=np.zeros([n_states,n_actions])
    epsilon=0.1
    n_trials=np.zeros(n_episodes)
    sum_rewards=np.zeros(n_episodes)
    for k in range(n_episodes):
        s=init_state
        if np.random.rand()<epsilon:
            a=np.random.randint(n_actions)      # random action
        else:
            mx=np.max(Q[s])
            a=np.random.choice(np.argwhere(Q[s]==mx).flatten())     # greedy action with random tie break
        for j in range(max_steps):
            sn=next_state[s][a]
            r=reward[s][a]
            sum_rewards[k]+=r
            if np.random.rand()<epsilon:
                an=np.random.randint(n_actions)      # random action
            else:
                mx=np.max(Q[sn])
                an=np.random.choice(np.argwhere(Q[sn]==mx).flatten())     # greedy action with random tie break
            Q[s,a]=(1.-alpha)*Q[s,a]+alpha*(r+gamma*Q[sn,an])
            if terminal[sn]:
                n_trials[k]=j+1
                break
            s=sn
            a=an
    return Q,n_trials,sum_rewards

# training
N=100   # number of trials for averaging (1000 would give smoother results)
sum_rewards_acc=np.zeros(n_episodes)
Q_acc=np.zeros([n_states,n_actions])
for _ in range(N):
    Q,n_trials,sum_rewards=Q_learning(init_state, n_states, n_actions, n_episodes, max_steps, alpha, gamma, reward, terminal, next_state)
    sum_rewards_acc+=sum_rewards
    Q_acc+=Q
sum_rewards_Q=sum_rewards_acc/N
Q_Q_learning=Q_acc/N

sum_rewards_acc=np.zeros(n_episodes)
Q_acc=np.zeros([n_states,n_actions])
for _ in range(N):
    Q,n_trials,sum_rewards=sarsa(init_state, n_states, n_actions, n_episodes, max_steps, alpha, gamma, reward, terminal, next_state)
    sum_rewards_acc+=sum_rewards
    Q_acc+=Q
sum_rewards_sarsa=sum_rewards_acc/N
Q_sarsa=Q_acc/N

#print sum_rewards_Q
print("Q-learning")
print("up")
print(Q_Q_learning[:,0].reshape((ny,nx)).transpose())
print("down")
print(Q_Q_learning[:,1].reshape((ny,nx)).transpose())
print("left")
print(Q_Q_learning[:,2].reshape((ny,nx)).transpose())
print("right")
print(Q_Q_learning[:,3].reshape((ny,nx)).transpose())
#print sum_rewards_sarsa
print("Sarsa")
print("up")
print(Q_sarsa[:,0].reshape((ny,nx)).transpose())
print("down")
print(Q_sarsa[:,1].reshape((ny,nx)).transpose())
print("left")
print(Q_sarsa[:,2].reshape((ny,nx)).transpose())
print("right")
print(Q_sarsa[:,3].reshape((ny,nx)).transpose())

plt.figure(1)

t=np.arange(n_episodes)+1
plt.plot(t, sum_rewards_Q, t, sum_rewards_sarsa)
plt.legend(['Q learning','Sarsa'], loc='lower right')
plt.grid(True)
plt.xlabel('Episodes')
plt.ylabel('Average reward')
plt.axis((0,n_episodes,-100,0))
#plt.savefig('cliff_walk.png', dpi=300, bbox_inches='tight')
plt.show()

