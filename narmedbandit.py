# n-armed bandit problem
# EE488 Special Topics in EE <Deep Learning and AlphaGo>, Fall 2016, Fall 2017
# Information Theory & Machine Learning Lab, School of EE, KAIST
# written by Sae-Young Chung in Oct. 2016

import matplotlib.pyplot as plt
import numpy as np

n=10			# number of arms
R=np.random.rand(n)	# average payout
stdev=0.3               # standard deviation for random Gaussian
N=1000			# number of plays
m=500			# number of trials for averaging

def epsilon_greedy(n, N, m, R, stdev, e_start, e_end, eN):
    avg_reward=np.zeros(N)
    Rmax=np.max(R)
    for _ in range(m):
        Q=np.zeros(n)
        Qn=np.zeros(n)

        for t in range(N):
            e=(e_end-e_start)*np.min([t,eN])/eN+e_start
            if np.random.rand(1)<e:
                i=np.random.randint(n)
            else:
                argmax_list=np.where(Q==max(Q))[0]   # list of all indices in argmax
                i=np.random.choice(argmax_list)      # random tie break
            r=np.random.normal(R[i], stdev, 1)
            Q[i]=(Q[i]*Qn[i]+r)/(Qn[i]+1)
            Qn[i]+=1
            avg_reward[t]+=r
    return avg_reward/(m*Rmax)

avg_reward0=epsilon_greedy(n, N, m, R, stdev, 0., 0., N)
avg_reward001=epsilon_greedy(n, N, m, R, stdev, 0.01, 0.01, N)
avg_reward01=epsilon_greedy(n, N, m, R, stdev, 0.1, 0.1, N)
avg_reward05=epsilon_greedy(n, N, m, R, stdev, 0.5, 0.5, N)
avg_reward1=epsilon_greedy(n, N, m, R, stdev, 1., 1., N)
avg_reward=epsilon_greedy(n, N, m, R, stdev, 1., 0., 500)

plt.figure(1)
plt.rc('text', usetex=True)

t=np.arange(N)+1
plt.plot(t, avg_reward0, t, avg_reward001, t, avg_reward01, t, avg_reward05, t, avg_reward1, t, avg_reward)
plt.legend(['$\epsilon=0$', '$\epsilon=0.01$', '$\epsilon=0.1$', '$\epsilon=0.5$', '$\epsilon=1$', '$\epsilon=(1 \Rightarrow 0)$'], loc='lower right')

plt.grid(True)
plt.xlabel('plays')
plt.ylabel('Normalized average reward')
# uncomment the following to save the figure
#plt.savefig('narmedbandit.png', dpi=300, bbox_inches='tight')
plt.show()

