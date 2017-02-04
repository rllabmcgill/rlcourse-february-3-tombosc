import numpy as np
import matplotlib.pyplot as plt
from utils import step

"""
on-policy first-visit monte carlo with action value function

this one is without exploring starts.

the gambler has to gamble at least 1, the discount rate is 1.
"""

n = 2**5
p_h = 0.3
print "n:", n
print "probability of heads:", p_h

def generate_episode(pi, returns, counts, epsilon, start_from=None):
    """
    generates an episode in a uniformly chosen random state and random action
    then follows policy pi
    updates returns and counts
    first visit variant
    """
    assert(np.array_equal(pi[0,:], np.zeros(n+1)))
    assert(np.allclose(pi.sum(axis=0)[1:], np.ones(n)))
   
    sample_uniform_action = lambda s: np.random.choice(s) + 1 # from 1 to s
    if start_from:
        s, a = start_from
        if a == None:
            a = sample_uniform_action(s)
    else:
        s = np.random.choice(n-1) + 1 # from 1 to n-1
        a = sample_uniform_action(s)

    def eps_greedy_sample_action(s):
        greedy = np.random.choice(2, p=[epsilon, 1-epsilon])
        if not greedy:
            return sample_uniform_action(s)
        else:
            return np.random.choice(n+1, p=pi[:,s]) # from greedy policy pi

    states = [s]
    actions = [a]
    rewards = [0]
    first_visited = np.ones(n+1, dtype=int) * (-1)

    while s != n and s != 0:
        if first_visited[s] == -1:
            first_visited[s] = len(states) - 1
        s_n, r = step(s,a, p_h, n)
        states.append(s_n)
        rewards.append(r)
        if s_n != 0 and s_n != n:
            actions.append(eps_greedy_sample_action(s_n))
        else:
            actions.append(0)
        s = states[-1]
        a = actions[-1]

    for s,t in enumerate(first_visited):
        if t == -1: # never visited
            continue
        a = actions[t]
        returns[a,s] += sum(rewards[t+1:])
        if first_visited[s] == t:
            counts[a,s] += 1
                
    
def policy_mc_iterate(max_iter=10000, delta=10**-6, epsilon=0.05, start_from=None):
    # init random policy pi[a,s]
    pi = np.zeros((n+1,n+1))
    for s in range(0, n+1):
        for a in range(1, s+1):
            pi[a,s] = np.random.uniform()
        pi[:,s] = pi[:,s] / pi[:,s].sum()
    pi = np.nan_to_num(pi)
    
    q = np.zeros((n,n+1)) 
    returns = np.zeros((n,n+1))
    counts = np.zeros((n,n+1))

    for i in range(max_iter):
        print "ITERATION", i
        old_q = np.copy(q)
        generate_episode(pi, returns, counts, epsilon, start_from)
        q = np.nan_to_num(returns / counts)
        # greedify:
        pi_argmax = np.argmax(q[1:,:], axis=0)
        pi = np.zeros((n+1, n+1))
        for s, a in enumerate(pi_argmax):
            pi[a+1, s] = 1
        print "convergence:", np.linalg.norm(old_q - q) 
    return q, pi


q, pi = policy_mc_iterate(max_iter=50000, epsilon=0.2, start_from=(30,None))

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14,5))
im_1 = ax1.imshow(q[1:,1:n], interpolation='None', origin='lower',
          extent=(1,n-1,1,n))
f.colorbar(im_1, ax=ax1)

ax1.set_title("q(a,s)")
im_2 = ax2.imshow(pi[1:,1:n], interpolation='None', origin='lower',
          extent=(1,n-1,1,n))
f.colorbar(im_2, ax=ax2)
ax2.set_title("Target policy pi")
plt.show()
f.savefig("gambler_mc_eps_50k_it_p_03_state_30_eps_0_2.png")


