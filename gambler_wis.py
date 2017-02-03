import numpy as np
import matplotlib.pyplot as plt
from utils import step, generate_episode

"""
off-policy monte carlo every-visit with weighted importance sampling. 

The behavior is uniform random.

the gambler has to gamble at least 1, the discount rate is 1.
"""

n = 2**5
p_h = 0.3
print "n:", n
print "probability of heads:", p_h

          
    
def policy_mc_iterate(max_iter=10, delta=10**-6):
    # init random policy pi[a,s]
    pi = np.zeros((n+1,n+1))
    for s in range(1, n+1):
        pi[1:s+1,s] = np.ones(s) * (1./s)

    # init uniform policy mu[a,s] that will be the behavior policy
    mu = np.zeros((n+1,n+1))
    for s in range(1, n+1):
        mu[1:s+1,s] = np.ones(s) * (1./s)
    
    q = np.zeros((n,n+1)) 
    C = np.zeros((n,n+1))
    for i in range(max_iter):
        print "EPISODE", i
        states, actions, rewards = generate_episode(mu, p_h, n)
        #print "states", states
        #print "actions", actions
        #print "rewards:", rewards
        G = 0
        W = 1
        for t in reversed(range(0, len(states) - 1)): 
            a = actions[t]
            s = states[t]
            G = G + rewards[t+1]
            C[a,s] = C[a,s] + W
            q[a,s] = q[a,s] + (W / C[a,s]) * (G - q[a,s])
            # greedify:
            pi[:,s] = np.zeros(n+1)
            action_max = np.argmax(q[1:,s]) + 1 
            # add 1 to action_max because we throw away bet 0
            pi[action_max, s] = 1
            if action_max != a: # i.e. if mu[a,s] = 0
                break
            W = W / mu[a,s]
    return q, pi


q, pi = policy_mc_iterate(max_iter=200000)
print q
print pi

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
f.savefig("gambler_wis_uni_p_0_3_200kit_randomstart2.png")
