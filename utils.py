import numpy as np
import matplotlib.pyplot as plt

def generate_episode(pi, p_h, n):
    """
    generates an episode in a random state following policy pi
    updates returns and counts
    """
    assert(np.array_equal(pi[0,:], np.zeros(n+1)))
    assert(np.allclose(pi.sum(axis=0)[1:-1], np.ones(n-1)))

    s = np.random.choice(n-1) + 1 # from 1 to n-1
    sample_action = lambda s: np.random.choice(n+1, p=pi[:,s]) # from 1 to s
    a = sample_action(s)
    states = [s]
    actions = [a]
    rewards = [0]

    while s != n and s != 0:
        s_n, r = step(s, a, p_h, n)
        states.append(s_n)
        rewards.append(r)
        if s_n != 0 and s_n != n:
            actions.append(sample_action(s_n))
        else:
            actions.append(0)
        s = states[-1]
        a = actions[-1]
    return (states, actions, rewards)
 
def step(s, a, p_h, n):
    """
    simulate a step

    agent is in state s, bet a where 1 <= a <= s
    returns (next state, reward)
    """
    assert(a>= 1 and a<=s)
    rand = np.random.choice(2, p=[1-p_h, p_h])
    if rand == 1:
        next_s = s + a
    else:
        next_s = s - a

    reward = 0
    if next_s >= n:
        reward = 1
        next_s = n
    elif next_s <= 0:
        next_s = 0
    return (next_s, reward)

