#########################################
# 6.7920 - Reinforcement Learning       # 
# MIT                                   # 
# Spencer Bertsch                       #
# Fall 2023                             # 
# Homework 3 - Problem 5                # 
#########################################

# imports 
import numpy as np
import matplotlib.pyplot as plt
import time 
np.random.seed = 2

# define problem parameters 
o_t = 1  # ordering cost for one unit
h_t = 4  # holding cost for one unit 
q_t = 2  # backlog cost for one unit 

# define time horizon 
T = 10

# define the space of demand
D_max = 10
# define the max action for the action space 
a_max = 20

# define action space 
action_space = np.arange(0,a_max+1)

# define the mesh representing all of the possible values for demand
dist_D = [(Dt, 1/(D_max+1)) for Dt in np.arange(0,D_max+1)]

# define the cache that we will use to store already computed values 
V_computed = {}  # maps tuples (t, s_t) to values (floats)
a_optimal = {}  # maps tuples (t, s_t) to values (floats)

def V(t, s_t):
    """
    Recursive function to compute the optimal value function for a single state using 
    dynamic programming. This function searches the entire space of actions and all 
    possible values of the discrete space of possible demands to find the value of 
    the state and the optimal action to be taken. 

    :param: t, int representing time step 
    :param: s_t, float representing the state 
    """

    # base case 1
    if t>T: # beyond horizon
        return 0
    
    # base case 2
    if t == T: # terminal
        return max(h_t*s_t, -q_t*s_t)
    
    # check to see if we've already computed this value 
    if (t, s_t) in V_computed: 
        return V_computed[(t, s_t)]
  
    # recursive case: compute V and optimal action
    v_at = []  # <-- define the values for each action at s_t
    for a_t in action_space:

        expected_value_list = []

        # iterate over mesh of all possible demand values 
        for D_t, prob_Dt in dist_D:
            
            # compute the cost for this state, action, and demand at time t
            next_state = s_t + a_t - D_t
            current_cost = o_t*a_t + max(h_t*next_state,-q_t*next_state) + V(t+1, next_state)
            
            expected_value_list.append(prob_Dt * current_cost)

        expected_value = sum(expected_value_list)
        v_at.append(expected_value)

    # compute V and optimal action
    a_index = np.argmin(v_at)
    at = action_space[a_index]
    Vt = v_at[a_index]

    # storing results for future use
    V_computed[(t, s_t)] = Vt
    a_optimal[(t,s_t)] = at

    return Vt


def approx_value_iteration():
    pass 


def main():

    # define problem parameters
    stocks_a = np.arange(-15, 15.1, 1)
    t = 0

    # compute the state value function for each state defined above 
    tic = time.time()
    V0 = []
    for i, s in enumerate(stocks_a): 
        curr_value = V(t, s)
        V0.append(curr_value)
        # if i%10==0:
        #     print(f'i={i}, V0={curr_value}')
    print(f'Wall time: {round(time.time() - tic, 2)} seconds.')

    actions_a_optimal = [a_optimal[(t, s)] for s in stocks_a]  # retrieve the optimal actions from the cache 

    # value function
    plt.title("Value function comparison")
    plt.plot(stocks_a, V0, color='red', marker='o')
    plt.xlabel("$s_t$")
    plt.ylabel("$V_t$")
    plt.legend(["Optimal"])
    plt.grid()
    plt.show()

    # optimal action
    plt.figure()
    plt.title("Policy comparison")
    plt.plot(stocks_a, actions_a_optimal, color='blue', marker='v')
    plt.xlabel("$s_t$")
    plt.ylabel("$a_t$")
    plt.legend(["Optimal"])
    plt.grid()
    plt.show()

if __name__ == "__main__": 
    main()
