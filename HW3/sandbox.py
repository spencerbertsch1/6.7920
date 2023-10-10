# imports 
import numpy as np
import matplotlib.pyplot as plt
np.random.seed = 2

def approx_inventory_control(d_t: int, h_t: int, q_t: int, t: str):
    """
    iterative approach
    """
    print(f'Running approximate inventory control problem where d_t: {d_t}, h_t: {h_t}, q_t: {q_t}')

    assert (t in ["0", "T"]), f'arg t should be either \"0\" or \"T\", not {t}'

    # define the action space 
    A = list(range(0, 21, 1))

    # define the mesh representing all of the possible values for demand
    W_t = np.around(np.arange (0, 10.1, 0.1), 2)

    # initialize the state value function as a list (we will add a value for each state in the mesh)
    V_s = []
    pi = []

    # iterate over all states in the mesh 
    for s in np.around(np.arange (-15, 15.1, 0.1), 2): 

        single_state_cost = []

        # compute the state value funciton for each state in the mesh 
        # for each possible action 
        for a in A: 

            single_action_cost = []

            # step through the mesh of all possible demands W_t
            for d_t in W_t: 
                if t == "0": 
                    # non-terminal cost
                    cost = d_t*a + max((h_t*s), (-q_t*s))
                else:
                    # non-terminal cost
                    cost = np.max(h_t*s, -q_t*s)
                single_action_cost.append(cost)

            action_cost = sum(single_action_cost)/len(single_action_cost)
            single_state_cost.append(action_cost)

        pi.append(np.argmin(single_state_cost)) # optimal policy for this state and period 
        V_s.append(np.min(single_state_cost)) # optimal value function for state s and period t

    return V_s, pi