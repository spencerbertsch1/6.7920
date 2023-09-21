#########################################
# 6.7920 - Reinforcement Learning       # 
# MIT                                   # 
# Spencer Bertsch                       #
# Fall 2023                             # 
# Homework 1 - Problem 5                # 
#########################################

# imports 
import numpy as np
import matplotlib.pyplot as plt
np.random.seed = 2

"""
Define problem parameters
"""
# define the action space: A = [0, 1, 2, ..., 20]
A = list(range(21))
# define the cost to purchase one unit of goods 
o_t = 1
# define the cost to hold one unit of goods for a single time period
h_t = 4
# define the cost for a back ordered unit of goods for single time period 
b_t = 2
# define the number of periods in the problem 
T=1
# define the action space
A = list(range(0, 21, 1))
# define the cardinality of the state space
state_space_size = 20
S_arr = list(range(-state_space_size, state_space_size+1, 1))
S = len(S_arr)
# define the elements of the discrete uniform distribution that represents demand
D_vals = list(range(0, 11, 1))


def backwards_ind(s_t: int, t: int):
    """
    backwards induction function used to generate the optimal value function V0*(s0) and the 
    optimal set of actions a0*(s0)  
    
    :param: s_t, with default starting value of zero 
    :param: t, with default starting value of zero 
    """
    
    # base case: 
    if t == T: 
        return np.max((h_t*s_t, -b_t*s_t))
    
    else: 
        # here we define the array that will hold the optimal value function values
        v = np.zeros((S, T))
        pi = np.zeros((S, T))

        # here we iterate through the reversed list (T-1 --> t)
        for t in reversed(list(range(t, T, 1))):
            print(f'--- Current period: {t} ---')
            # s_t = 0  # <-- remove later!!
            
            # for s_t in list(range(s-20, s+21, 1)): 
            for s_t in list(range(s_t-state_space_size, s_t+state_space_size+1, 1)): 
                print(f'Current state: {s_t}')
                
                all_action_cost_vec = []

                for a_t in A:
                    
                    single_action_cost_vec = []
                    
                    for D_t in D_vals: 

                        c_t = o_t*a_t + np.max(((h_t * (s_t + a_t - D_t)), (-b_t * (s_t + a_t - D_t))))

                        single_action_cost_vec.append(c_t + backwards_ind(s_t = (s_t + a_t - D_t), t=t+1))
                        
                    action_cost = sum(single_action_cost_vec)/len(single_action_cost_vec)
                    all_action_cost_vec.append(action_cost)

                pi[s_t+state_space_size, t] = np.argmin(all_action_cost_vec) # optimal policy for this state and period 
                v[s_t+state_space_size, t] = np.min(all_action_cost_vec) # optimal value function for state s and period t

        return {'v': v, 'pi': pi}


def plot_results(v_vec: np.array, pi_vec: np.array):

    v_data: list = [i[0] for i in list(v_vec)]
    pi_data: list = [i[0] for i in list(pi_vec)]
    # data to be plotted
    x = np.arange(0, len(v_data))
    
    # plotting
    plt.title("Line graph")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    # plt.plot(x, v_data, color ="red", label="V0*")
    plt.plot(x, pi_data, color ="blue", label="pi_0")
    plt.legend(loc="upper left")
    plt.show()


def main():
    """
    Driver code 
    """
    solution_dict: dict = backwards_ind(s_t=0, t=0)
    v = solution_dict['v']
    pi = solution_dict['pi']

    plot_results(v_vec=v, pi_vec=pi)


if __name__ == "__main__":
    main()
