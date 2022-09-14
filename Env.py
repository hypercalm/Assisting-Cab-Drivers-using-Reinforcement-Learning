# Importing the libraries

import numpy as np
import math
import random
from itertools import permutations

# Setting the hyperparameters
m = 5 # Number of cities, from 1 ..... m
t = 24 # Number of hours, from 0 .... t-1
d = 7  # Number of days, from 0 ... d-1
C = 5 # Hourly fuel and misc costs
R = 9 # Hourly revenue from the passenger

# {'A':1, 'B':2, 'C':3, 'D':4, 'E':5}
# {'MON':0, 'TUE':1, 'WED':2, 'THU':3, 'FRI':4, 'SAT':5, 'SUN':6}


class CabDriver():

    def __init__(self):
        """
        initialise your state and define your action space and state space
        """
        self.accum_travel_hours = 0
          
        # Locations:       A, B, C, D, E          
        #                  represented by integers 1, 2, 3, 4, 5 (start index 1)
        # Time of the day: 24 hours clock 00:00, 01:00, ..., 22:00, 23:00
        #                  represented by integers 0, 1, 2, 3, 4, ..., 22, 23
        # Day of the week: MON, TUE, WED, THU, FRI, SAT, SUN
        #                  represented by integers 0, 1, 2, 3, 4, 5, 6

          
        self.action_space = [(0, 0)] + \
            list(permutations([i for i in range(1,m+1)], 2))

        self.state_space = [[x, y, z]
                            for x in range(1, m+1) for y in range(t) for z in range(d)]

        # Initialize the state to a random state (location, hours, day)
        self.state_init = random.choice(self.state_space)
        
        # Start the first round
        self.reset()

    ## Encoding state-action for neural network input
    def state_encod_arch1(self, state):
        """
        convert the state into a vector so that it can be fed to the NN. 
        This method converts a given state into a vector format. 
        Hint: The vector is of size m + t + d.
        """
        
        if not state:
            return
        
        state_encod = [0] * (m + t + d)
        
        # Encoding location
        state_encod[state[0] - 1] = 1
        
        # Encoding hour of the day
        state_encod[m + state[1]] = 1
        
        # Encoding day of the week
        state_encod[m + t + state[2]] = 1
        
        return state_encod


    # Use this function if you are using architecture-2 
    def state_encod_arch2(self, state, action):
        """convert the (state-action) into a vector so that it can be fed to the NN. 
        This method converts a given state-action pair into a vector format. 
        Hint: The vector is of size m + t + d + m + m."""
        
        if not state:
            return
        
        state_encod = [0] * (m + t + d + m + m)
        
        # Encoding location
        state_encod[state[0] - 1] = 1
        
        # Encoding hour of the day
        state_encod[m + state[1]] = 1
        
        # Encoding day of the week
        state_encod[m + t + state[2]] = 1
        
        # An action is represeted using "From location" and "To location"
        # A -> B = [1,0,0,0,0] to [0,1,0,0,0]
        # for no-ride = [0,0,0,0,0] to [0,0,0,0,0]
        if action[0] and action[1]:
            state_encod[(m + t + d) + action[0] - 1] = 1
            state_encod[(m + t + d + m) + action[1] - 1] = 1
            
        return state_encod


    ## Obtaining the number of requests
    def requests(self, state):
        """
        Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations
        """
        
        # Use the Poisson distribution for generating random number of requests based on the average
        location = state[0]
        #print("\n The location is: ")
        #print(location)
        if location == 1:
            requests = np.random.poisson(2)
        if location == 2:
            requests = np.random.poisson(12)
        if location == 3:
            requests = np.random.poisson(4)
        if location == 4:
            requests = np.random.poisson(7)
        if location == 5:
            requests = np.random.poisson(8)
        
        # Limit the number of requests to a max of 15
        if requests >15:
            requests = 15

        # (0,0) is not considered as a customer request
        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests)
        actions = [self.action_space[i] for i in possible_actions_index]

        if (0, 0) not in actions:
            actions.append((0,0))
            possible_actions_index.append(20)

        return possible_actions_index,actions   



    def reward_func(self, state, action, Time_matrix):
        """
        Takes in state, action and Time-matrix and returns the reward
        
        reward = (revenue earned from pickup point 𝑝 to drop point 𝑞) - 
                 (Cost of battery used in moving from pickup point 𝑝 to drop point 𝑞) - 
                 (Cost of battery used in moving from current point 𝑖 to pick-up point 𝑝)        
        """
        cur_loc = state[0]
        st_loc = action[0]
        end_loc = action[1]
        tod = state[1]
        dow = state[2]

        #-----------------
        def get_new_time_day(tod, dow, total_time):
            """
            calculates new time and day
            """
            tod = tod + total_time % (t - 1)
            dow = dow + (total_time // (t - 1))
            
            if tod > (t-1):
                dow = dow + (tod // (t - 1))
                tod = tod % (t - 1)
                if dow > (d - 1):
                    dow = dow % (d - 1)        
            
            return tod, dow
                
        #-----------------
        def get_total_travel_time(cur_loc, st_loc, end_loc, tod, dow):
            """
            calculates the total time of trave based on 
            """
            if not st_loc and not end_loc:
                return 0, 1
            
            t1 = 0
            if st_loc and cur_loc != st_loc:
                t1 = int(Time_matrix[cur_loc-1][st_loc-1][tod][dow])

                # Compute new time of day and day of week after t1
                tod, dow = get_new_time_day(tod, dow, t1)
            
            t2 = int(Time_matrix[st_loc-1][end_loc-1][tod][dow])

            return t1, t2
        #-----------------   
        
        t1, t2 = get_total_travel_time(cur_loc, st_loc, end_loc, tod, dow)
        
        if not st_loc and not end_loc:
            reward = -C
        else:
            reward = R * t2 - C * (t1 + t2)        
        
        return reward

    

    def next_state_func(self, state, action, Time_matrix):
        """
        Takes state and action as input and returns next state
        """
        cur_loc = state[0]
        st_loc = action[0]
        end_loc = action[1]
        tod = state[1]
        dow = state[2]
        
        #-----------------
        def get_total_travel_time(cur_loc, st_loc, end_loc, tod, dow):
            """
            calculates the total time of trave based on 
            """
            if not st_loc and not end_loc:
                return 1
            
            t1 = 0
            if st_loc and cur_loc != st_loc:
                t1 = int(Time_matrix[cur_loc-1][st_loc-1][tod][dow])
                
                # Compute new time of day and day of week after t1
                tod, dow = get_new_time_day(tod, dow, t1)

            t2 = int(Time_matrix[st_loc-1][end_loc-1][tod][dow])
            return t1 + t2

        #-----------------
        def get_new_time_day(tod, dow, total_time):
            """
            calculates new time and day
            """
            tod = tod + total_time % (t - 1)
            dow = dow + (total_time // (t - 1))
            
            if tod > (t-1):
                dow = dow + (tod // (t - 1))
                tod = tod % (t - 1)
                if dow > (d - 1):
                    dow = dow % (d - 1)        
            
            return tod, dow
        #-----------------
        
        total_trv_time = get_total_travel_time(cur_loc, st_loc, end_loc, tod, dow)
        self.accum_travel_hours += total_trv_time
        new_tod, new_dow = get_new_time_day(tod, dow, total_trv_time)
        
        if not st_loc and not end_loc:
            new_loc = state[0]
        else:
            new_loc = action[1]

        return (new_loc, new_tod, new_dow)



    def reset(self):
        self.accum_travel_hours = 0
        self.state_init = random.choice([(1,0,0), (2,0,0), (3,0,0), (4,0,0), (5,0,0)])
        
        return self.action_space, self.state_space, self.state_init
