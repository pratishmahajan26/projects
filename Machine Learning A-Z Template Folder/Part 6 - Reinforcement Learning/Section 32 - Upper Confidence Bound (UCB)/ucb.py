# import libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
# import the dataset

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# implementing UCB Algoritm
# First : build the upper confidence bound Algoritmn
# Second : deal with the first 10 rounds coz if we will have no clue we cannot build the algo.
# so for random we will consider that one ad is selected in each round till round 10
# third: create a list to show ads selected in each round
# forth: calculate the number_of_selections by incrementing by 1
#fifth: then we have to calculate sum_of_rewards for each ad . for calculating this first we have to know the reward in each round for each ad
# last : calculate the total_reward ie total numbers of 1's
d = dataset.shape[1]
number_of_selections = [0] * d
sum_of_rewards = [0] * d
N = dataset.shape[0]
ads_selected = []
total_reward = 0
for n in range(0,N):
    max_upper_bound = 0   # coz we have to find maximum upper bound in each round
    ad = 0
    for i in range(0,d):
        if number_of_selections[i] > 0 :     # algo will work only if each ad is selected at least once
            average_reward = sum_of_rewards[i]/number_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n+1) / number_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if max_upper_bound < upper_bound:
            max_upper_bound = upper_bound
            ad = i             # coz we want to know the advertisement index that has the max_upper_bound
    ads_selected.append(ad)    # this will give ad selected in each round
    number_of_selections[ad] = number_of_selections[ad] + 1   # this will give number of times each ads are selected 
    reward = dataset.values[n,ad]    # to know the reward (0 or 1) of the ad at each round
    sum_of_rewards[ad] = sum_of_rewards[ad] + reward      # to calculate the sum of rewards for each ad
    total_reward = total_reward + reward

    
# visualising the data
plt.hist(ads_selected)
plt.xlabel('ads')
plt.ylabel('number of times each ads are selected')
plt.show()