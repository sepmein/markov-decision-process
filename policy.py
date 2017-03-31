from random import random
from pymongo import MongoClient
import numpy as np

"""
	Markov Decision Process
	Conponents:
		state - store values
		action
		state_action pair - store rewards and quality
			reward
			q function of (s,a)
		# policies
"""


class Policy:
    """
        MDP - Policy
    """

    def __init__(self, gamma, tao, db, default_value):
        self.gamma = gamma or 0.9
        self.tao = tao or 0.1
        self.db = db
        self.default_value = default_value

    def pai(self, actions, next_states, rewards, method='max'):
        """given a state and actions, calculate the optimal action"""
        # at some rate select random action
        # return optimal policy otherwise
        next_states_values = np.array([0.0 for state in next_states])
            # self.db.find_values(next_states, default_value=self.default_value)
        if random() < self.tao:
            random_index = int(actions.shape[0] * random())
            value = bellman_quality_equation(rewards[random_index], self.gamma,
                                             next_states_values[random_index])
            return (actions[random_index], value, 'explotary')
        else:
            # terminal_results = []
            # next_state_is_terminal = False
            # # for every next state which will be ended, update value and reward
            # for index, state in enumerate(next_states):
            #     ended, winner = tictac.judge_terminal(state)
            #     if ended:
            #         next_state_is_terminal = True
            #         terminal_results.append({
            #             'index': index,
            #             'winner': winner
            #         })
            (action_index, optimal_quality) = bellman_value_equation(
                rewards, self.gamma, next_states_values, method)
            next_actions = actions[action_index]
            # select a random optimal action
            random_index = int(next_actions.shape[0] * random())
            return (next_actions[random_index], optimal_quality, 'explantary')


def bellman_quality_equation(reward, gamma, next_state_value):
    """
            Bellman quality equation, simplified version
            Q(s,a) = R(s,a) + gamma * simga(T(s, a, s') * V(s'))
    """
    return reward + gamma * next_state_value


def bellman_value_equation(rewards, gamma, next_states_values, method='max'):
    """
            compute Bellman value function for the given state and actions
            V(s) = max of a(R(s,a) + gamma * sigma(T(s, a, s') * V(s')))
    """
    qualities = rewards + gamma * next_states_values
    if method == 'max':
        optimal_quality = np.max(qualities)
        action_index = np.argwhere(qualities == np.max(qualities)).flatten()
    elif method == 'min':
        optimal_quality = np.min(qualities)
        action_index = np.argwhere(qualities == np.min(qualities)).flatten()
    return (action_index, optimal_quality)
