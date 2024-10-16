from abc import ABC, abstractmethod
import numpy as np


class Learner(ABC):
    """
    Abstract class for a generic learner
    """

    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = [[] for _ in range(n_arms)]
        self.num_pulls_per_arm = np.zeros(n_arms)
        self.collected_rewards = np.array([])
        self.collected_regrets = np.array([])
        self.collected_accuracy = np.array([])
    @abstractmethod
    def pull_arm(self):
        # Select and return arms based on the learning policy
        pass

    @abstractmethod
    def update(self, chosen_arm, reward, regret):
        # Update the learner's internal state based on observed feedback
        pass
