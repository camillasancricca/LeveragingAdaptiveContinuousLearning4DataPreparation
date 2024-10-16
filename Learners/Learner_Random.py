from Learner import Learner
import numpy as np


class RandomPolicy(Learner):
    """
    Class of a learner that selects arms randomly
    """

    def __init__(self, n_arms):
        super().__init__(n_arms)

    def pull_arm(self):
        """
        Select an arm randomly
        :return: the index of the selected arm
        """
        return np.random.choice(self.n_arms)

    def pull_slate_arms(self, slate_size):
        """
        Select a slate of arms randomly
        :param slate_size: the number of arms to select
        :return: the indices of the selected arms
        """
        return np.random.choice(self.n_arms, slate_size, replace=False)

    def update(self, arm, reward, regret):
        """
        Update the learner's internal state based on observed feedback
        :param arm: arm that has been selected
        :param reward: reward obtained by selecting the arm
        """
        self.t += 1
        self.rewards_per_arm[arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)
        self.collected_regrets = np.append(self.collected_regrets, regret)
        self.num_pulls_per_arm[arm] += 1


