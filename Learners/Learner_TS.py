from Learner import Learner
import numpy as np


class ThompsonSamplingPolicy(Learner):
    """
    Class of a learner that selects arms based on the Thompson sampling policy
    """

    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.beta_parameters = np.ones((n_arms, 2))

    def pull_arm(self):
        """
        Select an arm based on the Thompson sampling policy
        :return: the index of the selected arm
        """
        sampled_theta = np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1])
        return np.argmax(sampled_theta)

    def pull_slate_arms(self, slate_size):
        """
        Select a slate of arms based on the Thompson sampling policy
        :param slate_size: the number of arms to select
        :return: the indices of the selected arms
        """
        sampled_theta = np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1])
        idx = np.argsort(sampled_theta)[::-1]
        return idx[:slate_size]
    def update(self, arm, reward, regret):
        """
        Update the learner's internal state based on observed feedback
        :param arm: the arm that has been selected
        :param reward: the reward obtained by selecting the arm
        """

        self.t += 1
        self.rewards_per_arm[arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)
        self.beta_parameters[arm, 0] = self.beta_parameters[arm, 0] + reward
        self.beta_parameters[arm, 1] = self.beta_parameters[arm, 1] + 1 - reward
        self.collected_regrets = np.append(self.collected_regrets, regret)
        self.num_pulls_per_arm[arm] += 1

        # Prevent beta_parameters[arm] to be <= 0
        if self.beta_parameters[arm, 0] <= 0:
            self.beta_parameters[arm, 0] = 1e-16
        if self.beta_parameters[arm, 1] <= 0:
            self.beta_parameters[arm, 1] = 1e-16



