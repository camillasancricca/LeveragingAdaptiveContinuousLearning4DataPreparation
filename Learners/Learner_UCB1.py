from Learner import Learner
import numpy as np


class UpperConfidenceBoundPolicy(Learner):
    """
    Class of a learner that selects arms based on the UCB1 policy
    """

    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.zeros(n_arms)

    def pull_arm(self):
        """
        Select an arm based on the UCB1 policy
        :return: the index of the selected arm
        """
        if self.t < self.n_arms:
            return self.t
        else:
            upper_confidence_bounds = self.empirical_means + self.confidence
            return np.argmax(upper_confidence_bounds)

    def pull_slate_arms(self, slate_size):
        """
        Select a slate of arms based on the UCB1 policy
        :param slate_size: the number of arms to select
        :return: the indices of the selected arms
        """
        if self.t < self.n_arms:
            # Explore by choosing self.t and other random arms if needed
            chosen_arms = [self.t]
            remaining_slots = slate_size - 1
            other_arms = list(range(self.n_arms))
            other_arms.remove(self.t)
            np.random.shuffle(other_arms)
            chosen_arms.extend(other_arms[:remaining_slots])
            return chosen_arms
        else:
            # Use UCB1 to select the best arms based on their upper confidence bounds
            upper_confidence_bounds = self.empirical_means + self.confidence
            return np.argsort(-upper_confidence_bounds)[:slate_size]
    def update(self, arm, reward, regret):
        """
        Update the learner's internal state based on observed feedback
        :param arm: the arm that has been selected
        :param reward: the reward obtained by selecting the arm
        """
        self.t += 1
        self.rewards_per_arm[arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)
        self.collected_regrets = np.append(self.collected_regrets, regret)
        self.empirical_means[arm] = np.mean(self.rewards_per_arm[arm])
        n_samples = len(self.rewards_per_arm[arm])
        self.confidence[arm] = np.sqrt(2 * np.log(self.t) / n_samples)
        self.num_pulls_per_arm[arm] += 1
