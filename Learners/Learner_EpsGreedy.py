from Learner import Learner
import numpy as np


class EpsilonGreedyPolicy(Learner):
    """
    Class of a learner that selects arms based on the epsilon-greedy policy
    """

    def __init__(self, n_arms, epsilon):
        super().__init__(n_arms)
        self.epsilon = epsilon

    def pull_arm(self):
        """
        Select an arm based on the epsilon-greedy policy
        :return: the index of the selected arm
        """
        if np.random.random() < self.epsilon:
            # Exploration: select a random arm
            return np.random.choice(self.n_arms)
        else:
            if all(len(rewards) == 0 for rewards in self.rewards_per_arm):
                # If all arms have empty rewards, revert to exploration
                return np.random.choice(self.n_arms)
            else:
                # Otherwise, proceed with exploitation
                avg_rewards = [np.mean(rewards) if len(rewards) > 0 else float('-inf') for rewards in
                               self.rewards_per_arm]
                idx = np.argmax(avg_rewards)
                return idx

    def pull_slate_arms(self, slate_size):
        """
        Select a slate of arms based on the epsilon-greedy policy
        :param slate_size: the number of arms to select
        :return: the indices of the selected arms
        """
        if np.random.random() < self.epsilon:
            # Exploration: select a random slate
            return np.random.choice(self.n_arms, slate_size, replace=False)
        else:
            if all(len(rewards) == 0 for rewards in self.rewards_per_arm):
                # If all arms have empty rewards, revert to exploration
                return np.random.choice(self.n_arms, slate_size, replace=False)
            else:
                # Otherwise, proceed with exploitation
                avg_rewards = [np.mean(rewards) if len(rewards) > 0 else float('-inf') for rewards in
                               self.rewards_per_arm]
                idx = np.argsort(avg_rewards)[::-1]
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
        self.collected_regrets = np.append(self.collected_regrets, regret)
        self.num_pulls_per_arm[arm] += 1


