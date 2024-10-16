import numpy as np


class AdaptiveRecommendationManager:
    """
    Class of a learner that selects the number of recommendations based on the user's preference
    """

    def __init__(self, max_recommendations=3):
        """
        Initialize the adaptive recommendation manager
        :param max_recommendations: the maximum number of recommendations to choose from
        """
        self.max_recommendations = max_recommendations

        # Initialize success and failure counts for Thompson Sampling
        self.successes = np.zeros((2, max_recommendations))
        self.failures = np.zeros((2, max_recommendations))

    def recommend_count(self, user_preference):
        """
        Method to recommend the number of items to the user based on their preference
        :param user_preference:
        :return:
        """
        # Determine the index for user preference (0 for low, 1 for high)
        preference_index = int(user_preference > 0)
        samples = np.zeros(self.max_recommendations)
        for i in range(self.max_recommendations):
            # Sample from a Beta distribution for each potential recommendation count
            samples[i] = np.random.beta(self.successes[preference_index, i] + 1, self.failures[preference_index, i] + 1)
        recommended_count = np.argmax(samples) + 1  # Adding 1 because recommendation counts start from 1
        return recommended_count

    def update_model(self, user_preference, chosen_count, success):
        """
        Method to update the model based on user feedback
        :param user_preference:
        :param chosen_count:
        :param success:
        :return:
        """
        # Update the success/failure counts based on feedback
        preference_index = int(user_preference > 0)
        if success:
            self.successes[preference_index, chosen_count - 1] += 1
        else:
            self.failures[preference_index, chosen_count - 1] += 1
