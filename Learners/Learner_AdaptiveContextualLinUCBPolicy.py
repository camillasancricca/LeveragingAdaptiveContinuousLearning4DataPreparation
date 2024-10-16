import numpy as np
import pandas as pd
from Learner import Learner
from Learner_AdaptiveRecommendationManager import AdaptiveRecommendationManager
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

property_names = ['n_tuples', 'missing_perc', 'uniqueness', 'min',
       'max', 'mean', 'median', 'std', 'skewness', 'kurtosis', 'mad', 'iqr',
       'p_min', 'p_max', 'k_min', 'k_max', 's_min', 's_max', 'entropy',
       'density']

kb_table_features = ['data_object', 'n_tuples', 'missing_perc', 'uniqueness', 'min',
       'max', 'mean', 'median', 'std', 'skewness', 'kurtosis', 'mad', 'iqr',
       'p_min', 'p_max', 'k_min', 'k_max', 's_min', 's_max', 'entropy',
       'density', 'technique_completeness', 'final_ml_value']

target_goal_mapping = {'classification': 0}

target_algorithm_mapping = {'DecisionTree' : 0, 'LogisticRegression' : 1, 'KNN' : 2, 'RandomForest' : 3,
                            'AdaBoost': 4, 'SVC': 5}

mapping_ArmToMethod = {
    0: "impute_standard",
    1: "impute_mean",
    2: "impute_median",
    3: "impute_random",
    4: "impute_knn",
    5: "impute_mice",
    6: "impute_linear_regression",
    7: "impute_random_forest",
    8: "impute_cmeans"
}

class AdaptiveContextualLinUCBPolicy(Learner):
    """
    Class of a learner that selects arms based on the LinUCB policy with
    adaptive recommendation count based on user automation preference
    """

    def __init__(self, n_arms, n_features, alpha, k, sim_threshold, max_recommendations=1):
        super().__init__(n_arms)
        self.alpha = alpha  # Confidence level
        self.A = [np.identity(n_features) for _ in range(n_arms)]  # One A matrix per arm
        self.b = [np.zeros(n_features) for _ in range(n_arms)]  # One b vector per arm
        self.theta = [np.zeros(n_features) for _ in range(n_arms)]  # Parameter vector for each arm
        self.adaptive_manager = AdaptiveRecommendationManager(max_recommendations)
        self.current_context = [None for _ in range(n_arms)]
        self.k = k
        self.sim_threshold = sim_threshold

        self.prob_t = np.zeros(n_arms)

        self.adapt_user_automation_preference = False
        self.user_automation_preference = 0

        # Initialize knowledge base
        self.knowledge_base = pd.DataFrame(columns=["subject", "predicate", "name_predicate", "value_predicate", "object"])
        self.index_kb = 0

        self.knowledge_base_table = pd.DataFrame(columns=['data_object', 'n_tuples', 'missing_perc', 'uniqueness', 'min',
       'max', 'mean', 'median', 'std', 'skewness', 'kurtosis', 'mad', 'iqr',
       'p_min', 'p_max', 'k_min', 'k_max', 's_min', 's_max', 'entropy',
       'density', 'technique_completeness', 'final_ml_value'])
        self.index_kb_table = 0

        self.profiling_table = pd.DataFrame(columns=property_names)

        self.index_name_kb = 0

        # Initialize arm statistics
        self.arm_stats = [{
            'avg_reward_score': 0.0,
            'target_analysis_value': 0.0,
            'num_uses': 0
        } for _ in range(n_arms)]

    def get_current_context(self):
        return self.current_context

    def get_knowledge_base(self):
        return self.knowledge_base

    def get_target_analysis_value(self):
        return self.arm_stats

    def get_target_slate_analysis_value(self, arms_pulled):
        # Find max target analysis value just in arm pulled
        max_target_value = 0
        max_pulled_arm = None
        for arm in arms_pulled:
            target_analysis_values = self.arm_stats[arm]['target_analysis_value']
            if target_analysis_values > max_target_value:
                max_target_value = target_analysis_values
                max_pulled_arm = arm
        return max_pulled_arm, max_target_value


    def generate_context(self, user_context, dataset_profiling):
        """
        Generates the current context for each arm by combining user-provided context
        with arm-specific statistics.
        :param user_context: The context information provided by the user.
        """

        self.calculate_target_analysis_value(dataset_profiling)


        target_goal = target_goal_mapping[user_context[0]]
        target_algorithm = target_algorithm_mapping[user_context[1]]

        user_context = np.array([target_goal, target_algorithm, user_context[2]], dtype=float)


        for arm in range(self.n_arms):
            # Combine user_context with the arm's target_analysis_value
            arm_specific_features = np.array([
                self.arm_stats[arm]['avg_reward_score'],
                self.arm_stats[arm]['target_analysis_value'],
                self.arm_stats[arm]['num_uses']
            ])
            # Assuming user_context is a numpy array; if not, convert or adjust as necessary
            self.current_context[arm] = np.concatenate((user_context, arm_specific_features))


    def initialize_knowledge_base(self, knowledge_base, knowledge_base_table):
        self.knowledge_base = knowledge_base
        self.knowledge_base_table = knowledge_base_table
        self.index_kb = len(knowledge_base)
        self.index_kb_table = len(knowledge_base_table)


    def update_knowledge_base_results(self, interaction, dataset_profiling):
        # Data Object -> Data Analysis Application
        subject = str(self.index_name_kb) + "_" + str(interaction["data_object"])
        predicate_1 = "performanceValue"
        name_predicate_1 = "F1"
        value_predicate_1 = interaction["final_ml_value"]
        object_1 = interaction["ml_algorithm"]

        self.knowledge_base.loc[self.index_kb] = [subject, predicate_1, name_predicate_1, value_predicate_1, object_1]
        self.index_kb += 1

        # Data Object -> Data Preparation Method
        predicate_2 = "isGenerated"
        name_predicate_2 = None
        value_predicate_2 = float("NaN")
        object_2 = interaction["technique_completeness"]

        self.knowledge_base.loc[self.index_kb] = [subject, predicate_2, name_predicate_2, value_predicate_2, object_2]
        self.index_kb += 1

        # Data Object -> Data Quality Metric
        predicate_3 = "metricValue"
        name_predicate_3 = None
        value_predicate_3 = interaction["initial_completeness"]
        object_3 = "%missing_values"

        self.knowledge_base.loc[self.index_kb] = [subject, predicate_3, name_predicate_3, value_predicate_3, object_3]
        self.index_kb += 1


        # Data Object -> Data Quality Metric
        subject = str(self.index_name_kb) + "_" + str(interaction["data_object"])
        predicate = "propertyValue"
        name_predicate = None

        for j in range(len(property_names)):
            value_predicate = dataset_profiling[property_names[j]]
            object = property_names[j]
            self.knowledge_base.loc[self.index_kb] = [subject, predicate, name_predicate, value_predicate, object]
            self.index_kb += 1

        self.index_name_kb += 1

        # Update the knowledge base table
        technique_completeness = interaction["technique_completeness"]
        final_ml_value = interaction["final_ml_value"]
        knowledge_base_table_features = []
        features = dataset_profiling.values[0].tolist()
        features[0] = subject
        knowledge_base_table_features.extend(features)
        knowledge_base_table_features.append(technique_completeness)
        knowledge_base_table_features.append(final_ml_value)
        self.knowledge_base_table.loc[self.index_kb_table] = [knowledge_base_table_features[i] for i in range(len(knowledge_base_table_features))]
        self.index_kb_table += 1

    def calculate_target_analysis_value_MeanAll(self, user_context):
        """
        Update the target analysis value for each arm based on the mean of 'final_ml_value' values
        obtained specifically by using that arm.
        """
        for arm in range(self.n_arms):
            # Filter the knowledge base for entries with 'final_ml_value' for this specific arm
            arm_entries = self.knowledge_base[
                (self.knowledge_base['predicate'] == 'performanceValue') &
                (self.knowledge_base['object'] == user_context[1])]
            final_ml_values = arm_entries['value_predicate'].astype(float)

            if not final_ml_values.empty:
                mean_final_ml_value = final_ml_values.mean()
            else:
                mean_final_ml_value = 0  # Default value if no data is available for this arm

            # Update the target analysis value for this specific arm
            self.arm_stats[arm]['target_analysis_value'] = mean_final_ml_value

    def calculate_target_analysis_value_Old(self, dataset_profiling):
        """
        Update the target analysis value for each arm based on the mean of 'final_ml_value' values
        obtained specifically by using that arm.
        """

        for arm in range(self.n_arms):
            arm_name = mapping_ArmToMethod[arm]

            # Take from the knowledge base table rows with technique_completeness equal to the arm
            arm_entries = self.knowledge_base_table[self.knowledge_base_table['technique_completeness'] == arm_name]

            num_arm_entries = len(arm_entries)
            #print(len(arm_entries))
            if num_arm_entries >= self.k:

                # Apply KNN to get the top-k most similar rows to the user context
                user_dataset_profiling = dataset_profiling.drop(columns=['data_object'])
                arm_entries_for_KNN = arm_entries.drop(columns=['data_object', 'technique_completeness',
                                                                'final_ml_value'])

                neigh = NearestNeighbors(n_neighbors=self.k)
                neigh.fit(arm_entries_for_KNN)
                distance, index = neigh.kneighbors(user_dataset_profiling)


                # Get the final_ml_value as the mean between the final_ml_value of the top-k most similar rows
                final_ml_values = arm_entries.iloc[index[0]]['final_ml_value'].astype(float)

                if not final_ml_values.empty:
                    mean_final_ml_value = final_ml_values.mean()
                else:
                    mean_final_ml_value = 0
            else:

                arm_entries = self.knowledge_base_table[self.knowledge_base_table['technique_completeness'] == arm]

                final_ml_values = arm_entries['final_ml_value'].astype(float)

                if not final_ml_values.empty:
                    mean_final_ml_value = final_ml_values.mean()
                else:
                    mean_final_ml_value = 0  # Default value if no data is available for this arm


            # Update the target analysis value for this specific arm
            self.arm_stats[arm]['target_analysis_value'] = mean_final_ml_value


    def calculate_target_analysis_value(self, dataset_profiling):
        """
        Update the target analysis value for each arm based on the mean of 'final_ml_value' values
        obtained specifically by using that arm.
        """

        for arm in range(self.n_arms):
            arm_name = mapping_ArmToMethod[arm]

            # Take from the knowledge base table rows with technique_completeness equal to the arm
            arm_entries = self.knowledge_base_table[self.knowledge_base_table['technique_completeness'] == arm_name]

            #num_arm_entries = len(arm_entries)

            user_dataset_profiling = dataset_profiling.drop(columns=['data_object'])

            arm_entries_for_KNN = arm_entries.drop(columns=['data_object', 'technique_completeness',
                                                            'final_ml_value'])

            if len(arm_entries_for_KNN) == 0:
                mean_final_ml_value = 0
            else:
                # Calculate similarity using cosine similarity
                similarities = cosine_similarity(arm_entries_for_KNN, user_dataset_profiling)

                # Find similar rows based on the threshold
                similar_rows_indices = [i for i, similarity in enumerate(similarities) if similarity >= self.sim_threshold]
                # Order the similar rows by similarity
                similar_rows_indices.sort(key=lambda x: similarities[x], reverse=True)
                similar_rows = self.knowledge_base_table.iloc[similar_rows_indices]

                # For the k most similar rows calculate the sum of the similarity * final_ml_value
                final_ml_values = similar_rows['final_ml_value'][:self.k]
                similarities = similarities[similar_rows_indices][:self.k]

                # Calculate the weighted average
                weighted_final_ml_values = [a * b for a, b in zip(similarities, final_ml_values)]


                if len(weighted_final_ml_values) == 0:
                    mean_final_ml_value = 0
                else:
                    mean_final_ml_value = sum(weighted_final_ml_values) / sum(similarities)
                    mean_final_ml_value = mean_final_ml_value[0]


            # Update the target analysis value for this specific arm
            self.arm_stats[arm]['target_analysis_value'] = mean_final_ml_value


    def pull_arm(self):
        """
        Method to select an arm based on the LinUCB policy
        :return:
        """
        p_t = np.zeros(self.n_arms)

        for arm in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[arm])
            self.theta[arm] = np.dot(A_inv, self.b[arm])
            p_t[arm] = np.dot(self.theta[arm], self.current_context[arm]) + self.alpha * np.sqrt(
                np.dot(self.current_context[arm].T, np.dot(A_inv, self.current_context[arm])))

        best_arm = np.argmax(p_t)

        if self.adapt_user_automation_preference:
            user_preference = self.user_automation_preference
            recommended_count = self.adapt_recommendation_count(user_preference)
            recommended_arms = np.argsort(-p_t)[:recommended_count]

        #user_preference = context[1]

        # Adapt the number of recommendations based on the user's preference
        #recommended_count = self.adapt_recommendation_count(user_preference)
        # Select the top 'recommend_count' arms based on their expected rewards
        #recommended_arms = np.argsort(-p_t)[:recommended_count]  # Sorts in descending order and selects top N

        return best_arm#, recommended_arms

    def pull_slate_arms(self, slate_size):
        """
        Method to select a slate of arms based on the LinUCB policy
        :return:
        """
        p_t = np.zeros(self.n_arms)


        for arm in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[arm])
            self.theta[arm] = np.dot(A_inv, self.b[arm])
            p_t[arm] = np.dot(self.theta[arm], self.current_context[arm]) + self.alpha * np.sqrt(
                np.dot(self.current_context[arm].T, np.dot(A_inv, self.current_context[arm])))

        if self.adapt_user_automation_preference:
            user_preference = self.user_automation_preference
            recommended_count = self.adapt_recommendation_count(user_preference)
            recommended_arms = np.argsort(-p_t)[:recommended_count]
        else:
            recommended_arms = np.argsort(-p_t)[:slate_size]

        self.prob_t = p_t
        return recommended_arms

    def update(self, chosen_arm, reward, regret):
        """
        Method to update the learner's internal state based on observed feedback
        :param chosen_arm: The arm that was chosen
        :param reward: The reward obtained from the chosen arm
        :return:
        """
        # Update the A and b matrices for the chosen arm
        self.A[chosen_arm] += np.outer(self.current_context[chosen_arm], self.current_context[chosen_arm])
        self.b[chosen_arm] += reward * self.current_context[chosen_arm]
        self.t += 1

        # Update the rewards per arm and the collected rewards
        self.rewards_per_arm[chosen_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)
        self.collected_regrets = np.append(self.collected_regrets, regret)
        self.rewards_per_arm[chosen_arm].append(reward)
        for arm in range(self.n_arms):
            if arm != chosen_arm:
                self.rewards_per_arm[arm].append(0)

        self.num_pulls_per_arm[chosen_arm] += 1

        self.arm_stats[chosen_arm]['num_uses'] += 1
        self.arm_stats[chosen_arm]['avg_reward_score'] += (reward - self.arm_stats[chosen_arm]['avg_reward_score']) / self.arm_stats[chosen_arm]['num_uses']



    def adapt_recommendation_count(self, user_preference):
        """
        Method to adapt the number of recommendations based on the user's preference
        :param user_preference:
        :return:
        """
        # Use the adaptive manager to decide on the number of recommendations
        return self.adaptive_manager.recommend_count(user_preference)

    def update_adaptive_model(self, user_preference, chosen_count, success):
        """
        Method to update the adaptive model based on the user's feedback
        :param user_preference:
        :param chosen_count:
        :param success:
        :return:
        """
        # Pass the feedback to the adaptive manager for updating its model
        self.adaptive_manager.update_model(user_preference, chosen_count, success)
