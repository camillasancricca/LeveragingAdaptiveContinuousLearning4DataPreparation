from Utils.Utils import *

mapping_ArmToMethod = {
    "impute_standard": 0,
    "impute_mean": 1,
    "impute_median": 2,
    "impute_random": 3,
    "impute_knn": 4,
    "impute_mice": 5,
    "impute_linear_regression": 6,
    "impute_random_forest": 7,
    "impute_cmeans": 8
}

mapping_MethodToArm = {
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


class SimulationEnvironmentLOO:
    def __init__(self, percentage_usage_data, num_interactions_in_kb, alpha, seed, dataset_name_test):
        self.completeness_results = pd.read_csv("../Datasets/datasetCreated/results.csv")
        self.completeness_profiling = pd.read_csv("../Datasets/datasetCreated/profiling.csv")

        # Take training dataset and shuffle it
        self.train_dataset, self.test_data, self.transactions_in_kb = create_traintest_split_with_knowledge_LOO(percentage_usage_data,
                                                num_interactions_in_kb, seed, dataset_name_test)

        self.dataset_name_test = dataset_name_test
        self.index = 0

        self.alpha = alpha
        self.beta = 1 - self.alpha

        self.current_interaction_dataframe = None


    def get_interaction(self):
        interaction = self.train_dataset.iloc[self.index]
        self.index += 1
        return interaction

    def get_user_context(self, interaction):
        target_goal = "classification"
        target_algorithm = interaction['ml_algorithm']
        completeness_score = interaction['initial_completeness']
        return np.array([target_goal, target_algorithm, completeness_score])

    def get_user_context_test(self, interaction):
        target_goal = "classification"
        target_algorithm = interaction['ml_algorithm'].values[0]
        completeness_score = interaction['initial_completeness'].values[0]
        return np.array([target_goal, target_algorithm, completeness_score])

    def get_dataset_profiling(self, interaction):
        dataset_name = interaction['data_object']
        initial_completeness = interaction['initial_completeness']

        # Compare both dataset name and initial completeness
        dataset_profiling = self.completeness_profiling[(self.completeness_profiling['data_object'] == dataset_name)]
        dataset_profiling = dataset_profiling[(dataset_profiling['missing_perc'] == initial_completeness)]
        return dataset_profiling

    def get_dataset_profiling_test(self, interaction):
        dataset_name = interaction['data_object'].values[0]
        initial_completeness = interaction['initial_completeness'].values[0]

        # Compare both dataset name and initial completeness
        dataset_profiling = self.completeness_profiling[(self.completeness_profiling['data_object'] == dataset_name)]
        dataset_profiling = dataset_profiling[(dataset_profiling['missing_perc'] == initial_completeness)]
        return dataset_profiling

    def get_offline_interaction(self, interaction, arm_pulled):
        data_object = interaction['data_object']
        initial_completeness = interaction['initial_completeness']
        algorithm = interaction['ml_algorithm']
        offline_interaction = self.completeness_results[(self.completeness_results['data_object'] == data_object)]
        offline_interaction = offline_interaction[(offline_interaction['initial_completeness'] == initial_completeness)]
        offline_interaction = offline_interaction[(offline_interaction['ml_algorithm'] == algorithm)]
        offline_interaction = offline_interaction[(offline_interaction['technique_completeness'] == mapping_MethodToArm[arm_pulled])]
        offline_interaction = offline_interaction.iloc[0]
        return offline_interaction


    def get_reward_per_arm(self, arm_recommended, target_analysis_value, interaction):
        user_selected_technique = mapping_ArmToMethod[interaction['technique_completeness']]

        user_choice = 1 if user_selected_technique == arm_recommended else 0

        delta_analysis = interaction['final_ml_value'] - target_analysis_value

        reward = self.alpha * user_choice + self.beta * delta_analysis

        return reward

    def get_offline_reward_per_arm(self, arm_recommended, target_analysis_value, offline_interaction):
        user_selected_technique = mapping_ArmToMethod[offline_interaction['technique_completeness']]

        user_choice = 1 if user_selected_technique == arm_recommended else 0

        delta_analysis = offline_interaction['final_ml_value'] - target_analysis_value

        reward = self.alpha * user_choice + self.beta * delta_analysis

        return reward

    def get_offline_reward_per_arm_slate(self, arms_chosen, target_analysis_value, offline_interaction):
        user_selected_technique = mapping_ArmToMethod[offline_interaction['technique_completeness']]

        user_choice = 1 if user_selected_technique in arms_chosen else 0

        delta_analysis = offline_interaction['final_ml_value'] - target_analysis_value

        reward = self.alpha * user_choice + self.beta * delta_analysis

        return reward

    def get_online_reward_per_arm(self, arm_recommended, target_analysis_value, interaction):
        user_selected_technique = mapping_ArmToMethod[interaction['technique_completeness']]

        user_choice = 1 if user_selected_technique == arm_recommended else 0

        delta_analysis = interaction['final_ml_value'] - target_analysis_value

        reward = self.alpha * user_choice + self.beta * delta_analysis

        return reward


    def get_online_reward_per_arm_slate(self, arms_chosen, target_analysis_value, interaction):
        user_selected_technique = mapping_ArmToMethod[interaction['technique_completeness']]

        user_choice = 1 if user_selected_technique in arms_chosen else 0

        delta_analysis = interaction['final_ml_value'] - target_analysis_value

        reward = self.alpha * user_choice + self.beta * delta_analysis

        return reward



    def get_offline_regret_per_arm(self, reward, interaction):
        best_triple = self.completeness_results[(self.completeness_results['data_object'] == interaction['data_object'])]
        best_triple = best_triple[(best_triple['initial_completeness'] == interaction['initial_completeness'])]
        best_triple = best_triple[(best_triple['ml_algorithm'] == interaction['ml_algorithm'])]

        best_value = best_triple['final_ml_value'].max()
        best_possible_reward = self.alpha + self.beta * best_value

        regret = best_possible_reward - reward

        return regret
    def get_regret_per_arm(self, reward, interaction):
        best_triple = self.completeness_results[(self.completeness_results['data_object'] == interaction['data_object'])]
        best_triple = best_triple[(best_triple['initial_completeness'] == interaction['initial_completeness'])]
        best_triple = best_triple[(best_triple['ml_algorithm'] == interaction['ml_algorithm'])]

        best_value = best_triple['final_ml_value'].max()
        best_possible_reward = self.alpha + self.beta * best_value

        regret = best_possible_reward - reward

        return regret


    def get_test_interaction(self, data_object_to_test, initial_completeness_to_test, algorithm_to_test):
        og_dataset = self.completeness_results.sample(len(self.completeness_results)).reset_index(drop=True)
        test_dataset = og_dataset[(og_dataset['data_object'] == data_object_to_test)]
        test_dataset = test_dataset[(test_dataset['initial_completeness'] == initial_completeness_to_test)]
        test_dataset = test_dataset[(test_dataset['ml_algorithm'] == algorithm_to_test)]
        self.current_interaction_dataframe = test_dataset
        test_dataset = test_dataset.sample(1)
        return test_dataset

    def get_true_best_technique(self, data_object_to_test, initial_completeness_to_test, algorithm_to_test):
        test_interaction = self.completeness_results[(self.completeness_results['data_object'] == data_object_to_test)]
        test_interaction = test_interaction[(test_interaction['initial_completeness'] == initial_completeness_to_test)]
        test_interaction = test_interaction[(test_interaction['ml_algorithm'] == algorithm_to_test)]
        test_interaction = test_interaction[(test_interaction['final_ml_value'] == test_interaction['final_ml_value'].max())]
        true_best_technique = test_interaction['technique_completeness'].values[0]
        return true_best_technique

    def get_top3_true_best_technique(self, data_object_to_test, initial_completeness_to_test, algorithm_to_test):
        test_interaction = self.completeness_results[(self.completeness_results['data_object'] == data_object_to_test)]
        test_interaction = test_interaction[(test_interaction['initial_completeness'] == initial_completeness_to_test)]
        test_interaction = test_interaction[(test_interaction['ml_algorithm'] == algorithm_to_test)]
        top3_interactions = self.current_interaction_dataframe.nlargest(3, 'final_ml_value')
        top3_true_best_techniques = top3_interactions['technique_completeness'].values
        top3_true_best_techniques = [mapping_ArmToMethod[technique] for technique in top3_true_best_techniques]
        return top3_true_best_techniques

    def calculate_map_at_1(self, recommended_arms):
        relevance_threshold = self.current_interaction_dataframe.nlargest(1, 'final_ml_value')['final_ml_value'].values[0] - 0.005
        #print(f'Real value: {test_interaction["final_ml_value"].values[0]}')
        #print(f'Max value: {self.current_interaction_dataframe.nlargest(1, "final_ml_value")["final_ml_value"].values[0]}')
        average_precision = 0
        if self.current_interaction_dataframe[
            self.current_interaction_dataframe['technique_completeness'] == mapping_MethodToArm[recommended_arms[0]]][
            'final_ml_value'].values[0] >= relevance_threshold:
            average_precision += 1

        relevance_rank = average_precision
        return average_precision, relevance_rank

    def calculate_map_at_k(self, recommended_arms, k):
        relevance_threshold = self.current_interaction_dataframe.nlargest(1, 'final_ml_value')['final_ml_value'].values[0] - 0.005

        average_precision = 0
        relevance_rank = 1
        for i in range(k):
            if self.current_interaction_dataframe[
                self.current_interaction_dataframe['technique_completeness'] == mapping_MethodToArm[recommended_arms[i]]][
                'final_ml_value'].values[0] >= relevance_threshold:
                average_precision += relevance_rank / (i + 1)
                relevance_rank += 1

        mean_average_precision = average_precision / k
        return mean_average_precision, relevance_rank


    def calculate_map_at_1_sim(self, recommended_arms, policy):
        relevance_threshold = self.current_interaction_dataframe.nlargest(1, 'final_ml_value')['final_ml_value'].values[0] - 0.005
        average_precision = 0
        if policy == "Random":
            relevance_threshold = self.current_interaction_dataframe.nlargest(1, 'final_ml_value')['final_ml_value'].values[0]
            if self.current_interaction_dataframe[
                self.current_interaction_dataframe['technique_completeness'] == mapping_MethodToArm[recommended_arms[0]]][
                'final_ml_value'].values[0] >= relevance_threshold:
                average_precision += 1
        else:
            if self.current_interaction_dataframe[
                self.current_interaction_dataframe['technique_completeness'] == mapping_MethodToArm[recommended_arms[0]]][
                'final_ml_value'].values[0] >= relevance_threshold:
                average_precision += 1

        relevance_rank = average_precision
        return average_precision, relevance_rank

    def calculate_map_at_k_sim(self, recommended_arms, k, policy):
        relevance_threshold = self.current_interaction_dataframe.nlargest(1, 'final_ml_value')['final_ml_value'].values[0] - 0.005

        average_precision = 0
        relevance_rank = 1
        if policy == "Random":
            relevance_threshold = self.current_interaction_dataframe.nlargest(1, 'final_ml_value')['final_ml_value'].values[0]
            for i in range(k):
                if self.current_interaction_dataframe[
                    self.current_interaction_dataframe['technique_completeness'] == mapping_MethodToArm[recommended_arms[i]]][
                    'final_ml_value'].values[0] >= relevance_threshold:
                    average_precision += relevance_rank / (i + 1)
                    relevance_rank += 1
        else:
            for i in range(k):
                if self.current_interaction_dataframe[
                    self.current_interaction_dataframe['technique_completeness'] == mapping_MethodToArm[
                        recommended_arms[i]]][
                    'final_ml_value'].values[0] >= relevance_threshold:
                    average_precision += relevance_rank / (i + 1)
                    relevance_rank += 1

        mean_average_precision = average_precision / k
        return mean_average_precision, relevance_rank





