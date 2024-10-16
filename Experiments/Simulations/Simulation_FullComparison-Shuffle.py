from Environments.SimulationEnvironment import SimulationEnvironment
from Learners.Learner_AdaptiveContextualLinUCBPolicy import AdaptiveContextualLinUCBPolicy
from Learners.Learner_TS import ThompsonSamplingPolicy
from Learners.Learner_UCB1 import UpperConfidenceBoundPolicy
from Learners.Learner_EpsGreedy import EpsilonGreedyPolicy
from Learners.Learner_Random import RandomPolicy
from Utils.Utils import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from datetime import datetime
import os

########################################################################################################################
# Output file creation
########################################################################################################################
init_file_name = "FullComparison-Shuffle_ALL"
dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_string = 'Simulations/' + init_file_name + '/' + dt_string
sub_dir = os.path.join('Output', folder_string)
directory = os.path.join('Output', folder_string)

# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

# Define file paths inside the directory
result_file_path = os.path.join(directory, "result_" + init_file_name + ".csv")

with open(result_file_path, 'w') as f:
    header = ['ID', 'Experiment', 'Seed', '% Data', '# Train', '# Test', '# Trans_KB',
              'Policy',
              'Policy CumRew', 'Policy CumReg', 'Policy MAP@3', 'Policy MAP@1'
              ]
    f.write(','.join(header) + '\n')

########################################################################################################################
# Variables to store the rewards and the metrics
########################################################################################################################
curr_policy_reward_per_experiment = []

curr_policy_regret_per_experiment = []

curr_policy_correct_predictions_per_experiment = 0

All_arms_pulled = {}

# At the end of all the experiments I want to plot all the different rewards and regrets for all the combinations
All_experiments_results = {}

########################################################################################################################
# Utils variables
########################################################################################################################

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

mapping_MethodToArm = {
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

########################################################################################################################
# Figure creation
########################################################################################################################
# Initialize figures and axes outside the policies loop
fig_cum_reward, ax_cum_reward = plt.subplots()
fig_cum_regret, ax_cum_regret = plt.subplots()
fig_inst_reward, ax_inst_reward = plt.subplots()
fig_inst_regret, ax_inst_regret = plt.subplots()
fig_accuracy, ax_accuracy = plt.subplots()

init_title = "Policy"

########################################################################################################################
# Simulation Parameters
########################################################################################################################
percentage_usage_data = 0.1
seeds = [0, 2, 3, 4, 5, 6, 7]
slate_number = 3

num_interactions_in_kb = 500
sim_threshold = 0.9
topK = 5
alpha_confidence = 0.9
alpha_reward = 0.9

n_arms = 9
n_features = 6
n_experiments = 5

#######################################################################################################################
# Hyperparameters
########################################################################################################################
policies = ["TS", "UCB1", "Random", "Eps", "LinUCB_KB"]

########################################################################################################################
# Run the experiments
########################################################################################################################
for policy in policies:
    # Reset variables for this experiment
    curr_policy_reward_per_experiment = []

    curr_policy_regret_per_experiment = []

    curr_policy_relevance_per_experiment = []

    opt_per_experiment = []

    for e in range(n_experiments):
        seed = seeds[e]
        print(f'Running experiment {e + 1}/{n_experiments} with policy: {policy} ...')

        # Initialize the simulation environment
        env = SimulationEnvironment(percentage_usage_data, num_interactions_in_kb, alpha_reward, seed)
        train_data = env.train_dataset
        test_data = env.test_data
        print(f'Initializing knowledge base with {len(env.transactions_in_kb)} transactions...')
        initial_knowledge_base, initial_knowledge_base_table = create_knowledge_base(env.transactions_in_kb)

        curr_relevance = []

        # Initialize Learner
        if policy == "TS":
            learner = ThompsonSamplingPolicy(n_arms=n_arms)
            fake_LinUCB = AdaptiveContextualLinUCBPolicy(n_arms=n_arms, n_features=n_features,
                                                         alpha=alpha_confidence,
                                                         sim_threshold=sim_threshold, k=topK)
            fake_LinUCB.initialize_knowledge_base(initial_knowledge_base, initial_knowledge_base_table)
        elif policy == "UCB1":
            learner = UpperConfidenceBoundPolicy(n_arms=n_arms)
            fake_LinUCB = AdaptiveContextualLinUCBPolicy(n_arms=n_arms, n_features=n_features,
                                                         alpha=alpha_confidence,
                                                         sim_threshold=sim_threshold, k=topK)
            fake_LinUCB.initialize_knowledge_base(initial_knowledge_base, initial_knowledge_base_table)
        elif policy == "Random":
            learner = RandomPolicy(n_arms=n_arms)
            fake_LinUCB = AdaptiveContextualLinUCBPolicy(n_arms=n_arms, n_features=n_features,
                                                         alpha=alpha_confidence,
                                                         sim_threshold=sim_threshold, k=topK)
            fake_LinUCB.initialize_knowledge_base(initial_knowledge_base, initial_knowledge_base_table)
        elif policy == "Eps":
            learner = EpsilonGreedyPolicy(n_arms=n_arms, epsilon=0.3)
            fake_LinUCB = AdaptiveContextualLinUCBPolicy(n_arms=n_arms, n_features=n_features,
                                                         alpha=alpha_confidence,
                                                         sim_threshold=sim_threshold, k=topK)
            fake_LinUCB.initialize_knowledge_base(initial_knowledge_base, initial_knowledge_base_table)
        elif policy == "LinUCB_KB":
            learner = AdaptiveContextualLinUCBPolicy(n_arms=n_arms, n_features=n_features,
                                                     alpha=alpha_confidence,
                                                     sim_threshold=sim_threshold, k=topK)
            learner.initialize_knowledge_base(initial_knowledge_base, initial_knowledge_base_table)

        ################################################################################################################
        # Training
        ################################################################################################################
        print('Training the learners ...')
        T_train = len(train_data)
        for t in tqdm(range(T_train)):
            # User and Dataset information retrieval
            interaction = env.get_interaction()

            user_context = env.get_user_context(interaction)
            dataset_profiling = env.get_dataset_profiling(interaction)

            # Create user context
            if policy == "LinUCB_KB":
                learner.generate_context(user_context, dataset_profiling)
            else:
                fake_LinUCB.generate_context(user_context, dataset_profiling)

            # Select an arm
            policy_arm_pulled = learner.pull_slate_arms(slate_number)

            # Obtain the offline interaction
            offline_interaction = env.get_offline_interaction(interaction, policy_arm_pulled[0])

            # Update the knowledge base
            if policy == "LinUCB_KB":
                learner.update_knowledge_base_results(offline_interaction, dataset_profiling)
            else:
                fake_LinUCB.update_knowledge_base_results(offline_interaction, dataset_profiling)

            if policy == "LinUCB_KB":
                # Obtain the target analysis value
                target_analysis_values = learner.get_target_analysis_value()[policy_arm_pulled[0]]['target_analysis_value']

                # Obtain the reward
                reward = env.get_offline_reward_per_arm(policy_arm_pulled[0], target_analysis_values, offline_interaction)

                # Obtain the reward for each learner
                regret = env.get_regret_per_arm(reward, offline_interaction)

                # Update each learner's state
                learner.update(policy_arm_pulled[0], reward, regret)
            else:
                # Obtain the target analysis value
                target_analysis_values = fake_LinUCB.get_target_analysis_value()[policy_arm_pulled[0]]['target_analysis_value']

                # Obtain the reward
                reward = env.get_offline_reward_per_arm(policy_arm_pulled[0], target_analysis_values, offline_interaction)

                # Obtain the reward for each learner
                regret = env.get_regret_per_arm(reward, offline_interaction)

                # Update each learner's state
                learner.update(policy_arm_pulled[0], reward, regret)


        ################################################################################################################
        # Testing Performance Evaluation
        ################################################################################################################
        curr_policy_map_at_1_per_experiment = 0

        curr_policy_map_at_3_per_experiment = 0

        print('\nTesting the learners ...')
        T_test = len(test_data)
        for i in tqdm(range(T_test)):
            interaction = test_data.iloc[i]

            arm_pulled = mapping_MethodToArm[interaction['technique_completeness']]

            test_interaction = env.get_test_interaction(interaction["data_object"],
                                                        interaction["initial_completeness"],
                                                        interaction["ml_algorithm"])
            true_best_technique = env.get_true_best_technique(interaction["data_object"],
                                                              interaction["initial_completeness"],
                                                              interaction["ml_algorithm"])

            top3_best_techniques = env.get_top3_true_best_technique(interaction["data_object"],
                                                                interaction["initial_completeness"],
                                                                interaction["ml_algorithm"])


            user_context = env.get_user_context_test(test_interaction)
            dataset_profiling = env.get_dataset_profiling_test(test_interaction)

            # Create user context
            if policy == "LinUCB_KB":
                learner.generate_context(user_context, dataset_profiling)
            else:
                fake_LinUCB.generate_context(user_context, dataset_profiling)

            policy_arm_pulled = learner.pull_slate_arms(slate_number)

            # Update the knowledge base
            if policy == "LinUCB_KB":
                learner.update_knowledge_base_results(interaction, dataset_profiling)
            else:
                fake_LinUCB.update_knowledge_base_results(interaction, dataset_profiling)

            # Calculate MAP@K
            map_at_1, relevance_rank_at_1 = env.calculate_map_at_1_sim(policy_arm_pulled, policy)
            map_at_3, relevance_rank_at_3 = env.calculate_map_at_k_sim(policy_arm_pulled, slate_number, policy)
            #map_at_1, relevance_rank_at_1 = env.calculate_map_at_1(policy_arm_pulled)
            #map_at_3, relevance_rank_at_3 = env.calculate_map_at_k(policy_arm_pulled, slate_number)
            curr_policy_map_at_1_per_experiment += map_at_1
            curr_policy_map_at_3_per_experiment += map_at_3
            curr_relevance.append(relevance_rank_at_3)

            if policy == "LinUCB_KB":
                # Obtain the target analysis value
                target_analysis_values = learner.get_target_analysis_value()[arm_pulled]['target_analysis_value']

                # Obtain the reward
                reward = env.get_online_reward_per_arm(policy_arm_pulled[0], target_analysis_values, interaction)

                # Obtain the reward for each learner
                regret = env.get_regret_per_arm(reward, interaction)

                # Update each learner's state
                learner.update(arm_pulled, reward, regret)
            else:
                # Obtain the target analysis value
                target_analysis_values = fake_LinUCB.get_target_analysis_value()[arm_pulled]['target_analysis_value']

                # Obtain the reward
                reward = env.get_online_reward_per_arm(policy_arm_pulled[0], target_analysis_values, interaction)

                # Obtain the reward for each learner
                regret = env.get_regret_per_arm(reward, interaction)

                # Update each learner's state
                learner.update(arm_pulled, reward, regret)

        experiment_name = f'Policy_{policy}'

        # Write the results of the experiment
        row = [experiment_name, e, seed, percentage_usage_data, len(train_data),
               len(test_data), len(env.transactions_in_kb),
               policy,
               round(np.sum(learner.collected_rewards), 3),
               round(np.sum(learner.collected_regrets), 3),
               round(curr_policy_map_at_3_per_experiment / len(test_data), 3),
               round(curr_policy_map_at_1_per_experiment / len(test_data), 3)
               ]

        with open(result_file_path, 'a') as f:
            f.write(','.join(map(str, row)) + '\n')

        if experiment_name not in All_arms_pulled:
            All_arms_pulled[experiment_name] = learner.num_pulls_per_arm
        else:
            All_arms_pulled[experiment_name] += learner.num_pulls_per_arm

        # Save the results of the experiment
        curr_policy_reward_per_experiment.append(learner.collected_rewards)

        curr_policy_regret_per_experiment.append(learner.collected_regrets)

        curr_policy_relevance_per_experiment.append(curr_relevance)

    ########################################################################################################################
    # Save the results
    ########################################################################################################################
    mean_cum_reward_curr_policy = np.mean(curr_policy_reward_per_experiment, axis=0)

    std_cum_reward_curr_policy = np.std(curr_policy_reward_per_experiment, axis=0)

    mean_cum_regret_curr_policy = np.mean(curr_policy_regret_per_experiment, axis=0)

    std_cum_regret_curr_policy = np.std(curr_policy_regret_per_experiment, axis=0)

    mean_relevance_curr_policy = np.mean(curr_policy_relevance_per_experiment, axis=0)

    ########################################################################################################################
    # Plot the results
    ########################################################################################################################
    total_rounds = T_train + T_test

    # Cumulative reward
    reward_curr_policy_cum = np.cumsum(mean_cum_reward_curr_policy)
    regret_curr_policy_cum = np.cumsum(mean_cum_regret_curr_policy)
    relevance_curr_policy_cum = np.cumsum(mean_relevance_curr_policy)

    # Calculate standard deviations for fill_between
    reward_upper = reward_curr_policy_cum + std_cum_reward_curr_policy
    reward_lower = reward_curr_policy_cum - std_cum_reward_curr_policy
    regret_upper = regret_curr_policy_cum + std_cum_regret_curr_policy
    regret_lower = regret_curr_policy_cum - std_cum_regret_curr_policy

    # Add to the cumulative reward plot
    ax_cum_reward.plot(range(total_rounds), reward_curr_policy_cum, label=policy)
    ax_cum_reward.fill_between(range(total_rounds), reward_lower, reward_upper, alpha=0.2)
    ax_cum_reward.axvline(x=T_train, color='black', linestyle='--',
                      label='End of Training' if policy == policies[5] else "")
    ax_cum_reward.set_title(init_title + ' - Cumulative Reward')
    ax_cum_reward.set_xlabel('Rounds')
    ax_cum_reward.set_ylabel('Cumulative Reward')

    # Add to the cumulative regret plot
    ax_cum_regret.plot(range(total_rounds), regret_curr_policy_cum, label=policy)
    ax_cum_regret.fill_between(range(total_rounds), regret_lower, regret_upper, alpha=0.2)
    ax_cum_regret.axvline(x=T_train, color='black', linestyle='--',
                      label='End of Training' if policy == policies[5] else "")
    ax_cum_regret.set_title(init_title + ' - Cumulative Regret')
    ax_cum_regret.set_xlabel('Rounds')
    ax_cum_regret.set_ylabel('Cumulative Regret')

    # Add to the instant reward plot
    ax_inst_reward.plot(range(total_rounds), mean_cum_reward_curr_policy, label=policy)
    ax_inst_reward.fill_between(range(total_rounds), mean_cum_reward_curr_policy + std_cum_reward_curr_policy,
                                mean_cum_reward_curr_policy - std_cum_reward_curr_policy, alpha=0.2)
    ax_inst_reward.axvline(x=T_train, color='black', linestyle='--',
                      label='End of Training' if policy == policies[5] else "")
    ax_inst_reward.set_title(init_title + ' - Instant Reward')
    ax_inst_reward.set_xlabel('Rounds')
    ax_inst_reward.set_ylabel('Instant Reward')

    # Add to the instant regret plot
    ax_inst_regret.plot(range(total_rounds), mean_cum_regret_curr_policy, label=policy)
    ax_inst_regret.fill_between(range(total_rounds), mean_cum_regret_curr_policy + std_cum_regret_curr_policy,
                                mean_cum_regret_curr_policy - std_cum_regret_curr_policy, alpha=0.2)
    ax_inst_regret.axvline(x=T_train, color='black', linestyle='--',
                      label='End of Training' if policy == policies[5] else "")
    ax_inst_regret.set_title(init_title + ' - Instant Regret')
    ax_inst_regret.set_xlabel('Rounds')
    ax_inst_regret.set_ylabel('Instant Regret')

    # Add to the accuracy plot
    ax_accuracy.plot(range(T_test), relevance_curr_policy_cum, label=policy)
    ax_accuracy.set_title(init_title + ' - Cumulative Relevance')
    ax_accuracy.set_xlabel('Rounds')
    ax_accuracy.set_ylabel('Cumulative Relevance')

# Add legends outside of the for loop to avoid redundancy
lgd_cum_reward = ax_cum_reward.legend(loc='center left', bbox_to_anchor=(1, 0.5))
lgd_cum_regret = ax_cum_regret.legend(loc='center left', bbox_to_anchor=(1, 0.5))
lgd_inst_reward = ax_inst_reward.legend(loc='center left', bbox_to_anchor=(1, 0.5))
lgd_inst_regret = ax_inst_regret.legend(loc='center left', bbox_to_anchor=(1, 0.5))
lgd_accuracy = ax_accuracy.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Save the plots after completing all policies
output_file_path = os.path.join(directory, init_file_name + "_Img_CumRewardPlot.pdf")
fig_cum_reward.savefig(output_file_path, bbox_extra_artists=(lgd_cum_reward,), bbox_inches='tight')
output_file_path = os.path.join(directory, init_file_name + "_Img_CumRegPlot.pdf")
fig_cum_regret.savefig(output_file_path, bbox_extra_artists=(lgd_cum_regret,), bbox_inches='tight')
output_file_path = os.path.join(directory, init_file_name + "_Img_InstRewPlot.pdf")
fig_inst_reward.savefig(output_file_path, bbox_extra_artists=(lgd_inst_reward,), bbox_inches='tight')
output_file_path = os.path.join(directory, init_file_name + "_Img_InstRegPlot.pdf")
fig_inst_regret.savefig(output_file_path, bbox_extra_artists=(lgd_inst_regret,), bbox_inches='tight')
output_file_path = os.path.join(directory, init_file_name + "_Img_RelevancePlot.pdf")
fig_accuracy.savefig(output_file_path, bbox_extra_artists=(lgd_accuracy,), bbox_inches='tight')


# Most picked arms plot
fig = plt.figure()
ax = fig.add_subplot(111)
x = np.arange(n_arms)
bar_width = 0.1  # Width of each bar
num_keys = len(All_arms_pulled.keys())
bar_distance = 0.5  # Initial distance between bars for different keys
for i, key in enumerate(All_arms_pulled.keys()):
    key_bar_distance = bar_distance / (num_keys - 1)  # Adjusted distance for bars of the same key
    ax.bar(x + (i - (num_keys - 1) / 2) * key_bar_distance, All_arms_pulled[key], width=bar_width, label=key)
lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_title(init_title + ' - Arms Distribution')
ax.set_xlabel('Arm')
ax.set_ylabel('Average Number of Pulls')
ax.set_xticks(x)
ax.set_xticklabels([mapping_ArmToMethod[i] for i in range(n_arms)], rotation=45, ha='right')  # Rotate labels
output_file_path = os.path.join(directory, init_file_name + "_Img_MostPickedArms.pdf")
fig.savefig(output_file_path, bbox_extra_artists=(lgd,), bbox_inches='tight')


