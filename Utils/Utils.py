import pandas as pd
import numpy as np
import random
from tqdm.auto import tqdm

def create_train_dataset(full_dataset, data_object_to_test):
    training_dataset = full_dataset[full_dataset['data_object'] != data_object_to_test]

    # Shuffle the full dataset
    training_dataset = training_dataset.sample(frac=1).reset_index(drop=True)

    return training_dataset

def get_ground_truth_technique(interaction):
    best_performing_algorithm = interaction['technique_completeness'].values[0]
    return best_performing_algorithm


def create_train_dataset_with_knowledge(num_data_objects_knowledge):
    og_dataset = pd.read_csv("../Datasets/datasetCreated/results.csv")
    data_objects = og_dataset['data_object'].unique()

    # Extract randomly from data_objects
    data_objects_in_knowledge = np.random.choice(data_objects, num_data_objects_knowledge, replace=False)
    #data_objects_in_knowledge = np.array(data_objects[0])

    data_objects_to_train = data_objects[~np.isin(data_objects, data_objects_in_knowledge)]

    # Training dataset = og_dataset - data_objects_in_knowledge
    training_dataset = og_dataset[og_dataset['data_object'].isin(data_objects_to_train)]
    training_dataset = training_dataset.sample(len(training_dataset)).reset_index(drop=True)

    return training_dataset, data_objects_in_knowledge

def create_experiment_dataset_with_knowledge(num_data_objects_knowledge):
    og_dataset = pd.read_csv("../Datasets/datasetCreated/results.csv")
    data_objects = og_dataset['data_object'].unique()

    # Extract randomly from data_objects
    data_objects_in_knowledge = np.random.choice(data_objects, num_data_objects_knowledge, replace=False)

    data_objects_for_experiment = data_objects[~np.isin(data_objects, data_objects_in_knowledge)]

    # Training dataset = og_dataset - data_objects_in_knowledge
    experiment_dataset = og_dataset[og_dataset['data_object'].isin(data_objects_for_experiment)]

    # Shuffle the experiment dataset (to get different imputation tecniques as first appearance)
    experiment_dataset = experiment_dataset.sample(len(experiment_dataset)).reset_index(drop=True)

    # Remove duplicates of data_object, initial_completeness, ml_algorithm triples
    experiment_dataset = experiment_dataset[
        experiment_dataset[['data_object', 'initial_completeness', 'ml_algorithm']].duplicated() == False]

    return experiment_dataset, data_objects_in_knowledge

def create_knowledge_base_OLD(data_objects_in_knowledge):
    og_kb_results = pd.read_csv("../Datasets/datasetCreated/kb_results.csv")
    og_kb_profiling = pd.read_csv("../Datasets/datasetCreated/kb_profiling.csv")
    initial_knowledge_base = og_kb_results[og_kb_results['subject'].isin(data_objects_in_knowledge)]
    initial_profiling_knowledge_base = og_kb_profiling[og_kb_profiling['subject'].isin(data_objects_in_knowledge)]
    initial_knowledge_base = pd.concat([initial_knowledge_base, initial_profiling_knowledge_base], axis=0)
    initial_knowledge_base.reset_index(drop=True, inplace=True)
    return initial_knowledge_base

def create_knowledge_base_table_OLD(data_objects_in_knowledge):
    og_dataset = pd.read_csv("../Datasets/datasetCreated/results.csv")
    og_profiling = pd.read_csv("../Datasets/datasetCreated/profiling.csv")
    knowledge_base_table = pd.DataFrame(columns=['data_object', 'n_tuples', 'missing_perc', 'uniqueness', 'min',
                                                 'max', 'mean', 'median', 'std', 'skewness', 'kurtosis', 'mad', 'iqr',
                                                 'p_min', 'p_max', 'k_min', 'k_max', 's_min', 's_max', 'entropy',
                                                 'density', 'technique_completeness', 'final_ml_value'])
    index = 0

    for data_object in data_objects_in_knowledge:
        results = og_dataset[og_dataset['data_object'] == data_object]
        profile = og_profiling[og_profiling['data_object'] == data_object]

        #results = results.sample(3).reset_index(drop=True)

        for i in range(len(results)):
            interaction = results.iloc[i]
            technique_completeness = interaction["technique_completeness"]
            final_ml_value = interaction["final_ml_value"]
            knowledge_base_table_features = []
            features = profile.values[0].tolist()
            subject = str(index) + "_" + features[0]
            features[0] = subject
            knowledge_base_table_features.extend(features)
            knowledge_base_table_features.append(technique_completeness)
            knowledge_base_table_features.append(final_ml_value)
            knowledge_base_table.loc[index] = [knowledge_base_table_features[i] for i in
                                                                  range(len(knowledge_base_table_features))]
            index += 1

    return knowledge_base_table

def create_knowledge_base(transactions_in_kb):
    og_profiling = pd.read_csv("../Datasets/datasetCreated/profiling.csv")

    property_names = ['n_tuples', 'missing_perc', 'uniqueness', 'min',
                      'max', 'mean', 'median', 'std', 'skewness', 'kurtosis', 'mad', 'iqr',
                      'p_min', 'p_max', 'k_min', 'k_max', 's_min', 's_max', 'entropy',
                      'density']

    knowledge_base_table = pd.DataFrame(columns=['data_object', 'n_tuples', 'missing_perc', 'uniqueness', 'min',
                                                 'max', 'mean', 'median', 'std', 'skewness', 'kurtosis', 'mad', 'iqr',
                                                 'p_min', 'p_max', 'k_min', 'k_max', 's_min', 's_max', 'entropy',
                                                 'density', 'technique_completeness', 'final_ml_value'])
    knowledge_base = pd.DataFrame(columns=["subject", "predicate", "name_predicate", "value_predicate", "object"])


    index = 0
    index_kb = 0
    index_name_kb = 0

    for i in tqdm(range(len(transactions_in_kb))):
        interaction = transactions_in_kb.iloc[i]
        profile = og_profiling[og_profiling["data_object"] == interaction["data_object"]]
        profile = profile[profile["missing_perc"] == interaction["initial_completeness"]]

        # Data Object -> Data Analysis Application
        subject = str(index_name_kb) + "_" + str(interaction["data_object"])
        predicate_1 = "performanceValue"
        name_predicate_1 = "F1"
        value_predicate_1 = interaction["final_ml_value"]
        object_1 = interaction["ml_algorithm"]

        knowledge_base.loc[index_kb] = [subject, predicate_1, name_predicate_1, value_predicate_1, object_1]
        index_kb += 1

        # Data Object -> Data Preparation Method
        predicate_2 = "isGenerated"
        name_predicate_2 = None
        value_predicate_2 = float("NaN")
        object_2 = interaction["technique_completeness"]

        knowledge_base.loc[index_kb] = [subject, predicate_2, name_predicate_2, value_predicate_2, object_2]
        index_kb += 1

        # Data Object -> Data Quality Metric
        predicate_3 = "metricValue"
        name_predicate_3 = None
        value_predicate_3 = interaction["initial_completeness"]
        object_3 = "%missing_values"

        knowledge_base.loc[index_kb] = [subject, predicate_3, name_predicate_3, value_predicate_3, object_3]
        index_kb += 1

        # Data Object -> Data Quality Metric
        subject = str(index_name_kb) + "_" + str(interaction["data_object"])
        predicate = "propertyValue"
        name_predicate = None

        for j in range(len(property_names)):
            value_predicate = profile[property_names[j]].values[0]
            object = property_names[j]
            knowledge_base.loc[index_kb] = [subject, predicate, name_predicate, value_predicate, object]
            index_kb += 1

        index_name_kb += 1

        technique_completeness = interaction["technique_completeness"]
        final_ml_value = interaction["final_ml_value"]
        knowledge_base_table_features = []
        features = profile.values[0].tolist()
        subject = str(index) + "_" + features[0]
        features[0] = subject
        knowledge_base_table_features.extend(features)
        knowledge_base_table_features.append(technique_completeness)
        knowledge_base_table_features.append(final_ml_value)
        knowledge_base_table.loc[index] = [knowledge_base_table_features[i] for i in
                                           range(len(knowledge_base_table_features))]
        index += 1

    return knowledge_base, knowledge_base_table

def get_testing_triples(data_object_to_test):
    og_dataset = pd.read_csv("../Datasets/datasetCreated/results.csv")
    unique_triples = og_dataset[['data_object', 'initial_completeness', 'ml_algorithm']].drop_duplicates()
    testing_triples = unique_triples[unique_triples['data_object'] == data_object_to_test]

    return testing_triples


def create_traintest_split_with_knowledge(percentage_data, num_interactions_in_kb, seed):
    og_dataset = pd.read_csv("../Datasets/datasetCreated/results.csv")
    #og_dataset = pd.read_csv("datasetCreated/results_selected.csv")

    # Extract randomly from all transactions
    transactions_in_KB = og_dataset.sample(num_interactions_in_kb, random_state=seed)

    # Experiment dataset = og_dataset - transactions in knowledge base
    experiment_dataset = og_dataset[~og_dataset.index.isin(transactions_in_KB.index)]

    num_data_experiment = int(len(experiment_dataset) * percentage_data)
    experiment_dataset = experiment_dataset.sample(num_data_experiment, random_state=seed).reset_index(drop=True)

    # 80% of the data is used for training and 20% for testing
    num_data_test = int(num_data_experiment * 0.2)
    test_dataset = experiment_dataset.sample(num_data_test, random_state=seed).reset_index(drop=True)
    train_dataset = experiment_dataset[~experiment_dataset.index.isin(test_dataset.index)].reset_index(drop=True)

    return train_dataset, test_dataset, transactions_in_KB

def create_traintest_split_with_knowledge_LOO(percentage_data, num_interactions_in_kb, seed, dataset_name_test):
    '''
    all_datasets_name = ['abalone', 'BachChoralHarmon', 'bank', 'cancer',
                         'default of credit card clients', 'drug', 'electricity-normalized',
                         'fried', 'frogs', 'german', 'house', 'iris', 'letter', 'mv',
                         'phoneme', 'ringnorm', 'Run_or_walk_information', 'shuttle',
                         'stars', 'visualizing_soil', 'wall-robot-navigation',
                         'numerai28.6']
    '''
    og_dataset = pd.read_csv("../Datasets/datasetCreated/results.csv")
    #dataset_name_test = random.choice(all_datasets_name)
    #dataset_name_test = all_datasets_name[seed]
    #dataset_name = dataset_name_to_test
    test_dataset_full = og_dataset[og_dataset['data_object'].str.contains(dataset_name_test)]
    test_dataset_full = test_dataset_full.sample(len(test_dataset_full)).reset_index(drop=True)
    test_dataset = test_dataset_full.drop_duplicates(subset=['data_object', 'initial_completeness', 'ml_algorithm'])
    test_dataset = test_dataset.sample(len(test_dataset)).reset_index(drop=True)

    experiment_dataset = og_dataset[~og_dataset['data_object'].str.contains(dataset_name_test)]

    # Extract randomly from all transactions
    transactions_in_KB = experiment_dataset.sample(num_interactions_in_kb, random_state=seed)

    # Experiment dataset = og_dataset - transactions in knowledge base
    experiment_dataset = experiment_dataset[~experiment_dataset.index.isin(transactions_in_KB.index)]

    # 80% of the data is used for training
    num_data_experiment = int(len(experiment_dataset) * percentage_data)
    num_data_train = int(num_data_experiment * 0.8)
    train_dataset = experiment_dataset.sample(num_data_train, random_state=seed).reset_index(drop=True)

    return train_dataset, test_dataset, transactions_in_KB


def create_traintest_split_with_knowledge_KBTrainingImpact(percentage_data, num_interactions_in_kb, seed):
    og_dataset = pd.read_csv("../Datasets/datasetCreated/results.csv")

    # Extract randomly from all transactions
    transactions_in_KB = og_dataset.sample(num_interactions_in_kb, random_state=seed)

    # Train dataset = transactions_in_kb
    train_dataset = transactions_in_KB

    # Experiment dataset = og_dataset - transactions in knowledge base
    experiment_dataset = og_dataset[~og_dataset.index.isin(train_dataset.index)]

    num_data_experiment = int(len(experiment_dataset) * percentage_data)

    # 80% of the data is used for training and 20% for testing
    num_data_test = int(num_data_experiment * 0.2)
    test_dataset = experiment_dataset.sample(num_data_test, random_state=seed).reset_index(drop=True)

    return train_dataset, test_dataset, transactions_in_KB





