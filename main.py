import math
from random import randint
import numpy as np
import pandas as pd
from pandas import ExcelWriter

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

xl_file = pd.ExcelFile("iris-data.xlsx")
dfs = pd.read_excel(xl_file)

k = 5

ratio_number = 4

result_list = []


def normalize_dataset(dataset):
    normalized_dataset = (dataset - dataset.min()) / (dataset.max() - dataset.min())

    writer = ExcelWriter('normalized.xlsx')
    normalized_dataset.to_excel(writer, 'normalized')
    writer.save()

    return normalized_dataset


def remove_random_values(dataset):
    for i in range(len(dataset)):
        dataset.iloc[i, randint(0, len(dataset.columns) - 1)] = math.nan

    return dataset


def create_test_set(dataset):
    test_dataset = dataset.iloc[0::ratio_number, :]
    train_dataset = dataset.drop(test_dataset.index)

    test_dataset = remove_random_values(test_dataset.copy())

    return train_dataset, test_dataset


def euclidean_distance(a, b, length, missing_value_index):
    sums = 0
    for i in range(length):
        if i != missing_value_index:
            sums += math.pow((a[i] - b[i]), 2)

    distance = math.sqrt(sums)
    return distance


def weighted_average(nearest_neighbours, weights, missing_value_index):
    weighted_values = 0
    total_weight = 0
    matching_values = 0
    for i in range(len(nearest_neighbours)):
        if weights[i] == 0:
            matching_values += nearest_neighbours.iloc[i, missing_value_index]
        else:
            weighted_values += nearest_neighbours.iloc[i, missing_value_index] * (1 / weights[i])
            total_weight += (1 / weights[i])
    if matching_values == 0:
        estimated_value = weighted_values / total_weight
    else:
        estimated_value = matching_values / k
    return estimated_value


def knn(training_set, test_instance, instance_index, k):
    length = len(test_instance)

    distances = [0 for x in range(len(training_set))]

    missing_value_index = np.argwhere(np.isnan(test_instance))[0][0]

    for i in range(len(training_set)):
        distances[i] = euclidean_distance(training_set.iloc[i, :], test_instance, length, missing_value_index)

    # sorts distances and index_connector lists synced to each other
    # so index_connector list keeps the indexes of sorted distances array
    # then it sorts training_set indexes according to index connector list
    index_connector = [i for i in range(len(training_set)) if i % ratio_number != 0]
    # print("************ Distances and indexes ************")
    # print(distances)
    # print(index_connector)
    # print("************ Sorted Distances and indexes ************")
    distances, index_connector = (list(t) for t in zip(*sorted(zip(distances, index_connector))))
    # print(distances)
    # print(index_connector)
    # print("************ test_instance and training_set ************")
    sorted_training_set = training_set.reindex(index_connector)
    # print(pd.DataFrame([test_instance]))
    # print(sorted_training_set)

    # print("************ test_instance ************\n", pd.DataFrame([test_instance]))
    nearest_neighbours = sorted_training_set.head(k)
    # print("\n************ nearest_neighbours ************\n", nearest_neighbours)
    # print("\n************ distances ************\n", pd.DataFrame(distances[0:k]))

    estimated_value = weighted_average(nearest_neighbours, distances[0:k], missing_value_index)

    error = abs(estimated_value - dfs.iloc[instance_index][missing_value_index])
    print("Test instance index : ", instance_index)
    print("Missing index : ", missing_value_index)
    print("Estimated value :", estimated_value)
    print("Real value      :", dfs.iloc[instance_index][missing_value_index])
    print("Error percentage: %", error * 100)
    result_list.append((instance_index, missing_value_index, estimated_value,
                        dfs.iloc[instance_index][missing_value_index], error * 100))


dfs = normalize_dataset(dfs)

dfs_train, dfs_test = create_test_set(dfs)

# print(dfs_train)
# print(dfs_test)


for i in range(len(dfs_test)):
    knn(dfs_train, dfs_test.iloc[i, :], i * ratio_number, k)
    print("############")

for result in result_list:
    print(result)


results = pd.DataFrame(result_list)
results.columns = ['Test instance index', 'Missing index', 'Estimated value', 'Real value', 'Error percentage']


writer = ExcelWriter('results.xlsx')
results.to_excel(writer, 'results')
writer.save()
