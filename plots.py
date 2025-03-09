from utils import calculate_sequential_heterogeneity, calculate_total_complexity 
from utilsPlots import errVSsqh, errVStc
import numpy as np
import pickle
# Given accuracies for each task
task_accuracies = {
    "Task 1": [0.7991, 0.6498, 0.6582, 0.5536, 0.6937, 0.7597, 0.8137, 0.9838],
    "Task 2": [0.6613, 0.6607, 0.6618, 0.7549, 0.5516, 0.8228, 0.8026, 0.983],
    "Task 3": [0.94, 0.8091, 0.7501, 0.7243, 0.7285, 0.7457, 0.6928, 0.6884, 0.7368, 0.8008, 0.8101, 0.7945, 0.8151, 0.954, 0.9844],
    "Task 4": [0.8051, 0.9217, 0.6867, 0.8258, 0.8195, 0.7921, 0.8068, 0.742, 0.7851, 0.8565, 0.9478, 0.8129, 0.6994, 0.7618, 0.9848],
}

# Compute average error rates (1 - accuracy)
task_errors = {task: [1 - acc for acc in accs] for task, accs in task_accuracies.items()}
average_errors = {task: np.mean(errors) for task, errors in task_errors.items()}

with open('embeddings_EXPERIMENT_1.p', 'rb') as f:
    embeddings12 = pickle.load(f, encoding='latin1')

with open('embeddings_EXPERIMENT_3.p', 'rb') as f:
    embeddings34 = pickle.load(f, encoding='latin1')

new_order2 = [1, 3, 0, 4, 2, 7, 5, 6]
new_order4 = [7, 3, 11, 0, 14, 9, 2, 10, 8, 6, 5, 1, 12, 13, 4]

# Simulated sequential heterogeneity (randomized for visualization purposes)
sequential_heterogeneity = {
    "Task 1": calculate_sequential_heterogeneity(embeddings12[1]),
    "Task 2": calculate_sequential_heterogeneity(embeddings12[1], new_order2),
    "Task 3": calculate_sequential_heterogeneity(embeddings34[1]),
    "Task 4": calculate_sequential_heterogeneity(embeddings34[1], new_order4),
}
total_complexity = {
    "Task 1": calculate_total_complexity(embeddings12[1]),
    "Task 2": calculate_total_complexity(embeddings12[1], new_order2),
    "Task 3": calculate_total_complexity(embeddings34[1]),
    "Task 4": calculate_total_complexity(embeddings34[1], new_order4),
}
print("Total complexity:",total_complexity)
print("Sequential Heterogenrity:",sequential_heterogeneity)

errVStc(total_complexity, average_errors)
errVSsqh(sequential_heterogeneity, average_errors)