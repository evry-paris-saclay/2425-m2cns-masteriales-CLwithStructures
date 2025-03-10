from utils import calculate_sequential_heterogeneity, calculate_total_complexity 
from utilsPlots import errVSsqh, errVStc
import numpy as np
import pickle
# Given accuracies for each task
task_accuracies = {
    "Task 1": [0.6662, 0.35, 0.4157, 0.382, 0.4761, 0.5631, 0.776, 0.9717],
    "Task 2": [0.4226, 0.4162, 0.4287, 0.4937, 0.4379, 0.7836, 0.7716, 0.9696],
    "Task 3": [0.8887, 0.5954, 0.3861, 0.3021, 0.3501, 0.3842, 0.3742, 0.38, 0.4317, 0.4556, 0.5124, 0.597, 0.7887, 0.9405, 0.9715],
    "Task 4": [0.5216, 0.917, 0.4991, 0.6796, 0.7127, 0.5138, 0.7298, 0.488, 0.4578, 0.6907, 0.8982, 0.6737, 0.626, 0.721, 0.9679],
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