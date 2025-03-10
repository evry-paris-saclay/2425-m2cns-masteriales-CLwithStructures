import numpy as np
import matplotlib.pyplot as plt

def errVSsqh(sequential_heterogeneity, average_errors):
    """Plot Average Error vs Sequential Heterogeneity with labeled data points."""
    plt.figure(figsize=(8, 5))
    plt.scatter(sequential_heterogeneity.values(), average_errors.values(), color='blue', label='Tasks')

    # Add labels for each point
    for task, (x, y) in zip(sequential_heterogeneity.keys(), zip(sequential_heterogeneity.values(), average_errors.values())):
        plt.text(x, y, task, fontsize=9, ha='right', va='bottom', color='black')

    # Fit linear regression line
    x_values = np.array(list(sequential_heterogeneity.values()))
    y_values = np.array(list(average_errors.values()))
    m, b = np.polyfit(x_values, y_values, 1)
    plt.plot(x_values, m * x_values + b, color='red', linestyle="--", label="Linear Fit")

    # Labels and title
    plt.xlabel("Sequential Heterogeneity")
    plt.ylabel("Average Error Rate")
    plt.title("Average Error vs. Sequential Heterogeneity")
    plt.legend()
    plt.grid(True)

    # Save plot
    plt.savefig("errVSsqh.png", dpi=300)

    # Display the plot
    plt.show()


def errVStc(total_complexity, average_errors):
    """Plot Average Error vs Total Complexity with labeled data points."""
    plt.figure(figsize=(8, 5))
    plt.scatter(total_complexity.values(), average_errors.values(), color='purple', label='Tasks')

    # Add labels for each point
    for task, (x, y) in zip(total_complexity.keys(), zip(total_complexity.values(), average_errors.values())):
        plt.text(x, y, task, fontsize=9, ha='right', va='bottom', color='black')

    # Fit linear regression line
    x_values = np.array(list(total_complexity.values()))
    y_values = np.array(list(average_errors.values()))
    m, b = np.polyfit(x_values, y_values, 1)
    plt.plot(x_values, m * x_values + b, color='red', linestyle="--", label="Linear Fit")

    # Labels and title
    plt.xlabel("Total Complexity")
    plt.ylabel("Average Error Rate")
    #plt.title("Average Error vs. Total Complexity")
    plt.legend()
    plt.grid(True)

    # Save plot
    plt.savefig("errVStc.png", dpi=300)

    # Display the plot
    plt.show()
