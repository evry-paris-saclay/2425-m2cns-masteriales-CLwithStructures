import numpy as np
import matplotlib.pyplot as plt

def errVSsqh(sequential_heterogeneity, average_errors):
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.scatter(sequential_heterogeneity.values(), average_errors.values(), color='blue', label='Data points')

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

    # save plot
    plt.savefig("errVSsqh.png", dpi=300)

    # Display the plot
    plt.show()


def errVStc(total_complexity, average_errors):
    # Plotting total heterogeneity vs. average error rate
    plt.figure(figsize=(8, 5))
    plt.scatter(total_complexity.values(), average_errors.values(), color='purple', label='Data points')

    # Fit linear regression line
    x_values = np.array(list(total_complexity.values()))
    y_values = np.array(list(average_errors.values()))
    m, b = np.polyfit(x_values, y_values, 1)
    plt.plot(x_values, m * x_values + b, color='red', linestyle="--", label="Linear Fit")

    # Labels and title
    plt.xlabel("Total Complexity")
    plt.ylabel("Average Error Rate")
    plt.title("Average Error vs. Total Complexity")
    plt.legend()
    plt.grid(True)

    # save plot
    plt.savefig("errVStc.png", dpi=300)

    # Display the plot
    plt.show()