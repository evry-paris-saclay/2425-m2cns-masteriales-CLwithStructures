def plotAEvsTC(total_complexity_values,average_error_values):
    # Scatter plot: Average Error vs Total Complexity
    plt.figure(figsize=(8, 6))
    plt.scatter(total_complexity_values, average_error_values, color='blue', alpha=0.7, label="Task Sequences")

    # Trendline (linear regression fit)
    z = np.polyfit(total_complexity_values, average_error_values, 1)
    p = np.poly1d(z)
    plt.plot(total_complexity_values, p(total_complexity_values), "r--", label="Trendline")

    # Labels and Titles
    plt.xlabel("Total Complexity")
    plt.ylabel("Average Error")
    plt.title("Average Error as a Function of Total Complexity")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()