import numpy as np
import matplotlib.pyplot as plt

# 1. We create the class 
class TotalEndowments:
    # a. We generate the seed and number of elements
    def __init__(self, seed=69, n=50):
        np.random.seed(seed)
        self.n = n

    # b. We plot the figure
    def plot_endowments(self, wA1, wA2):
        result = self.pareto_improvements()

        # Plot the Edgeworth box with the endowments
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlabel("$x_1^A$") # setting x-axis label
        ax.set_ylabel("$x_2^A$") # setting y-axis label
        # Setting the limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Plotting endowment points
        ax.scatter(self.w_A1, self.w_A2, marker='s', color='black', label='Endowment A')
        ax.scatter(self.w_B1, self.w_B2, marker='s', color='red', label='Endowment B')

        # Plotting Pareto improvements
        for allocation in result:
            ax.scatter(allocation[0], allocation[1], color='green') 

        ax.legend() # adding legend
        plt.show() # display the plot

    