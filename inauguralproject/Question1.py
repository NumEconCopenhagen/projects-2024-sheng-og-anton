import matplotlib.pyplot as plt

class EdgeworthBoxClass:
    def __init__(self):
        # Initial endowment
        self.w_A1 = 0.8
        self.w_A2 = 0.3
        self.w_B1 = 1 - self.w_A1
        self.w_B2 = 1 - self.w_A2

        # Utility function parameters
        self.alpha = 1/3
        self.beta = 2/3

        # Number of allocations
        self.N = 75

    def u_A(self, x_A1, x_A2):
        # Define the utility function for Consumer A
        return x_A1**self.alpha * x_A2**(1 - self.alpha)

    def u_B(self, x_B1, x_B2):
        # Define the utility function for Consumer B
        return x_B1**self.beta * x_B2**(1 - self.beta)

    def pareto_improvements(self):
        pareto_improvements = []
        # Using a nested for loop to find and define x_A1 and x_A2
        for i in range(self.N + 1):
            x_A1 = i / self.N
            for j in range(self.N + 1):
                x_A2 = j / self.N

                # Calculate x_B1 and x_B2 based on x_A1 and x_A2
                x_B1 = 1 - x_A1
                x_B2 = 1 - x_A2

                # Checking the Pareto improvement relative to the endowment
                if self.u_A(x_A1, x_A2) >= self.u_A(self.w_A1, self.w_A2) and \
                        self.u_B(x_B1, x_B2) >= self.u_B(self.w_B1, self.w_B2) and \
                        x_B1 == 1 - x_A1 and x_B2 == 1 - x_A2:
                    # Storing combination of x_A1 and x_A2.
                    pareto_improvements.append((x_A1, x_A2))

        return pareto_improvements

    def plot_edgeworth_box(self):
        result = self.pareto_improvements()

        # Plot the Edgeworth box with Pareto improvements
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

# Create an instance of EdgeworthBox
box = EdgeworthBoxClass()

# Plot the Edgeworth box with Pareto improvements for set C
box.plot_edgeworth_box()
