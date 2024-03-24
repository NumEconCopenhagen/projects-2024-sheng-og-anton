# 1. We import the two packages
import numpy as np
import matplotlib.pyplot as plt

# 2. We create the class 
class TotalEndowments:
    
    # a. We generate a seed
    def random_numbers(self, seed=69, n=50):
        np.random.seed(seed) # We set a seed, so that we get the same pseduo-random numbers everytime
        self.n = n # We create an attribute for the class
    
    # b. We generate the random endowments
    def random_endowments(self):
        w1A = np.random.uniform(0, 1, self.n) # Generates random number for endowment 1A
        w2A = np.random.uniform(0, 1, self.n) # Generates random number for endowment 2A
        return w1A, w2A
    
    # c. We generate the plot
    def endowments_plot(self, wA1, wA2):
        plt.figure(figsize=(8, 6)) # Sets the size of the figure
        plt.scatter(w1A, w2A, c='green') # Creates a scatterplot
        plt.title('The set of endowments with 50 elements') # Create a title
        plt.xlabel('ωA1') # Labels the x-axis
        plt.ylabel('ωA2') # Labels the y-axis
        plt.grid(True) # Creates grids around the figure
        plt.show() # Shows the figure
