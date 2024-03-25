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
        plt.figure(figsize=(8, 6))
        plt.scatter(wA1, wA2, c='green')
        plt.title('The set of endowments with 50 elements')
        plt.xlabel('ωA1')
        plt.ylabel('ωA2')
        plt.grid(True)
        plt.show()
