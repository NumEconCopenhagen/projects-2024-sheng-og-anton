import numpy as np
import matplotlib.pyplot as plt

class TotalEndowments:
    
    def __init__(self, seed=69, n=50):
        np.random.seed(seed)
        self.n = n
    
    def generate_random_endowments(self):
        wA1 = np.random.uniform(0, 1, self.n)
        wA2 = np.random.uniform(0, 1, self.n)
        return wA1, wA2
    
    def plot_endowments(self, wA1, wA2):
        plt.figure(figsize=(8, 6))
        plt.scatter(wA1, wA2, c='green')
        plt.title('The set of endowments with 50 elements')
        plt.xlabel('ωA1')
        plt.ylabel('ωA2')
        plt.grid(True)
        plt.show()
