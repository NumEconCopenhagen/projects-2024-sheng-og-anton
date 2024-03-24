# 1. We import the packages
import numpy as np
import matplotlib.pyplot as plt

# 2. We create the class
class OptimalAllocations:
    
    # a. We define the parameters
    def __init__(self, n=50, alpha=1/3, p2=1):
        np.random.seed(69)
        self.n = n
        self.alpha = alpha
        self.p1 = np.linspace(0.5, 2.5, n)
        self.omega_A1 = np.random.uniform(0, 1, n)
        self.omega_A2 = np.random.uniform(0, 1, n)
    
    # b. We define the set of optimal allocations
    def set_of_optimal_allocations(self, omega_A1, omega_A2):
        xA1 = self.alpha * (self.p1 * omega_A1 + p2 * omega_A2) / self.p1
        xA2 = (1 - self.alpha) * (self.p1 * omega_A1 + p2 * omega_A2) / p2
        return xA1, xA2

    # c. We generate the optimal allocations
    def allocations_generation(self):
        allocations = np.array([self.set_of_optimal_allocations(omega_A1, omega_A2) for omega_A1, omega_A2 in zip(self.omega_A1, self.omega_A2)])
        return allocations
    