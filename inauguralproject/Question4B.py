# Question 4.B
# 1. We import the two packages
import numpy as np
from scipy import optimize

# 2. We create the class
class OptimizationWithNoUpperBound:
    
    # a. We define the parameters
    def __init__(self, wA1=0.8, wA2=0.3, N=75, p2=1):
        self.wA1 = wA1
        self.wA2 = wA2
        self.wB1 = 1 - wA1
        self.wB2 = 1 - wA2
        self.N = N
        self.p2 = p2
    
    # b. We define the utility function for consumer A
    def utility_A(self, p1, xA1, xA2, alpha=1/3):
        return xA1 ** alpha * xA2 ** (1 - alpha)

    # c. We define the objective function for the maximization problem
    def utility_A_maximization(self, p1, wB1, wB2):
        bounds = [(0.0000001, None)]
        x0 = [0.5]
        solution_to_4b = optimize.minimize(self.utility_A, x0, args=(p1, wB1, wB2), bounds=bounds)
        if solution_to_4b.success:
            return -solution_to_4b.fun
        return -np.inf

    # d. We define the optimal price for good 1
    def optimal_price(self):
        max_utility = -np.inf
        best_price = None
        P1 = np.linspace(0.5, 2.5, self.N)
        for p1 in P1:
            utility = self.utility_A_maximization(p1, self.wB1, self.wB2)
            if utility > max_utility:
                max_utility = utility
                p1_optimal = p1
        return p1_optimal

    # e. We calculate the optimal allocation for consumer A
    def calculate_optimal_allocation(self):
        best_price = self.find_best_price()
        optimal_allocation_A1 = self.alpha * (best_price * self.wA1 + self.p2 * self.wA2) / best_price
        optimal_allocation_A2 = (1 - self.alpha) * (best_price * self.wA1 + self.p2 * self.wA2) / self.p2
        return optimal_allocation_A1, optimal_allocation_A2
