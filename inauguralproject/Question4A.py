import numpy as np
from scipy import optimize

# we create the class
class UtilityOptimization:
    def __init__(self, wA1, wA2, N, alpha, p2):
        self.wA1 = wA1
        self.wA2 = wA2
        self.wB1 = 1 - wA1
        self.wB2 = 1 - wA2
        self.N = N
        self.alpha = alpha
        self.p2 = p2
    
    # We define the utility function
    def utility_A(self, p1):
        return -(1 - (1 - self.wB1) / p1) * (1 - (1 - self.wB2))

    # We define the objective function
    def maximize_A_utility(self, p1):
        return -self.utility_A(p1)  # Negative sign because we are maximizing

    # We define the price vector
    def find_optimal_price_and_allocation(self):
        P1 = [0.5 + 2 * i / self.N for i in range(self.N + 1)]

        max_utility = -np.inf
        best_price = None

        # We find the optimal price
        for p1 in P1:
            constraints = [{'type': 'ineq', 'fun': lambda p1: p1}]
            bounds = [(0, 2.5)]  
            solution = optimize.minimize(self.maximize_A_utility, [0.5], args=(self.wB1, self.wB2), bounds=bounds, constraints=constraints)
            if solution.success:
                utility = -solution.fun
                if utility > max_utility:
                    max_utility = utility
                    best_price = solution.x[0]
        # We find the optimal allocation
        xA1_optimal = self.alpha * (best_price*self.wA1 + self.p2*self.wA2) / best_price
        xA2_optimal = (1-self.alpha) * (best_price*self.wA1 + self.p2*self.wA2) / self.p2

        return best_price, xA1_optimal, xA2_optimal
