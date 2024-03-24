# 1. We import the two packages
from scipy import optimize
import numpy as np

# 2. We create the class
class UtilitarianSocialPlanner:
    # a. We define the parameters
    def parameters(self, alpha):
        self.alpha = alpha # The value of alpha
        self.beta = 2/3  # The value of beta
        
    # b. We define the utility function for consumer A
    def utility_A(self, xA1, xA2):
        return xA1**(self.alpha) * xA2**(1 - self.alpha)
    
    # c. We define the utility function for consumer B
    def utility_B(self, xB1, xB2):
        return xB1**(self.beta) * xB2**(1 - self.beta)
    
    # d. We aggregate the utility functions
    def aggregate_utility_functions(self, x):
        xA1, xA2 = x
        return -(self.u_A(xA1, xA2) + self.u_B(1 - xA1, 1 - xA2))
    
    # e. We find the optimal allocations for the utilitarian social planner
    def optimal_allocation(self):
        bounds = [(0, 1), (0, 1)] # Creates the bounds
        solution_to_socialplanner = optimize.minimize(self.aggregate_utility_functions, x0=[0.5, 0.5], method='SLSQP', bounds=bounds)
        
        if solution_to_socialplanner.success:
            xA1_optimal, xA2_optimal = solution_to_socialplanner.x
            xB1_optimal = 1 - xA1_optimal # Calculates the optimal allocation of xB1
            xB2_optimal = 1 - xA2_optimal # Calculates the optimal allocation of xB2
            print("The optimal allocation for the utilitarian social planner is the following:")
            print(f"((xA1_optimal,xA2_optimal),(xB1_optimal,xB2_optimal)) = (({xA1_optimal}, {xA2_optimal}), ({xB1_optimal}, {xB2_optimal}))")
        else:
            print("The optimal allocation for the utilitarian social planner was not found") # Print statement if optimization fails