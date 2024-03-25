# Question 4.B
# 1. We import the two packages
import numpy as np
from scipy import optimize

# 2. We create the class
class OptimizationWithNoUpperBound:
    
    # Utility function for consumer A
    def utility_A(x1A, x2A, alpha=1/3):
        return x1A ** alpha * x2A ** (1 - alpha)
    # We write the demand function
    def demand_A(p1, p2=1, w1A=0.8, w2A=0.3, alpha=1/3):
        # Demand function for consumer A
        x1A_optimal = alpha * (p1 * w1A + p2 * w2A) / p1
        x2A_optimal = (1 - alpha) * p1 * w1A + p2 * w2A
        return x1A_optimal, x2A_optimal
    # We maximize utility for consumer A
    def maximize_A_utility(price, alpha=1/3, w1B=0.2, w2B=0.7):
        p1 = price[0]  
        x1A, x2A = demand_A(p1)
        return -utility_A(x1A, x2A, alpha)
