# Question 4.A

# 1. We import the optimizer
from scipy import optimize 

# 2. We create the class
class MaximizationConsumerA:
    
    # a. We define the parameters 
    def __init__(self, alpha=1/3, w1A=0.8, w2A=0.3):
        self.alpha = alpha
        self.w1A = w1A
        self.w2A = w2A
    
    # b. We define the utility function for consumer A
    def utility_function_a(self, x1A, x2A):
        return x1A ** self.alpha * x2A ** (1 - self.alpha)

    # c. We define the demand function for consumer A
    def demand_function_a(self, p1, p2=1):
        x1A_optimal = self.alpha * (p1 * self.w1A + p2 * self.w2A) / p1
        x2A_optimal = (1 - self.alpha) * (p1 * self.w1A + p2 * self.w2A) / p2
        return x1A_optimal, x2A_optimal

    # d. We define the objective function
    def utility_a_maximization(self, p1):
        x1A, x2A = self.demand_function_a(p1)
        return -self.utility_function_a(x1A, x2A) 
