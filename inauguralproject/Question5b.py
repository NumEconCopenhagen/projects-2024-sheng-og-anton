from scipy.optimize import minimize

class AgentOptimization2:
    def __init__(self, x0, alpha):
        self.x0 = x0
        self.alpha = alpha
        self.bounds = [(0, 1), (0, 1)]  # Bounds for x_A1 and x_A2

    def utility_A(self, x_A):
        return -(x_A[0] ** self.alpha * x_A[1] ** (1 - self.alpha))  # We use negative because we want to maximize

    def objective(self, x_A):
        return self.utility_A(x_A)

    def solve(self):
        result = minimize(self.objective, self.x0, bounds=self.bounds)
        optimal_allocation = result.x
        optimal_utility = -result.fun  # Convert back to positive for utility
        return optimal_allocation, optimal_utility

