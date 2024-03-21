from scipy.optimize import minimize

class AgentOptimization:
    def __init__(self, w_A1, w_A2, x0, alpha, bounds):
        self.w_A1 = w_A1
        self.w_A2 = w_A2
        self.x0 = x0
        self.alpha = alpha
        self.bounds = bounds

    def utility_A(self, x_A):
        return -(x_A[0] ** self.alpha * x_A[1] ** (1 - self.alpha))  # We use negative because we want to maximize

    def objective(self, x_A):
        return self.utility_A(x_A)

    def constraint1(self, x_A):
        return self.utility_A(x_A) - self.utility_A((self.w_A1, self.w_A2))

    def constraint2(self, x_A):
        x_B1 = 1 - x_A[0]
        x_B2 = 1 - x_A[1]
        return x_B1 - x_B2  # Constraint x_B1 = 1 - x_A1 and x_B2 = 1 - x_A2

    def solve(self):
        constraints = [{'type': 'ineq', 'fun': self.constraint1}, {'type': 'eq', 'fun': self.constraint2}]
        result = minimize(self.objective, self.x0, bounds=self.bounds, constraints=constraints)
        return result.x, -result.fun  # Return optimal allocation and utility

