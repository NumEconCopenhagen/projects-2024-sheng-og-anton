from scipy import optimize 

class MarketClearPriceClass:
    
    def utility_A(self, x1A, x2A, alpha):  
        # Utility function for consumer A
        return x1A ** alpha * x2A ** (1 - alpha)

    def utility_B(self, x1B, x2B, beta):  
        # Utility function for consumer B
        return x1B ** beta * x2B ** (1 - beta)

    def demand_A(self, p1, w1A, w2A, alpha):  
        # Demand function for consumer A
        p2 = 1
        x1A_star = alpha * p1 * w1A + p2 * w2A
        x2A_star = (1 - alpha) * p1 * w1A + p2 * w2A
        return x1A_star, x2A_star

    def demand_B(self, p1, w1A, w2A, beta):  
        # Demand function for consumer B
        p2 = 1
        x1B_star = beta * p1 * (1 - w1A) + p2 * (1 - w2A)
        x2B_star = (1 - beta) * p1 * (1 - w1A) + p2 * (1 - w2A)
        return x1B_star, x2B_star

    def market_clearing_condition(self, prices, endowments, alphas, betas):  
        # Market clearing condition based on demand and endowments
        p1, p2 = prices  
        w1A, w2A = endowments
        alpha, beta = alphas, betas

        # Calculate demand quantities for both consumers
        x1A_star, x2A_star = self.demand_A(p1, w1A, w2A, alpha)
        x1B_star, x2B_star = self.demand_B(p1, w1A, w2A, beta)

        # Calculate excess demand (eps) for both goods
        eps1 = x1A_star - w1A + x1B_star - (1 - w1A)
        eps2 = x2A_star - w2A + x2B_star - (1 - w2A)

        return eps1, eps2

    def objective_function(self, endowments, alphas, betas, N):
        # Objective function to minimize squared errors
        total_squared_errors = 0
        rho1 = [0.5 + 2 * i / N for i in range(N + 1)]  # Generate rho1 values
        for p1 in rho1:
            prices = (p1, 1)  # Set prices for each iteration
            eps1, eps2 = self.market_clearing_condition(prices, endowments, alphas, betas)
            total_squared_errors += eps1 ** 2 + eps2 ** 2  # Accumulate squared errors
        return total_squared_errors

