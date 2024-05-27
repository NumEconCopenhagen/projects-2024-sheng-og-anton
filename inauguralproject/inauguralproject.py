import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace

from scipy import optimize 
from scipy.optimize import minimize


#Question 1
class EdgeworthBoxClass:
    def __init__(self):
        # Initial endowment
        self.w_A1 = 0.8
        self.w_A2 = 0.3
        self.w_B1 = 1 - self.w_A1
        self.w_B2 = 1 - self.w_A2

        # Utility function parameters
        self.alpha = 1/3
        self.beta = 2/3

        # Number of allocations
        self.N = 75

    def u_A(self, x_A1, x_A2):
        # Define the utility function for Consumer A
        return x_A1**self.alpha * x_A2**(1 - self.alpha)

    def u_B(self, x_B1, x_B2):
        # Define the utility function for Consumer B
        return x_B1**self.beta * x_B2**(1 - self.beta)

    def pareto_improvements(self):
        pareto_improvements = []
        # Using a nested for loop to find and define x_A1 and x_A2
        for i in range(self.N + 1):
            x_A1 = i / self.N
            for j in range(self.N + 1):
                x_A2 = j / self.N

                # Calculate x_B1 and x_B2 based on x_A1 and x_A2
                x_B1 = 1 - x_A1
                x_B2 = 1 - x_A2

                # Checking the Pareto improvement relative to the endowment
                if self.u_A(x_A1, x_A2) >= self.u_A(self.w_A1, self.w_A2) and \
                        self.u_B(x_B1, x_B2) >= self.u_B(self.w_B1, self.w_B2) and \
                        x_B1 == 1 - x_A1 and x_B2 == 1 - x_A2:
                    # Storing combination of x_A1 and x_A2.
                    pareto_improvements.append((x_A1, x_A2))

        return pareto_improvements

    def plot_edgeworth_box(self):
        result = self.pareto_improvements()

        # Plot the Edgeworth box with Pareto improvements
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlabel("$x_1^A$") # setting x-axis label
        ax.set_ylabel("$x_2^A$") # setting y-axis label
        # Setting the limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Plotting endowment points
        ax.scatter(self.w_A1, self.w_A2, marker='s', color='black', label='Endowment A')
        ax.scatter(self.w_B1, self.w_B2, marker='s', color='red', label='Endowment B')

        # Plotting Pareto improvements
        for allocation in result:
            ax.scatter(allocation[0], allocation[1], color='green') 

        ax.legend() # adding legend
        plt.show() # display the plot

# Create an instance of EdgeworthBox
box = EdgeworthBoxClass()

# Plot the Edgeworth box with Pareto improvements for set C
box.plot_edgeworth_box()

#Question 2 
from types import SimpleNamespace

class ErrorMarketClass:
    def __init__(self):
        par = self.par = SimpleNamespace()
        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3
        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3
        par.p2 = 1
        self.N = 75

    def utility_A(self, x1A, x2A):
        return x1A ** self.par.alpha * x2A ** (1 - self.par.alpha)

    def utility_B(self, x1B, x2B):
        return x1B ** self.par.beta * x2B ** (1 - self.par.beta)

    def demand_A(self, p1):
        x1A_star = self.par.alpha * (p1 * self.par.w1A + self.par.p2 * self.par.w2A) / p1
        x2A_star = (1 - self.par.alpha) * (p1 * self.par.w1A + self.par.p2 * self.par.w2A) / self.par.p2
        return x1A_star, x2A_star

    def demand_B(self, p1):
        x1B_star = self.par.beta * (p1 * (1 - self.par.w1A) + self.par.p2 * (1 - self.par.w2A)) / p1
        x2B_star = (1 - self.par.beta) * (p1 * (1 - self.par.w1A) + self.par.p2 * (1 - self.par.w2A)) / self.par.p2
        return x1B_star, x2B_star

    def check_market_clearing(self):
        par = self.par
        self.rho1 = [0.5 + 2 * i / self.N for i in range(self.N + 1)]

        errors = []

        for p1 in self.rho1:
            x1A_star, x2A_star = self.demand_A(p1)
            x1B_star, x2B_star = self.demand_B(p1)
            eps1 = x1A_star - par.w1A + x1B_star - (1 - par.w1A)
            eps2 = x2A_star - par.w2A + x2B_star - (1 - par.w2A)
            errors.append((eps1, eps2))

        return errors

errormarket = ErrorMarketClass()
result = errormarket.check_market_clearing()

print("Errors in the market clearing condition:")
for eps1, eps2 in result:
    print(f"Error 1: {eps1}, Error 2: {eps2}")


#Question 3
class MarketClearPriceClass:
    
    def utility_A(self, x1A, x2A, alpha):  
        # Utility function for consumer A
        return x1A ** alpha * x2A ** (1 - alpha)

    def utility_B(self, x1B, x2B, beta):  
        # Utility function for consumer B
        return x1B ** beta * x2B ** (1 - beta)

    def demand_A(self, p1, w1A, w2A, alpha):  
        # Demand function or consumer A
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

#Question 4A
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

    def demand_A(p1, p2=1, w1A=0.8, w2A=0.3, alpha=1/3):
        # Demand function for consumer A
        x1A_optimal = alpha * (p1 * w1A + p2 * w2A) / p1
        x2A_optimal = (1 - alpha) * p1 * w1A + p2 * w2A
        return x1A_optimal, x2A_optimal

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

#Question 4B
class OptimizationWithNoUpperBound:
    
    # Utility function for consumer A
    def utility_A(self, x1A, x2A, alpha=1/3):
        return x1A ** alpha * x2A ** (1 - alpha)
    # We write the demand function
    def demand_A(self, p1, p2=1, w1A=0.8, w2A=0.3, alpha=1/3):
        # Demand function for consumer A
        x1A_optimal = alpha * (p1 * w1A + p2 * w2A) / p1
        x2A_optimal = (1 - alpha) * p1 * w1A + p2 * w2A
        return x1A_optimal, x2A_optimal
    # We maximize utility for consumer A
    def maximize_A_utility(self, price, alpha=1/3, w1B=0.2, w2B=0.7):
        p1 = price[0]  
        x1A, x2A = self.demand_A(p1)
        return -self.utility_A(x1A, x2A, alpha)


#Question 5a
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


#Question 5b
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


#Question 6a
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
        return -(self.utility_A(xA1, xA2) + self.utility_B(1 - xA1, 1 - xA2))
    
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


#Question 6b
class UtilitarianSocialPlannerEdgeworthBox:
    # a. We define the values for w1bar and w2bar
    def __init__(self, w1bar=1.0, w2bar=1.0):
        self.w1bar = w1bar
        self.w2bar = w2bar

    # b. We plot the Edgeworth box
    def plot_edgeworth_box(self, xA1_optimal, xA2_optimal, xB1_optimal, xB2_optimal):
        fig = plt.figure(frameon=False, figsize=(6, 6), dpi=100)
        ax_A = fig.add_subplot(1, 1, 1)
        ax_A.set_xlabel("$x_1^A$") # Sets the label of the first x-axis
        ax_A.set_ylabel("$x_2^A$") # Sets the label of the first y-axis
        
        temp = ax_A.twinx()
        temp.set_ylabel("$x_2^B$") # Sets the label of the second y-axis
        
        ax_B = temp.twiny()
        ax_B.set_xlabel("$x_1^B$") # Sets the label of the second x-axis
        
        # i. We invert the second axes
        ax_B.invert_xaxis()
        ax_B.invert_yaxis()
        
        # ii. We create the scatter plots
        ax_A.scatter(xA1_optimal, xA2_optimal, marker='s', color='black', label='endowment')
        ax_A.scatter(xB1_optimal, xB2_optimal, marker='s', color='black')
        
        # iii. We plot the scatter plots
        ax_A.plot([0, self.w1bar], [0, 0], lw=2, color='black')
        ax_A.plot([0, self.w1bar], [self.w2bar, self.w2bar], lw=2, color='black')
        ax_A.plot([0, 0], [0, self.w2bar], lw=2, color='black')
        ax_A.plot([self.w1bar, self.w1bar], [0, self.w2bar], lw=2, color='black')
        
        # iv. We set the limits of the axes
        ax_A.set_xlim([-0.1, self.w1bar + 0.1])
        ax_A.set_ylim([-0.1, self.w2bar + 0.1])
        ax_B.set_xlim([self.w1bar + 0.1, -0.1])
        ax_B.set_ylim([self.w2bar + 0.1, -0.1])
        
        ax_A.legend(frameon=True, loc='upper right', bbox_to_anchor=(1.6, 1.0))
        plt.show() # Shows the figure


#Question 7
class TotalEndowments:
    # a. We generate the seed and number of elements
    def __init__(self, seed=69, n=50):
        np.random.seed(seed)
        self.n = n

    # b. We plot the figure
    def plot_endowments(self, wA1, wA2):
        result = self.pareto_improvements()

        # Plot the Edgeworth box with the endowments
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlabel("$x_1^A$") # setting x-axis label
        ax.set_ylabel("$x_2^A$") # setting y-axis label
        # Setting the limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Plotting endowment points
        ax.scatter(self.w_A1, self.w_A2, marker='s', color='black', label='Endowment A')
        ax.scatter(self.w_B1, self.w_B2, marker='s', color='red', label='Endowment B')

        # Plotting Pareto improvements
        for allocation in result:
            ax.scatter(allocation[0], allocation[1], color='green') 

        ax.legend() # adding legend
        plt.show() # display the plot



#Question 8
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