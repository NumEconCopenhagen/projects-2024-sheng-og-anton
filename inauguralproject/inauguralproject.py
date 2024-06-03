#
import matplotlib.pyplot as plt
#
import numpy as np
#
from types import SimpleNamespace
#
import scipy
#
from scipy import optimize 
#
from scipy.optimize import minimize

#
class EdgeworthBoxClass:
    #
    def __init__(self, alpha, beta, endowment_A, num_pairs=50):
        #
        self.alpha = alpha
        #
        self.beta = beta

        #
        self.endowment_A = endowment_A
        #
        self.endowment_B = [1 - e for e in endowment_A]

        # Number of allocations
        self.N = 75

        #
        self.num_pairs = num_pairs
        #
        self.pairs = None

    #
    def u_A(self, x_A1, x_A2):
        # Define the utility function for consumer A
        return x_A1**self.alpha * x_A2**(1 - self.alpha)

    #
    def u_B(self, x_B1, x_B2):
        # Define the utility function for consumer B
        return x_B1**self.beta * x_B2**(1 - self.beta)

    #
    def demand_A_x1(self, p1, p2):
        #
        return self.alpha * (p1*self.endowment_A[0] + p2*self.endowment_A[1]) / p1

    #
    def demand_A_x2(self, p1, p2):
        #
        return (1 - self.alpha) * (p1*self.endowment_A[0] + p2*self.endowment_A[1]) / p2

    #
    def demand_B_x1(self, p1, p2):
        #
        return self.beta * (p1*self.endowment_B[0] + p2*self.endowment_B[1]) / p1

    #
    def demand_B_x2(self, p1, p2):
        #
        return (1 - self.beta) * (p1*self.endowment_B[0] + p2*self.endowment_B[1]) / p2

    #
    def pareto_improvements(self):
        #
        pareto_improvements = []
        # Using a nested for loop to find and define x_A1 and x_A2
        for i in range(self.N + 1):
            #
            x_A1 = i / self.N
            #
            for j in range(self.N + 1):
                #
                x_A2 = j / self.N

                # Calculate x_B1 and x_B2 based on x_A1 and x_A2
                x_B1 = 1 - x_A1
                
                #
                x_B2 = 1 - x_A2

                # Checking the Pareto improvement relative to the endowment
                if self.u_A(x_A1, x_A2) >= self.u_A(self.endowment_A[0], self.endowment_A[1]) and \
                        self.u_B(x_B1, x_B2) >= self.u_B(self.endowment_B[0], self.endowment_B[1]) and \
                        x_B1 == 1 - x_A1 and x_B2 == 1 - x_A2:
                    
                    # Storing combination of x_A1 and x_A2.
                    pareto_improvements.append((x_A1, x_A2))

        #
        return pareto_improvements
    
    #
    def plot_edgeworth_box(self):
        #
        result = self.pareto_improvements()

        #
        result = np.array(result)

        # Plot the Edgeworth box with Pareto improvements
        fig, ax = plt.subplots(figsize=(8, 8))

        #
        ax.set_xlabel("$x_1^A$")  # setting x-axis label
       
        #
        ax.set_ylabel("$x_2^A$")  # setting y-axis label
        temp = ax.twinx()
        temp.set_ylabel("$x_2^B$")
        ax = temp.twiny()
        ax.set_xlabel("$x_1^B$")
        ax.invert_xaxis()
        ax.invert_yaxis()
        
        # Setting the limits
        ax.set_xlim(0, 1)
        
        #
        ax.set_ylim(0, 1)

        # Plotting endowment points
        ax.scatter(self.endowment_A[0], self.endowment_A[1], marker='s', color='black', label='Endowment')
    
        # Plotting Pareto improvements
        ax.scatter(result[:, 0], result[:, 1], color='green', label='Pareto Improvements') 

        #
        ax.legend()  # adding legend
        
        #
        plt.show()  # display the plot

    # Market clearing price
    def market_clearing_price(self):
        #
        def excess_demand_x1(p1):
            #
            aggregate_demand_x1 = self.demand_A_x1(p1, 1) + self.demand_B_x1(p1, 1)
            
            #
            total_endowment_x1 = self.endowment_A[0] + self.endowment_B[0]
            
            #
            return aggregate_demand_x1 - total_endowment_x1

        #
        p1_clearing = optimize.brentq(excess_demand_x1, 0.01, 10)
        
        return p1_clearing
    
    #
    def SocialPlanner(self):
        #
        def utilitarian_objective_function(x):
            #
            return-(self.u_A(x[0],x[1])+self.u_B(1-x[0],1-x[1]))
        
        #
        bounds = ((0,1),(0,1))
        
        #
        initial_guess = [0.8,0.3]

        #
        solution_to_question_6a = scipy.optimize.minimize(
            utilitarian_objective_function, initial_guess, method='SLSQP',
            bounds = bounds)
        
        #
        utility_for_A = self.u_A(solution_to_question_6a.x[0],solution_to_question_6a.x[1])
        
        #
        utility_for_B = self.u_B(1-solution_to_question_6a.x[0],1-solution_to_question_6a.x[1])
        
        #
        aggregate_utility = utility_for_A + utility_for_B

        #
        print(f'The utilitarian social planner chooses the allocation for consumer A: (x1A, x2A) = ({solution_to_question_6a.x[0]:.2f},{solution_to_question_6a.x[1]:.2f})')
        #
        print(f'Utility of consumer A at this allocation is {utility_for_A:.2f}, and the utility for consumer B is {utility_for_B:.2f}')
        #
        print(f'The aggregated utility becomes {aggregate_utility:.2f}')

        #
        return solution_to_question_6a.x[0], solution_to_question_6a.x[1]
    
    #
    def generate_random_endowments(self):
        #
        self.pairs = np.random.uniform(0, 1, (self.num_pairs, 2))
    
    #
    def print_random_endowments(self):
        #
        if self.pairs is None:
            #
            print("Pairs not generated yet. Call generate_pairs() first.")
        #
        else:
            #
            for i, (omega1, omega2) in enumerate(self.pairs):
                #
                print(f"Element {i + 1}: ({omega1:.2f}, {omega2:.2f})")
    
    #
    def plot_random_endowments(self):
        #
        if self.pairs is None:
            #
            print("Pairs not generated yet. Call generate_pairs() first.")
        #
        else:
            #
            plt.scatter(self.pairs[:, 0], self.pairs[:, 1])
            #
            plt.title("Scatter Plot of Random Pairs (ωA1, ωA2)")
            #
            plt.xlabel("ωA1")
            #
            plt.ylabel("ωA2")
            #
            plt.show()

#
class PointPlotter:
    #
    def __init__(self, points, labels_for_points, colors, w1bar=1.0, w2bar=1.0):
        #
        self.points = points
        
        #
        self.labels_for_points = labels_for_points
        
        #
        self.colors = colors
        
        #
        self.w1bar = w1bar
        
        #
        self.w2bar = w2bar

    #
    def plot_the_previous_results(self):
        # Figure set up
        fig = plt.figure(frameon=False, figsize=(6, 6), dpi=100)
        
        #
        ax_A = fig.add_subplot(1, 1, 1)

        #
        ax_A.set_xlabel("$x_1^A$")
        
        #
        ax_A.set_ylabel("$x_2^A$")
        #
        temp = ax_A.twinx()
        #
        temp.set_ylabel("$x_2^B$")
        #
        ax_B = temp.twiny()
        #
        ax_B.set_xlabel("$x_1^B$")
        #
        ax_B.invert_xaxis()
        #
        ax_B.invert_yaxis()

        # Plot the limits
        ax_A.plot([0, self.w1bar], [0, 0], lw=2, color='black')
        #
        ax_A.plot([0, self.w1bar], [self.w2bar, self.w2bar], lw=2, color='black')
        #
        ax_A.plot([0, 0], [0, self.w2bar], lw=2, color='black')
        #
        ax_A.plot([self.w1bar, self.w1bar], [0, self.w2bar], lw=2, color='black')

        #
        ax_A.set_xlim([-0.1, self.w1bar + 0.1])
        #
        ax_A.set_ylim([-0.1, self.w2bar + 0.1])
        #
        ax_B.set_xlim([self.w1bar + 0.1, -0.1])
        #
        ax_B.set_ylim([self.w2bar + 0.1, -0.1])

        # Plot the points with colors and labels
        for point, label, color in zip(self.points, self.labels_for_points, self.colors):
            #
            ax_A.scatter(*point, color=color, label=label)

        #
        ax_A.legend(frameon=True, loc='lower right', bbox_to_anchor=(1.4, 0.6))

        #
        plt.show()

#Question 2 
class ErrorMarketClass:
    #
    def __init__(self):
        #
        par = self.par = SimpleNamespace()
        
        # a. preferences
        par.alpha = 1/3
        #
        par.beta = 2/3
        
        # b. Endowments
        par.w1A = 0.8
        #
        par.w2A = 0.3
        #
        par.p2 = 1
        
        #
        self.N = 75

    #
    def utility_A(self, x1A, x2A):
        #
        return x1A ** self.par.alpha * x2A ** (1 - self.par.alpha)

    #
    def utility_B(self, x1B, x2B):
        #
        return x1B ** self.par.beta * x2B ** (1 - self.par.beta)

    #
    def demand_A(self, p1):
        #
        x1A_star = self.par.alpha * (p1 * self.par.w1A + self.par.p2 * self.par.w2A) / p1
        #
        x2A_star = (1 - self.par.alpha) * (p1 * self.par.w1A + self.par.p2 * self.par.w2A) / self.par.p2
        #
        return x1A_star, x2A_star

    #
    def demand_B(self, p1):
        #
        x1B_star = self.par.beta * (p1 * (1 - self.par.w1A) + self.par.p2 * (1 - self.par.w2A)) / p1
        #
        x2B_star = (1 - self.par.beta) * (p1 * (1 - self.par.w1A) + self.par.p2 * (1 - self.par.w2A)) / self.par.p2
        #
        return x1B_star, x2B_star

    #
    def check_market_clearing(self):
        #
        par = self.par
        #
        self.rho1 = [0.5 + 2 * i / self.N for i in range(self.N + 1)]

        #
        errors = []

        #
        for p1 in self.rho1:
            #
            x1A_star, x2A_star = self.demand_A(p1)
            #
            x1B_star, x2B_star = self.demand_B(p1)
            #
            eps1 = round(x1A_star - par.w1A + x1B_star - (1 - par.w1A),2)
            #
            eps2 = round(x2A_star - par.w2A + x2B_star - (1 - par.w2A),2)
            #
            errors.append((eps1, eps2))

        #
        return errors
        
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
