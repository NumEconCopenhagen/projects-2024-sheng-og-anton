# 1. Import relevant packages
# a. We use matplotlib.pyplot to show nice graphs 
import matplotlib.pyplot as plt
# b. 
import numpy as np
# c. 
from types import SimpleNamespace
# d.
import scipy
# e. 
from scipy import optimize 
# f.
from scipy.optimize import minimize

# 2. Defines a class that is used to answer most of the questions
class EdgeworthBoxClass:
    # a. Create an instance of the class with the given values of parameters and endowments
    def __init__(self, alpha, beta, endowment_A, num_pairs=50):
        
        self.alpha = alpha
        #
        self.beta = beta

        #
        self.endowment_A = endowment_A
        #
        self.endowment_B = [1 - e for e in endowment_A]

        # v. Set the number of allocations
        self.N = 75
        
        self.num_pairs = num_pairs

        self.pairs = None

        # vi. Define an instance of the number of pairs of endowments for consumer A
        self.num_pairs = num_pairs

    # b. Define the utility function for consumer A
    def u_A(self, x_A1, x_A2):
        # i. Returns the value of the utility function for consumer A
        return x_A1**self.alpha * x_A2**(1 - self.alpha)

    # c. Define the utility function for consumer B
    def u_B(self, x_B1, x_B2):
        # i. Returns the value of the utility function for consumer B
        return x_B1**self.beta * x_B2**(1 - self.beta)

    # d. Define the demand function for good 1 for consumer A 
    def demand_A_x1(self, p1, p2):
        # i. Returns the value of the demand function
        return self.alpha * (p1*self.endowment_A[0] + p2*self.endowment_A[1]) / p1

    # e. Define the demand function for good 2 for consumer A
    def demand_A_x2(self, p1, p2):
        # i. Returns the value of the demand function
        return (1 - self.alpha) * (p1*self.endowment_A[0] + p2*self.endowment_A[1]) / p2

    # f. Define the demand function for good 1 for consumer B
    def demand_B_x1(self, p1, p2):
        # i. Returns the value of the demand function
        return self.beta * (p1*self.endowment_B[0] + p2*self.endowment_B[1]) / p1

    # g. Define the demand function for good 2 for consumer B
    def demand_B_x2(self, p1, p2):
        # i. Returns the value of the demand function
        return (1 - self.beta) * (p1*self.endowment_B[0] + p2*self.endowment_B[1]) / p2

    # h. Define the function that finds the pareto improvements
    def pareto_improvements(self):
        # i. Create an empty list that will store the pareto improvements
        pareto_improvements = []
        # ii. Using a nested for loop to find and define x_A1 and x_A2
        for i in range(self.N + 1):
            # I.
            x_A1 = i / self.N
            # II.
            for j in range(self.N + 1):
                # II.a
                x_A2 = j / self.N

                # II.b Calculate x_B1 based on x_A1
                x_B1 = 1 - x_A1
                
                # II.c Calculate x_B2 based on x_A2
                x_B2 = 1 - x_A2

                # II.d Checking the Pareto improvement relative to the endowment
                if self.u_A(x_A1, x_A2) >= self.u_A(self.endowment_A[0], self.endowment_A[1]) and \
                        self.u_B(x_B1, x_B2) >= self.u_B(self.endowment_B[0], self.endowment_B[1]) and \
                        x_B1 == 1 - x_A1 and x_B2 == 1 - x_A2:
                    
                    # II.d.1 Storing combination of x_A1 and x_A2.
                    pareto_improvements.append((x_A1, x_A2))

        # iii. Return the list of pareto improvements
        return pareto_improvements
    
    # i. Define a function that plots the Edgeworth Box
    
    def plot_edgeworth_box(self):
        result = self.pareto_improvements()
        result = np.array(result)

        fig, ax = plt.subplots(figsize=(8, 8))

        #
        ax.set_xlabel("$x_1^A$")  # setting x-axis label
        
        #
        ax.set_ylabel("$x_2^A$")  # setting y-axis label
        
        # Setting the limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.scatter(self.endowment_A[0], self.endowment_A[1], marker='s', color='black', label='Endowment')

        ax.scatter(result[:, 0], result[:, 1], color='green', label='Pareto Improvements')

        ax.legend()
        
        plt.title('Market Equilibrium Allocations in the Edgeworth Box')
        plt.show()

    # j. Find the market clearing price for question 3
    def market_clearing_price(self):
        # i. Define a functions that calculates the excess demand 
        def excess_demand_x1(p1):
            # i.a. Set aggregate demand for good 1 equal to demand for consumer A plus the demand for consumer B 
            aggregate_demand_x1 = self.demand_A_x1(p1, 1) + self.demand_B_x1(p1, 1)
            
            # i.b. Set total endowment for good 1 equal to endowment for consumer A plus endowment for consumer B
            total_endowment_x1 = self.endowment_A[0] + self.endowment_B[0]
            
            # 1.c. Return the excess demand for good 1 defined by the difference between the aggregate demand and total endowment
            return aggregate_demand_x1 - total_endowment_x1

        # ii. Calculate the market clearing price by minimizing the excess demand for good 1, since there is no excess demand at this price
        p1_clearing = optimize.brentq(excess_demand_x1, 0.01, 10)
        
        # iii. Return the market clearing price
        return p1_clearing
    
    # k. Define a function that calculates the optimal allocation under a utilitarian social planner in question 6.a
    def SocialPlanner(self): 
        # i. Define a function that is the objective function for the utilitarian social planner
        def utilitarian_objective_function(x):
            # ia. Return the utility based on the total utility
            return-(self.u_A(x[0],x[1])+self.u_B(1-x[0],1-x[1]))
        
        # ii. Define the bounds for the allocations of the goods
        bounds = ((0,1),(0,1))
        
        # iii. Make an initial guess, which should be feasible under the bounds
        initial_guess = [0.8,0.3]

        # iv. Calculate the optimal allocation for consumer A under the utilitarian social planner
        solution_to_question_6a = scipy.optimize.minimize(
            utilitarian_objective_function, initial_guess, method='SLSQP',
            bounds = bounds)
        
        # v. 
        utility_for_A = self.u_A(solution_to_question_6a.x[0],solution_to_question_6a.x[1])
        
        # vi.
        utility_for_B = self.u_B(1-solution_to_question_6a.x[0],1-solution_to_question_6a.x[1])
        
        # vii. 
        aggregate_utility = utility_for_A + utility_for_B

        # viii. Print statement for the optimal allocation for consumer A with a utilitarian social planner
        print(f'The utilitarian social planner chooses the allocation for consumer A: (x1A, x2A) = ({solution_to_question_6a.x[0]:.2f},{solution_to_question_6a.x[1]:.2f})')
        # ix. Print statement for the utility for consumer A with the new optimal allocation
        print(f'Utility of consumer A at this allocation is {utility_for_A:.2f}, and the utility for consumer B is {utility_for_B:.2f}')
        # x. Print statement for the aggregated utility for both consumers
        print(f'The aggregated utility becomes {aggregate_utility:.2f}')

        # xi. Return the optimal allocation for consumer A
        return solution_to_question_6a.x[0], solution_to_question_6a.x[1]
    
    # l. Define a function that generates the random endowments
    def generate_random_endowments(self):
        # i. Calculate pairs of endowments based on the uniform
        self.pairs = np.random.uniform(0, 1, (self.num_pairs, 2))
    
    # m. Define a function that prints the random endowments
    def print_random_endowments(self):
        # i. If-statement for if there have been generated endowments
        if self.pairs is None:
            # I. Print statement when there have not been generated any random endowments
            print("The random endowments have not been generated yet. Call generate_pairs() first to do so.")
        # ii. If there are random endowments generated then this else statement will run
        else:
            # I. Do a for-loop for every endowment of good 1 and good 2 in the list of pairs
            for i, (omega1, omega2) in enumerate(self.pairs):
                # I.a Print statement for the value of the random endowments
                print(f"Element {i + 1}: ({omega1:.2f}, {omega2:.2f})")
    
    # n. Define a function that plots the random endowments
    def plot_random_endowments(self):
        # i. If-statement for if there have been generated endowments
        if self.pairs is None:
            # I. Print statement when there have not been generated any random endowments 
            print("The random endowments have not been generated yet. Call generate_pairs() first to do so.")
        # ii. If there are random endowments generated then this else statement will run
        else:
            # I. Make a scatterplot
            plt.scatter(self.pairs[:, 0], self.pairs[:, 1])
            
            # II. Make a title
            plt.title("Scatter Plot of Random Pairs (ωA1, ωA2)")
            
            # III. Label the x-axis
            plt.xlabel("ωA1")
            
            # IV. Label the y-axis
            plt.ylabel("ωA2")
            
            # V. Show the plot
            plt.show()

# 3. 
class ErrorMarketClass:
    # a. 
    def __init__(self):
        # i.
        par = self.par = SimpleNamespace()
        
        # ii. preferences
        par.alpha = 1/3
        # iii. 
        par.beta = 2/3
        
        # iv. Endowments
        par.w1A = 0.8
        # v. 
        par.w2A = 0.3
        # vi. 
        par.p2 = 1
        
        # vii.
        self.N = 75

    # b. 
    def utility_A(self, x1A, x2A):
        #
        return x1A ** self.par.alpha * x2A ** (1 - self.par.alpha)

    # c.
    def utility_B(self, x1B, x2B):
        #
        return x1B ** self.par.beta * x2B ** (1 - self.par.beta)

    # d.
    def demand_A(self, p1):
        #
        x1A_star = self.par.alpha * (p1 * self.par.w1A + self.par.p2 * self.par.w2A) / p1
        #
        x2A_star = (1 - self.par.alpha) * (p1 * self.par.w1A + self.par.p2 * self.par.w2A) / self.par.p2
        #
        return x1A_star, x2A_star

    # e.
    def demand_B(self, p1):
        #
        x1B_star = self.par.beta * (p1 * (1 - self.par.w1A) + self.par.p2 * (1 - self.par.w2A)) / p1
        #
        x2B_star = (1 - self.par.beta) * (p1 * (1 - self.par.w1A) + self.par.p2 * (1 - self.par.w2A)) / self.par.p2
        #
        return x1B_star, x2B_star

    # f.
    def check_market_clearing(self):
        # i.
        par = self.par
        # ii.
        self.rho1 = [0.5 + 2 * i / self.N for i in range(self.N + 1)]

        # iii.
        errors = []

        # iv.
        for p1 in self.rho1:
            # iva.
            x1A_star, x2A_star = self.demand_A(p1)
            # ivb.
            x1B_star, x2B_star = self.demand_B(p1)
            # ivc.
            eps1 = round(x1A_star - par.w1A + x1B_star - (1 - par.w1A),2)
            # ivd.
            eps2 = round(x2A_star - par.w2A + x2B_star - (1 - par.w2A),2)
            # ive.
            errors.append((eps1, eps2))

        # v. Return the errors for the market clearing conditions
        return errors

# 4. Define a class that plots the different optimal allocations in the figure in question 6.B 
class PointPlotter:
    # a. Define a function
    def __init__(self, points, labels_for_points, colors, w1bar=1.0, w2bar=1.0):
        # i. 
        self.points = points
        
        # ii. 
        self.labels_for_points = labels_for_points
        
        # iii. 
        self.colors = colors
        
        # iv. 
        self.w1bar = w1bar
        
        # v.
        self.w2bar = w2bar

    # b. Define a function that plots the previous optimal allocations from the questions
    def plot_the_previous_results(self):
        # i. Set up of the figure
        fig = plt.figure(frameon=False, figsize=(6, 6), dpi=100)
        
        # ii. Adds secondary axes
        ax_A = fig.add_subplot(1, 1, 1)

        # iii. Set the label on the x-axis
        ax_A.set_xlabel("$x_1^A$")
        
        # iv. Set the label on the primary y-axis
        ax_A.set_ylabel("$x_2^A$")
        # v. 
        temp = ax_A.twinx()
        # vi. Set the label for the secondary y-axis
        temp.set_ylabel("$x_2^B$")
        # vii. 
        ax_B = temp.twiny()
        # ix. Set the label on the secondary x-axis
        ax_B.set_xlabel("$x_1^B$")
        # x. Invert the secondary x-axis
        ax_B.invert_xaxis()
        # xi. Invert the secondary y-axis
        ax_B.invert_yaxis()

        # Plot the limits
        ax_A.plot([0, self.w1bar], [0, 0], lw=2, color='black')
        #
        ax_A.plot([0, self.w1bar], [self.w2bar, self.w2bar], lw=2, color='black')
        #
        ax_A.plot([0, 0], [0, self.w2bar], lw=2, color='black')
        #
        ax_A.plot([self.w1bar, self.w1bar], [0, self.w2bar], lw=2, color='black')

        # Set the limits for the primary x-axis
        ax_A.set_xlim([-0.1, self.w1bar + 0.1])
        # Set the limits for the primary y-axis
        ax_A.set_ylim([-0.1, self.w2bar + 0.1])
        # Set the limits for the secondary x-axis
        ax_B.set_xlim([self.w1bar + 0.1, -0.1])
        # Set the limits for the secondary y-axis
        ax_B.set_ylim([self.w2bar + 0.1, -0.1])

        # Plot the points with the colors and the labels
        for point, label, color in zip(self.points, self.labels_for_points, self.colors):
            # Make a scatterplot
            ax_A.scatter(*point, color=color, label=label)

        # Create a legend
        ax_A.legend(frameon=True, loc='lower right', bbox_to_anchor=(1.4, 0.6))

        # Show the plot
        plt.show()

class RandomParetoImprovements:
    def __init__(self, seed=69, alpha=1/3, beta=2/3, num_points=50):
        self.alpha = alpha
        self.beta = beta
        self.num_points = num_points
        np.random.seed(seed)  # For reproducibility
        self.omega_A = np.random.uniform(0, 1, (num_points, 2))
        self.omega_B = 1 - self.omega_A  # Total endowment of each good is always 1
    
    def u_A(self, x1, x2):
        return x1**self.alpha * x2**(1-self.alpha)

    def u_B(self, x1, x2):
        return x1**self.beta * x2**(1-self.beta)

    def market_equilibrium(self, omega):
        def objective(p):
            # Inverse demand functions derived from utility maximization
            xA1_star = self.alpha * (omega[0] + p * omega[1]) / p
            xB1_star = self.beta * ((1 - omega[0]) + p * (1 - omega[1])) / p
            # Market clearing condition error for good 1
            error = np.abs(xA1_star + xB1_star - 1)
            return error
        
        # Find p1 that minimizes the market clearing condition error
        res = minimize(objective, 0.5, bounds=[(0.01, 5)])
        p1_star = res.x[0]
        
        # Calculate equilibrium allocations using p1_star
        xA1_star = self.alpha * (omega[0] + p1_star * omega[1]) / p1_star
        xA2_star = (1 - self.alpha) * (omega[0] + p1_star * omega[1])
        xB1_star = self.beta * ((1 - omega[0]) + p1_star * (1 - omega[1])) / p1_star
        xB2_star = (1 - self.beta) * ((1 - omega[0]) + p1_star * (1 - omega[1]))
        
        return xA1_star, xA2_star, xB1_star, xB2_star

    def plot_of_the_random_pareto_improvements(self):
        # Create the figure and the main axis
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot in the Edgeworth box
        for omega in self.omega_A:
            xA1_star, xA2_star, xB1_star, xB2_star = self.market_equilibrium(omega)
            ax.plot(xA1_star, xA2_star, 'bo')  # Plot A's allocation

        # Set labels and grid for the main axis
        ax.set_xlabel('$x_1^A$')
        ax.set_ylabel('$x_2^A$')
        ax.grid(True)

        # Create twin axes sharing the y-axis
        ax_right = ax.twinx()
        ax_right.set_ylabel('$x_2^B$')
        ax_right.set_ylim(1, 0)  # Invert the y-axis for B's perspective

        # Create twin axes sharing the x-axis
        ax_top = ax.twiny()
        ax_top.set_xlabel('$x_1^B$')
        ax_top.set_xlim(1, 0)  # Invert the x-axis for B's perspective

        # Set limits for all axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Add title and show the plot
        plt.title('Market Equilibrium Allocations in the Edgeworth Box')
        plt.show()

