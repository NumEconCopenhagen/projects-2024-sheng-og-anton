# 1. Import relevant packages 
import matplotlib.pyplot as plt 
import numpy as np
from types import SimpleNamespace
import scipy 
from scipy import optimize 
from scipy.optimize import minimize

# 2. Defines a class that is used to answer most of the questions
class EdgeworthBoxClass:
    """
    Class used to analyze the different optimal allocations for consumer A in the Edgeworth Box

    Attributes:
        alpha (float): Preference parameter for consumer A.
        beta (beta): Preference parameter for consumer B.
        endowment_A (list): The initial endowment for consumer A.
        endowment_B (list): The initial endowment for consumer B.
        N (int): Number of allocations used for plotting in question 7.
        num_pairs (int): The number of pairs of random endowments which should be generated in question 7.
        pairs (list): The list of the pairs of random endowments.
    """
    # a. Create an instance of the class with the given values of parameters and endowments
    def __init__(self, alpha, beta, endowment_A, num_pairs=50):
        """
        Initialize the EdgeworthBoxClass with preferences, endowments, and the number of pairs.

        Args:
            alpha (float): Preference parameter for consumer A.
            beta (float): Preference parameter for consumer B.
            endowment_A (list): Initial endowments for consumer A.
            num_pairs (int): Number of pairs of random endowments to be generated in question 7.
        """
        self.alpha = alpha
        
        self.beta = beta

        self.endowment_A = endowment_A
        
        self.endowment_B = [1 - e for e in endowment_A]

        # Set the number of allocations
        self.N = 75
        
        self.num_pairs = num_pairs

        self.pairs = None

        # Define an instance of the number of pairs of endowments for consumer A
        self.num_pairs = num_pairs

    # b. Define the utility function for consumer A
    def u_A(self, x_A1, x_A2):
        """
        Calculates the utility for consumer A based on her utility function.

        Args: 
            x_A1 (float): The amount of good 1 which is consumed by consumer A.
            x_A2 (float): The amount of good 2 which is consumed by consumer A.

        Returns:
            (float): The utility of consumer A.
        """
        # i. Returns the value of the utility function for consumer A
        return x_A1**self.alpha * x_A2**(1 - self.alpha)

    # c. Define the utility function for consumer B
    def u_B(self, x_B1, x_B2):
        """
        Calculate the utility for consumer B.

        Args: 
            x_B1 (float): The amount of good 1, which is consumed by consumer B.
            x_B2 (float): The amount of good 2, which is consumed by consumer B.

        Returns:
            (float): The utility of consumer B.

        """
        # i. Returns the value of the utility function for consumer B
        return x_B1**self.beta * x_B2**(1 - self.beta)

    # d. Define the demand function for good 1 for consumer A 
    def demand_A_x1(self, p1, p2):
        """
        Calculates the demand for good 1 by consumer A.

        Args:
            p1 (float): Price of good 1.
            p2 (float): Price of good 2.

        Returns: 
            (float): Demand for good 1 by consumer A.
        """
        # i. Returns the value of the demand function
        return self.alpha * (p1*self.endowment_A[0] + p2*self.endowment_A[1]) / p1

    # e. Define the demand function for good 2 for consumer A
    def demand_A_x2(self, p1, p2):
        """
        Calculates the demand for good 2 by consumer A.

        Args: 
            p1 (float): Price of good 1.
            p2 (float): Price of good 2, which is set to 1, since it is numeraire.

        Returns: 
            (float): Demand for good 1 by consumer A.
        """
        # i. Returns the value of the demand function
        return (1 - self.alpha) * (p1*self.endowment_A[0] + p2*self.endowment_A[1]) / p2

    # f. Define the demand function for good 1 for consumer B
    def demand_B_x1(self, p1, p2):
        """
        Calculate the demand for good 1 by consumer B.

        Args:
            p1 (float): Price of good 1.
            p2 (float): Price of good 2.

        Returns:
            (float): Demand for good 1 by consumer B.            
        """
        # i. Returns the value of the demand function
        return self.beta * (p1*self.endowment_B[0] + p2*self.endowment_B[1]) / p1

    # g. Define the demand function for good 2 for consumer B
    def demand_B_x2(self, p1, p2):
        """
        Calculate the demand for good 2 by consumer B.

        Args: 
            p1 (float): Price of good 1.
            p2 (float): Price of good 2.
        
        Returns: 
            (float): Demand for good 2 by consumer B.
        """
        # i. Returns the value of the demand function
        return (1 - self.beta) * (p1*self.endowment_B[0] + p2*self.endowment_B[1]) / p2

    # h. Define the function that finds the pareto improvements
    def pareto_improvements(self):
        """
        Finds the Pareto improvements from the initial endowments for the two consumers.

        Returns:
            (list): A list of tuples.
                (tuple): The tuples contain the Pareto improvements. 
        """
        # i. Create an empty list that will store the pareto improvements
        pareto_improvements = []
        # ii. Using a nested for loop to find and define x_A1 and x_A2
        for i in range(self.N + 1):
            
            x_A1 = i / self.N
            
            for j in range(self.N + 1):
                
                x_A2 = j / self.N
               
                x_B1 = 1 - x_A1
                
                x_B2 = 1 - x_A2

                if self.u_A(x_A1, x_A2) >= self.u_A(self.endowment_A[0], self.endowment_A[1]) and \
                        self.u_B(x_B1, x_B2) >= self.u_B(self.endowment_B[0], self.endowment_B[1]) and \
                        x_B1 == 1 - x_A1 and x_B2 == 1 - x_A2:
                    
                    # Storing combination of x_A1 and x_A2
                    pareto_improvements.append((x_A1, x_A2))

        # iii. Return the list of pareto improvements
        return pareto_improvements
    
    # i. Define a function that plots the Edgeworth Box
    def plot_edgeworth_box(self):
        """
        Plots the Edgeworth Box with Pareto imrovements and initial endowments for the two consumers.
        """
        result = self.pareto_improvements()
        result = np.array(result)

        fig, ax = plt.subplots(figsize=(8, 8))

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

        ax.scatter(self.endowment_A[0], self.endowment_A[1], marker='s', color='black', label='Endowment')

        ax.scatter(result[:, 0], result[:, 1], color='green', label='Pareto Improvements')

        ax.legend()
        
        plt.title('Market Equilibrium Allocations in the Edgeworth Box')
        plt.show()

    # j. Find the market clearing price for question 3
    def market_clearing_price(self):
        """
        Finds the market clearing price for good 1 for the two consumers.

        Returns:
            (float): Market clearing price for good 1.
        """
        # i. Define a functions that calculates the excess demand 
        def excess_demand_x1(p1):
            """
            Calculates the excess demand for good 1.

            Args: 
                p1 (float): Price of good 1.

            Returns:
                (float): Excess demand for good 1.
            """
            # i.a. Set aggregate demand for good 1 equal to demand for consumer A plus the demand for consumer B 
            aggregate_demand_x1 = self.demand_A_x1(p1, 1) + self.demand_B_x1(p1, 1)
            
            # i.b. Set total endowment for good 1 equal to endowment for consumer A plus endowment for consumer B
            total_endowment_x1 = self.endowment_A[0] + self.endowment_B[0]
            
            # i.c. Return the excess demand for good 1 defined by the difference between the aggregate demand and total endowment
            return aggregate_demand_x1 - total_endowment_x1

        # ii. Calculate the market clearing price by minimizing the excess demand for good 1, since there is no excess demand at this price
        p1_clearing = optimize.brentq(excess_demand_x1, 0.01, 10)
        
        # iii. Return the market clearing price
        return p1_clearing
    
    # k. Define a function that calculates the optimal allocation under a utilitarian social planner in question 6.a
    def SocialPlanner(self): 
        """
        Calculates the optimal allocation for the two consumers under a utilitarian social planner which maximizes aggregate utility.

        Returns:
            (tuple): A tuple that contains the optimal allocation for consumer A.
        """
        # i. Define a function that is the objective function for the utilitarian social planner
        def utilitarian_objective_function(x):
            """
            Calculates the negative of the aggregate utility function for the social planner, which will be minimized.

            Args:
                x (list): Allocation of the two goods for consumer A.
                    x[0]: The allocation of good 1 for consumer A.
                    x[1]: The allocation of good 2 for consumer A.

            Returns:
                (float): The negative of the total utility for the consumers.
            """
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
        
        # v. Set the utility for consumer A equal to the utility after the maximization problem has been solved
        utility_for_A = self.u_A(solution_to_question_6a.x[0],solution_to_question_6a.x[1])
        
        # vi. Set the utility for consumer B equal to 1 minus the optimal allocation for consumer A 
        utility_for_B = self.u_B(1-solution_to_question_6a.x[0],1-solution_to_question_6a.x[1])
        
        # vii. Defines aggregate utility as utility for consumer A plus utility for consumer B
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
        """
        Generates the random endowments for both consumers.
        """
        # i. Calculate pairs of endowments based on the uniform
        self.pairs = np.random.uniform(0, 1, (self.num_pairs, 2))
    
    # m. Define a function that prints the random endowments
    def print_random_endowments(self):
        """
        Solves the model for each of the pairs of random endowments and finds the market clearing prices.
        """
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
        """
        Plots the generated random endowments.
        """
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

# 3. Define the ErrorMarketClass that calculates the errors in question 2
class ErrorMarketClass:
    """
    This class represents the exchange economy with two consumers, A and B.
    
    Attributes:
        par (SimpleNamespace): A namespace which contains the parameters of the exchange economy.
        N (int): The number of prices which should be evaluated for the market clearing conditions.
    """
    # a. Define parameters and allocations
    def __init__(self):
        
        """
        Initializes the market with the default parameters to solve question 2.
        """
        par = self.par = SimpleNamespace()
        par.alpha = 1/3
        par.beta = 2/3
        par.w1A = 0.8
        par.w2A = 0.3
        par.p2 = 1
        self.N = 75

    # b. Define the utility function for consumer A
    def utility_A(self, x1A, x2A):
        """
        Calculates the utility for consumer A.

        Args: 
            x1A (float): The amount of good 1 which is consumed by consumer A.
            x2A (float): The amount of good 2 which is consumed by consumer A.

        Returns: 
            (float): The utility for consumer A.
        """
        return x1A ** self.par.alpha * x2A ** (1 - self.par.alpha)

    # c. Define the utility for consumer B
    def utility_B(self, x1B, x2B):
        """
        Calculates the utility for consumer B.

        Args: 
            x1B (float): Amount of good 1 which is consumed by consumer B.
            x2B (float): Amount of goof 2 which is consumed by consumer B.

        Returns:
            (float): Utility for consumer B.
        """
        return x1B ** self.par.beta * x2B ** (1 - self.par.beta)

    # d. Define the demand for consumer A
    def demand_A(self, p1):
        """
        Calculates the demand for the two goods for consumer A.

        Args: 
            p1 (float): The price of good 1.

        Returns:
            (tuple): A tuple which contains the optimal allocation of the two goods, where:
                x1A_opt (float): The optimal allocation of good 1 for consumer A.
                x2A_opt (float): The optimal allocation of good 2 for consumer A.
        """
        x1A_opt = self.par.alpha * (p1 * self.par.w1A + self.par.p2 * self.par.w2A) / p1
        x2A_opt = (1 - self.par.alpha) * (p1 * self.par.w1A + self.par.p2 * self.par.w2A) / self.par.p2

        return x1A_opt, x2A_opt

    # e. Define the demand for consumer B
    def demand_B(self, p1):
        """
        Calculates the demand for the two goods for consumer B.

        Args: 
            p1 (float): Price of good 1.
        
        Returns: 
            (tuple): A tuple which contains the optimal allocation of good 1 (x1B_star) and good 2 (x2B_star) for consumer B.
                x1B_opt (float): Optimal allocation of good 1 for consumer B.
                x2B_opt (float): Optimal allocation of good 2 for consumer B.
        """
        x1B_opt = self.par.beta * (p1 * (1 - self.par.w1A) + self.par.p2 * (1 - self.par.w2A)) / p1
        x2B_opt = (1 - self.par.beta) * (p1 * (1 - self.par.w1A) + self.par.p2 * (1 - self.par.w2A)) / self.par.p2
        return x1B_opt, x2B_opt

    # f. Define a method that checks the market clearing conditions
    def check_market_clearing(self):
        """
        Checks market clearing conditions across a range of prices. 
        
        Returns: 
            errors (list): A list which contains a tuple of the errors for the two market clearing conditions.
                (tuple): A tuple which contains the two errors (eps1 and eps2) for the market clearing conditions.
                    eps1 (float): Error in the market clearing condition for good 1.
                    eps2 (float): Error in the market clearing condition for good 2.
        """
        par = self.par
        self.rho1 = [0.5 + 2 * i / self.N for i in range(self.N + 1)]
        errors = []

        for p1 in self.rho1:
            x1A_opt, x2A_opt = self.demand_A(p1)
            x1B_opt, x2B_opt = self.demand_B(p1)
            eps1 = round(x1A_opt - par.w1A + x1B_opt - (1 - par.w1A),2)
            eps2 = round(x2A_opt - par.w2A + x2B_opt - (1 - par.w2A),2)
            errors.append((eps1, eps2))

        return errors

# 4. Define the PointPlotterClass that plots the different optimal allocations in the figure in Question 6.B 
class PointPlotterClass:
    """
    A class used for plotting the points of optimal allocations with labels and colors.
    
    Attributes:
        Points (list): List of tuples which contains the 
        labels_for_points (list): List of labels which corresponds to each of the points for optimal allocation.
        colors (list): List of colors which corresponds to each of the points for optimal allocation.
        w1bar (float): The upper limit for the x-axis, which is total feasible endowment of good 1.
        w2bar (float): The upper limit for the y-axis, which is total feasible endowment of good 2.
    """
    # a. Define a function that initializes the class
    def __init__(self, points, labels_for_points, colors, w1bar=1.0, w2bar=1.0):
        """
        Initializes the PointPlotter class with points, labels, colors, and axis limits (total feasible endowments)
        
        Args:
            points (list): List of tuples which contains the points that should be plotted.
            labels_for_points (list): List of labels which corresponds to each of the points for optimal allocation.
            colors (list): List of colors which corresponds to each of the points for optimal allocation. 
            w1bar (float): Upper limit for the x-axis, which is the total feasible endowment for good 1 for the consumers.
            w2bar (float): Upper limit for the y-axis, which is the total feasible endowment for good 2 for the consumers.
        """
        self.points = points
        self.labels_for_points = labels_for_points
        self.colors = colors
        self.w1bar = w1bar
        self.w2bar = w2bar

    # b. Define a function that plots the previous optimal allocations from the questions
    def plot_the_previous_results(self):
        """
        Plots the points for optimal allocation with labels and colors, and sets up the plot 
        """
        # i. Set up of the figure
        fig = plt.figure(frameon=False, figsize=(6, 6), dpi=100)
        
        # ii. Adds secondary axes
        ax_A = fig.add_subplot(1, 1, 1)

        # iii. Set the label on the x-axis
        ax_A.set_xlabel("$x_1^A$")
        
        # iv. Set the label on the primary y-axis
        ax_A.set_ylabel("$x_2^A$")
        # v. Create a secondary x-axis
        temp = ax_A.twinx()
        # vi. Set the label for the secondary y-axis
        temp.set_ylabel("$x_2^B$")
        # vii. Create a secondary x-axis
        ax_B = temp.twiny()
        # ix. Set the label on the secondary x-axis
        ax_B.set_xlabel("$x_1^B$")
        # x. Invert the secondary x-axis
        ax_B.invert_xaxis()
        # xi. Invert the secondary y-axis
        ax_B.invert_yaxis()

        # xii. Plot the limits
        ax_A.plot([0, self.w1bar], [0, 0], lw=2, color='black')
        ax_A.plot([0, self.w1bar], [self.w2bar, self.w2bar], lw=2, color='black')
        ax_A.plot([0, 0], [0, self.w2bar], lw=2, color='black')
        ax_A.plot([self.w1bar, self.w1bar], [0, self.w2bar], lw=2, color='black')

        # xiii. Set the limits for the primary x-axis
        ax_A.set_xlim([-0.1, self.w1bar + 0.1])
        # xiv. Set the limits for the primary y-axis
        ax_A.set_ylim([-0.1, self.w2bar + 0.1])
        # xv. Set the limits for the secondary x-axis
        ax_B.set_xlim([self.w1bar + 0.1, -0.1])
        # xvi. Set the limits for the secondary y-axis
        ax_B.set_ylim([self.w2bar + 0.1, -0.1])

        # xvii. Plot the points with the colors and the labels
        for point, label, color in zip(self.points, self.labels_for_points, self.colors):
            ax_A.scatter(*point, color=color, label=label)

        # xviii. Create a legend
        ax_A.legend(frameon=True, loc='lower right', bbox_to_anchor=(1.4, 0.6))

        # xix. Show the plot
        plt.show()

# 5. Defines a class that calculates and plots the random pareto improvements
class RandomParetoImprovementsClass:
    """
    A class used for generating random endowments and plotting the market equilibria in the Edgeworth box
    
    Attributes:
        alpha (float): Preference parameter in the utility function for consumer A.
        beta (float): Preference parameter in the utility function for consumer B.
        num_points (int): Number of random endowments which should be generated.
        omega_A (ndarray): NumPy array of random endowments for the goods for consumer A.
        omega_B (ndarray): NumPy array of random endowments for the goods for consumer B.
    """
    def __init__(self, seed=69, alpha=1/3, beta=2/3, num_points=50):
        """
        Initializes the RandomParetoImprovementsClass with parameters, seed, and number of points.

        Args: 
            seed (int): The seed value for the generation of the random endowments. 
            alpha (float): Preference parameter for the utility function for consumer A.
            beta (float): Preference parameter for the utility function for consumer B.
            num_points (int): The number of points.
        """
        self.alpha = alpha
        self.beta = beta
        self.num_points = num_points
        # i. We set a seed to reproduce and get the same pseduo-random endowments
        np.random.seed(seed)
        # ii. Calculate the endowments for consumer A using that they are uniform distributed
        self.omega_A = np.random.uniform(0, 1, (num_points, 2))
        # iii. The total endowment will also sum to one
        self.omega_B = 1 - self.omega_A
    
    def u_A(self, x1A, x2A):
        """
        Calculates the utility for consumer A.

        Args: 
            x1A (float): The amount of good 1, which is consumed by consumer A.
            x2A (float): The amount of good 2, which is consumed by consumer A.

        Returns:
            (float): The utility for consumer A.  
        """
        return x1A**self.alpha * x2A**(1-self.alpha)

    def u_B(self, x1B, x2B):
        """
        Calculates the utility for consumer B.

        Args:
            x1B (float): The amount of good 1, which is consumed by consumer B. 
            x2B (float): The amount of good 2, which is consumed by consumer B.

        Returns:
            (float): The utility for consumer B.
        """
        return x1B**self.beta * x2B**(1-self.beta)

    def market_equilibrium(self, omega):
        """
        Calculates optimal allocations for a given set of endowments.
        
        Args: 
            Omega (ndarray): NumPy array of the initial endowments for the consumers.
        
        Returns:
            (tuple): A tuple which contains the optimal allocation for the two consumers. 
                x1A_opt (float): The optimal allocation of good 1 for consumer A.
                x2A_opt (float): The optimal allocation of good 2 for consumer A.
                x1B_opt (float): The optimal allocation of good 1 for consumer B.
                x2B_opt (float): The optimal allocation of good 2 for consumer B.
        """
        # i. Define the objective function
        def objective_function(p):
            """
            Objective function to minimize the errors in the market clearing conditions.

            Args: 
                p (float): The price of good 1.

            Returns:
                (float): Errors in the market clearing condition for good 1.
            """
            # i. The inverse demand functions
            x1A_opt = self.alpha * (omega[0] + p * omega[1]) / p
            x1B_opt = self.beta * ((1 - omega[0]) + p * (1 - omega[1])) / p
            
            # ii. Market clearing condition error for good 1
            error = np.abs(x1A_opt + x1B_opt - 1)

            return error
        
        # ii. Find the optimal price for good 1 that minimizes the market clearing condition error
        res = minimize(objective_function, 0.5, bounds=[(0.01, 5)])
        p1_opt = res.x[0]
        
        # iii. Calculate the equilibrium allocations using the optimal price for good 1
        x1A_opt = self.alpha * (omega[0] + p1_opt * omega[1]) / p1_opt
        x2A_opt = (1 - self.alpha) * (omega[0] + p1_opt * omega[1])
        x1B_opt = self.beta * ((1 - omega[0]) + p1_opt * (1 - omega[1])) / p1_opt
        x2B_opt = (1 - self.beta) * ((1 - omega[0]) + p1_opt * (1 - omega[1]))
        
        return x1A_opt, x2A_opt, x1B_opt, x2B_opt

    def plot_of_the_random_pareto_improvements(self):
        """
        Plots the generated random endowments.
        """
        # i. Create the figure and the main axes
        fig, ax = plt.subplots(figsize=(8, 8))

        # ii. Plot in the Edgeworth box
        for omega in self.omega_A:
            x1A_opt, x2A_opt, x1B_opt, x2B_opt = self.market_equilibrium(omega)
            ax.plot(x1A_opt, x2A_opt, 'bo') # b (blue) is the color of the points, and o (dot) is the shape of the points 

        # iii. Set labels and grid for the axes for consumer A
        ax.set_xlabel('$x_1^A$')
        ax.set_ylabel('$x_2^A$')
        ax.grid(True)

        # iv. Create twin y-axis that is inverted for consumer B
        ax_right = ax.twinx()
        ax_right.set_ylabel('$x_2^B$')
        ax_right.set_ylim(1, 0)

        # vi. Create the twin x-axis that is inverted for consumer B
        ax_top = ax.twiny()
        ax_top.set_xlabel('$x_1^B$')
        ax_top.set_xlim(1, 0)

        # vii. Set limits for all axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # viii. Add title and show the plot of the equilibrium allocations
        plt.title('Market Equilibrium Allocations in the Edgeworth Box')
        plt.show()