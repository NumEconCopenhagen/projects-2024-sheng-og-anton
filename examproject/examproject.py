import numpy as np
from scipy.optimize import root_scalar
from scipy.optimize import minimize, root_scalar
from types import SimpleNamespace
import matplotlib.pyplot as plt

# Problemset 1: Production economy and CO2 taxation

class ProductionEconomy:
    def __init__(self):
        # Parameters
        self.par = SimpleNamespace()
        self.par.A = 1.0  
        self.par.gamma = 0.5  
        self.par.alpha = 0.3  
        self.par.nu = 1.0  
        self.par.epsilon = 2.0 
        self.par.tau = 0.0  
        self.par.T = 0.0  
        self.par.kappa = 0.1  
        
        self.num_p = 10  # Number of grid points for prices
        self.p1_grid = np.linspace(0.1, 2.0, self.num_p)  # Grid for prices of good 1
        self.p2_grid = np.linspace(0.1, 2.0, self.num_p)  # Grid for prices of good 2
        
        self.grid_mkt_clearing_1 = np.zeros((self.num_p, self.num_p))  # Market clearing grid for good 1
        self.grid_mkt_clearing_2 = np.zeros((self.num_p, self.num_p))  # Market clearing grid for good 2
        
    def compute_implied_profits(self, p, w):
        # Compute the implied profits for a given price p and wage w
        return (1 - self.par.gamma) / self.par.gamma * w * (p * self.par.A * self.par.gamma / w) ** (1 / (1 - self.par.gamma))

    def compute_market_clearing(self, p1, p2, w=1.0):
        # Calculate market clearing conditions for given prices p1 and p2, and wage w
        pi1_optimal = self.compute_implied_profits(p1, w)  # Implied profit for firm 1
        pi2_optimal = self.compute_implied_profits(p2, w)  # Implied profit for firm 2
        
        ell1_optimal = (p1 * self.par.A * self.par.gamma / w) ** (1 / (1 - self.par.gamma))  # Optimal labor supply for firm 1
        ell2_optimal = (p2 * self.par.A * self.par.gamma / w) ** (1 / (1 - self.par.gamma))  # Optimal labor supply for firm 2
        ell_optimal = ell1_optimal + ell2_optimal  # Total optimal labor supply
        
        c1_optimal = self.par.alpha * (w * ell_optimal + self.par.T + pi1_optimal + pi2_optimal) / p1  # Optimal consumption of good 1
        c2_optimal = (1 - self.par.alpha) * (w * ell_optimal + self.par.T + pi1_optimal + pi2_optimal) / (p2 + self.par.tau)  # Optimal consumption of good 2
        
        y1_optimal = self.par.A * ell1_optimal ** self.par.gamma  # Optimal production of good 1
        y2_optimal = self.par.A * ell2_optimal ** self.par.gamma  # Optimal production of good 2
        
        labor_clearing = ell_optimal  # Total labor market clearing
        good1_clearing = c1_optimal - y1_optimal  # Market clearing condition for good 1
        good2_clearing = c2_optimal - y2_optimal  # Market clearing condition for good 2
        
        return labor_clearing, good1_clearing, good2_clearing

    def evaluate_equilibrium(self, p1, p2):
        # Evaluate equilibrium conditions for given prices p1 and p2
        good1_clearing, good2_clearing = self.compute_market_clearing(p1, p2)[1:3] 
        return good1_clearing, good2_clearing

    def compute_market_clearing_grid(self):
        # Compute market clearing conditions over a grid of prices
        for i, p1 in enumerate(self.p1_grid):
            for j, p2 in enumerate(self.p2_grid):
                good1_clearing, good2_clearing = self.evaluate_equilibrium(p1, p2)  # Evaluate equilibrium for each pair of prices
                
                self.grid_mkt_clearing_1[i, j] = good1_clearing  # Store market clearing condition for good 1
                self.grid_mkt_clearing_2[i, j] = good2_clearing  # Store market clearing condition for good 2
                
                print(f'p1 = {p1:.2f}, p2 = {p2:.2f} -> Good market 1 = {good1_clearing:.8f}, Good market 2 = {good2_clearing:.8f}')


    def find_equilibrium_price_p1(self, p2_left, p2_right):
        # Find equilibrium price for good 1 with a fixed price range for good 2
        fixed_p2 = np.mean([p2_left, p2_right])  # Fix the price of good 2
        
        def good1(p1):
            # Calculate excess demand for good 1
            return self.evaluate_equilibrium(p1, fixed_p2)[0]  # Only the first component (good 1)
        
        res_p1 = root_scalar(good1, bracket=[self.p1_grid[0], self.p1_grid[-1]], method='bisect')  # Find root
        if res_p1.converged:
            return res_p1.root
        else:
            return None

    def find_equilibrium_price_p2(self, p1_left, p1_right):
        # Find equilibrium price for good 2 with a fixed price range for good 1
        fixed_p1 = np.mean([p1_left, p1_right])  # Fix the price of good 1
        
        def good2(p2):
            # Calculate excess demand for good 2
            return self.evaluate_equilibrium(fixed_p1, p2)[1]  # Only the second component (good 2)
        
        res_p2 = root_scalar(good2, bracket=[self.p2_grid[0], self.p2_grid[-1]], method='bisect')  # Find root
        if res_p2.converged:
            return res_p2.root
        else:
            return None
        
    def compute_market(self, p1, p2, w=1):
        # Compute the market conditions given prices p1, p2, and wage w
        pi1_optimal = self.compute_implied_profits(p1, w)  # Implied profits for firm 1
        pi2_optimal = self.compute_implied_profits(p2, w)  # Implied profits for firm 2
        
        ell1_optimal = (p1 * self.par.A * self.par.gamma / w) ** (1 / (1 - self.par.gamma))  # Optimal labor for firm 1
        ell2_optimal = (p2 * self.par.A * self.par.gamma / w) ** (1 / (1 - self.par.gamma))  # Optimal labor for firm 2
        ell_optimal = ell1_optimal + ell2_optimal  # Total optimal labor supply
        
        c1_optimal = self.par.alpha * (w * ell_optimal + self.par.T + pi1_optimal + pi2_optimal) / p1  # Optimal consumption for good 1
        c2_optimal = (1 - self.par.alpha) * (w * ell_optimal + self.par.T + pi1_optimal + pi2_optimal) / (p2 + self.par.tau)  # Optimal consumption for good 2
        
        y2_optimal = self.par.A * ell2_optimal ** self.par.gamma  # Optimal production of good 2
        
        return c1_optimal, c2_optimal, ell_optimal, y2_optimal

    def objective(self, x, p1, p2):
        # Objective function to minimize (negative social welfare function)
        tau, T = x
        c1_optimal, c2_optimal, ell_optimal, y2_optimal = self.compute_market(p1, p2 + tau, 1.0)
        U = np.log(c1_optimal ** self.par.alpha * c2_optimal ** (1 - self.par.alpha)) - self.par.nu * (ell_optimal ** (1 + self.par.epsilon)) / (1 + self.par.epsilon)  # Utility calculation
        SWF = U - self.par.kappa * y2_optimal  # Social welfare function
        return -SWF  # Return negative SWF to maximize SWF

    def find_optimal_tau_T(self, p1, p2):
        # Find optimal tau and T using numerical optimization
        initial_guesses = [0.01, 0.01]  # Initial guesses for tau and T
        bounds = [(0, None), (0, None)]  # Bounds to ensure non-negative tau and T
        result = minimize(self.objective, initial_guesses, args=(p1, p2), bounds=bounds, method='SLSQP')  # Minimize objective function
        if result.success:
            return result.x[0], result.x[1]
        else:
            return None, None



# Problem 2: Career Choice Model


class CareerChoice:

    def __init__(self, J=3, N=10, K=10000, sigma=2, v=np.array([1, 2, 3]), c=1):
        self.par = SimpleNamespace()
        self.J = J  # Number of career tracks
        self.N = N  # Number of graduates
        self.K = K  # Number of simulations
        self.sigma = sigma  # Standard deviation
        self.v = v  # Array of values for each career track
        self.c = c  # Switching cost

    # Question 1
    def simulate_career_choices(self):
        # Simulate the expected utility
        np.random.seed(69)  # For reproducibility
        epsilon = np.random.normal(0, self.sigma, (self.J, self.K))
        
        # Calculate expected utility for each career track
        expected_utility = self.v[:, np.newaxis] + np.mean(epsilon, axis=1)[:, np.newaxis] / self.K
        average_expected_utility = np.mean(expected_utility, axis=1)
        
        # Realized utility
        realized_utility = self.v + np.mean(epsilon, axis=1)
        
        # Print results
        print("Expected Utility for each career track:")
        for j in range(self.J):
            print(f"Career choice {j+1}: {average_expected_utility[j]:.4f}")
        
        print("\nAverage Realized Utility for each career track:")
        for j in range(self.J):
            print(f"Career choice {j+1}: {realized_utility[j]:.4f}")

    # Question 2
    def simulate_and_plot(self):
        # Storage for results
        career_choices = np.zeros((self.N, self.J))
        subjective_utilities = np.zeros(self.N)
        realized_utilities = np.zeros(self.N)

        # Simulation
        np.random.seed(69)  # For reproducibility
        for k in range(self.K):
            for i in range(self.N):
                F_i = i + 1  # Number of friends for graduate i (1 to N)
                
                # Simulate the noise terms for friends
                epsilon_friends = np.random.normal(0, self.sigma, (self.J, F_i))
                
                # Calculate prior expected utility for each career track
                prior_expected_utility = self.v + np.mean(epsilon_friends, axis=1) / F_i
                
                # Draw own noise terms
                epsilon_own = np.random.normal(0, self.sigma, self.J)
                
                # Choose career track with highest expected utility
                chosen_career = np.argmax(prior_expected_utility)
                
                # Store the chosen career, prior expectation, and realized value
                career_choices[i, chosen_career] += 1
                subjective_utilities[i] += prior_expected_utility[chosen_career]
                realized_utilities[i] += self.v[chosen_career] + epsilon_own[chosen_career]

        # Calculate averages
        career_share = career_choices / self.K
        avg_subjective_utility = subjective_utilities / self.K
        avg_realized_utility = realized_utilities / self.K

        
        # Plot the results
        plt.figure(figsize=(18, 6))

        # Share of graduates choosing each career
        plt.subplot(1, 3, 1)
        bar_width = 0.2
        for j in range(self.J):
            plt.bar(np.arange(self.N) + j * bar_width, career_share[:, j], width=bar_width, label=f'Career {j+1}')

        plt.xlabel('Graduate Type (i)')
        plt.ylabel('Share Choosing Career')
        plt.title('Share of Graduates Choosing Each Career')
        plt.xticks(np.arange(self.N) + (self.J - 1) * bar_width / 2, np.arange(1, self.N + 1))
        plt.legend()

        # Average subjective expected utility
        plt.subplot(1, 3, 2)
        plt.plot(range(1, self.N + 1), avg_subjective_utility, marker='o')
        plt.xlabel('Graduate Type (i)')
        plt.ylabel('Average Subjective Expected Utility')
        plt.title('Average Subjective Expected Utility')

        # Average realized utility
        plt.subplot(1, 3, 3)
        plt.plot(range(1, self.N + 1), avg_realized_utility, marker='o')
        plt.xlabel('Graduate Type (i)')
        plt.ylabel('Average Realized Utility')
        plt.title('Average Realized Utility')

        plt.tight_layout()
        plt.show()
    
    #Question 3
    def simulate_and_plot_switching(self):
        # Storage for results
        initial_career_choices = np.zeros((self.N, self.J))
        final_career_choices = np.zeros((self.N, self.J))
        subjective_utilities = np.zeros(self.N)
        realized_utilities = np.zeros(self.N)
        switch_counts = np.zeros((self.N, self.J))  # Track switching counts by initial choice

        # Simulation
        np.random.seed(42)  # For reproducibility
        for k in range(self.K):
            for i in range(self.N):
                F_i = i + 1  # Number of friends for graduate i (1 to N)

                # Simulate the noise terms for friends
                epsilon_friends = np.random.normal(0, self.sigma, (self.J, F_i))

                 # Calculate prior expected utility for each career track
                prior_expected_utility = self.v + np.mean(epsilon_friends, axis=1) / F_i

                # Draw own noise terms
                epsilon_own = np.random.normal(0, self.sigma, self.J)

                # Year 1: Initial choice
                initial_chosen_career = np.argmax(prior_expected_utility)
                initial_realized_utility = self.v[initial_chosen_career] + epsilon_own[initial_chosen_career]

                # Store initial results
                initial_career_choices[i, initial_chosen_career] += 1
                subjective_utilities[i] += prior_expected_utility[initial_chosen_career]
                realized_utilities[i] += initial_realized_utility

                # Year 2: Calculate new priors and utilities after potential switch
                final_prior_expected_utility = prior_expected_utility - self.c
                final_prior_expected_utility[initial_chosen_career] = initial_realized_utility

                # Determine final career choice and realized utility
                final_chosen_career = np.argmax(final_prior_expected_utility)
                if final_chosen_career != initial_chosen_career:
                    final_realized_utility = self.v[final_chosen_career] + epsilon_own[final_chosen_career] - self.c
                else:
                    final_realized_utility = initial_realized_utility

                # Store final results
                final_career_choices[i, final_chosen_career] += 1
                realized_utilities[i] += final_realized_utility - initial_realized_utility

                # Track switching counts
                switch_counts[i, initial_chosen_career] += 1 if final_chosen_career != initial_chosen_career else 0

        # Calculate averages
        initial_career_share = initial_career_choices / self.K
        final_career_share = final_career_choices / self.K
        avg_subjective_utility = subjective_utilities / self.K
        avg_realized_utility = realized_utilities / self.K
        switch_share = switch_counts / self.K

         # Plot the results
        plt.figure(figsize=(18, 12))

        # Share of graduates choosing each career initially (Bar plot)
        plt.subplot(2, 3, 1)
        bar_width = 0.35
        for j in range(self.J):
            plt.bar(np.arange(self.N) + j * bar_width, initial_career_share[:, j], width=bar_width, label=f'Career {j+1}')
        plt.xlabel('Graduate Type (i)')
        plt.ylabel('Share Choosing Career')
        plt.xticks(np.arange(self.N) + bar_width, np.arange(1, self.N + 1))
        plt.legend()
        plt.title('Share of Graduates Choosing Each Career Initially')

        # Share of graduates choosing each career after potential switch (Bar plot)
        plt.subplot(2, 3, 2)
        for j in range(self.J):
            plt.bar(np.arange(self.N) + j * bar_width, final_career_share[:, j], width=bar_width, label=f'Career {j+1}')
        plt.xlabel('Graduate Type (i)')
        plt.ylabel('Share Choosing Career')
        plt.xticks(np.arange(self.N) + bar_width, np.arange(1, self.N + 1))
        plt.legend()
        plt.title('Share of Graduates Choosing Each Career After Potential Switch')

        # Average subjective expected utility (Line plot)
        plt.subplot(2, 3, 3)
        plt.plot(range(1, self.N + 1), avg_subjective_utility, marker='o')
        plt.xlabel('Graduate Type (i)')
        plt.ylabel('Average Subjective Expected Utility')
        plt.title('Average Subjective Expected Utility')

        # Average realized utility (Line plot)
        plt.subplot(2, 3, 4)
        plt.plot(range(1, self.N + 1), avg_realized_utility, marker='o')
        plt.xlabel('Graduate Type (i)')
        plt.ylabel('Average Realized Utility')
        plt.title('Average Realized Utility')

        # Share of graduates switching careers (Bar plot)
        plt.subplot(2, 3, 5)
        for j in range(self.J):
            plt.bar(np.arange(self.N) + j * bar_width, switch_share[:, j], width=bar_width, label=f'Initial Career {j+1}')
        plt.xlabel('Graduate Type (i)')
        plt.ylabel('Share Switching Careers')
        plt.xticks(np.arange(self.N) + bar_width, np.arange(1, self.N + 1))
        plt.legend()
        plt.title('Share of Graduates Switching Careers')

        plt.tight_layout()
        plt.show()

# 
class BarycentricInterpolationClass:
    """
    A class which does barycentric interpolation and finds a point (y) inside triangles using an algorithm.

    Attributes:
        X (np.ndarray): A NumPy array of random points in a unit square.
    """
    
    def __init__(self, X): 
        """
        Initializes the BarycentricInterpolationClass with the points.

        Parameters:
            X (np.ndarray): The NumPy array of points.
        """
        self.X = X
        self.F = None  # F is not set during initialization

    def set_F(self, F):
        """
        Sets the function values F.

        Parameters:
            F (np.ndarray): The array of function values at the points.
        """
        self.F = F

    def find_the_nearest_point(self, y, condition):
        """
        Finds the nearest point in X to y that satisfies the given condition.

        Args:
            y (np.ndarray): The coordinates for the point y.
            condition: A function that takes a point and y, and returns True if the point satisfies the condition.

        Returns:
            (np.ndarray): The nearest point satisfying the condition.
            (None): If the condition is not met.
        """
        distances = np.sqrt(np.sum((self.X - y) ** 2, axis=1))
        valid_indices = [i for i in range(len(self.X)) if condition(self.X[i], y)]
        if not valid_indices:
            return None
        nearest_index = min(valid_indices, key=lambda i: distances[i])
        return self.X[nearest_index]

    def find_the_triangle_points(self, y):
        """
        Finds the points A, B, C, and D as described in the algorithm.

        Args:
            y (np.ndarray): The coordinates for the point y.

        Returns:
            A (np.ndarray): A NumPy array that contains the coordinates for the point A. 
            B (np.ndarray): A NumPy array that contains the coordinates for the point B. 
            C (np.ndarray): A NumPy array that contains the coordinates for the point C. 
            D (np.ndarray): A NumPy array that contains the coordinates for the point D. 
        """
        A = self.find_the_nearest_point(y, lambda p, y: p[0] > y[0] and p[1] > y[1])
        B = self.find_the_nearest_point(y, lambda p, y: p[0] > y[0] and p[1] < y[1])
        C = self.find_the_nearest_point(y, lambda p, y: p[0] < y[0] and p[1] < y[1])
        D = self.find_the_nearest_point(y, lambda p, y: p[0] < y[0] and p[1] > y[1])
        return A, B, C, D

    def barycentric_coordinates(self, A, B, C, D, y, triangle_type):
        """
        Computes the three barycentric coordinates (r1, r2, and r3) of y in the specified triangle.

        Args:
            A, B, C, D (np.ndarray): The points of the triangle.
            y (np.ndarray): The coordinates for the point y.
            triangle_type (str): The type of triangle ('ABC' or 'CDA').

        Returns:
            r1 (float): The first barycentric coordinate.
            r2 (float): The second barycentric coordinate.
            r3 (float): The third barycentric coordinate.
        """
        if triangle_type == 'ABC':
            denominator = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
            r1 = ((B[1] - C[1]) * (y[0] - C[0]) + (C[0] - B[0]) * (y[1] - C[1])) / denominator
            r2 = ((C[1] - A[1]) * (y[0] - C[0]) + (A[0] - C[0]) * (y[1] - C[1])) / denominator
        elif triangle_type == 'CDA':
            denominator = (D[1] - A[1]) * (C[0] - A[0]) + (A[0] - D[0]) * (C[1] - A[1])
            r1 = ((D[1] - A[1]) * (y[0] - A[0]) + (A[0] - D[0]) * (y[1] - A[1])) / denominator
            r2 = ((A[1] - C[1]) * (y[0] - A[0]) + (C[0] - A[0]) * (y[1] - A[1])) / denominator
        else:
            raise ValueError("Invalid triangle type. Only 'ABC' or 'CDA' can be entered.")
        
        r3 = 1 - r1 - r2
        return round(r1, 2), round(r2, 2), round(r3, 2)

    def inside_triangle_check(self, r):
        """
        Checks if the point with barycentric coordinates (r1, r2, and r3) is inside the triangle.

        Args:
            r (tuple): The barycentric coordinates (r1, r2, r3).

        Returns:
            (boolean): True if the point is inside the given triangle, False otherwise.
        """
        return all(0 <= barycentric_coordinate <= 1 for barycentric_coordinate in r)

    def interpolate(self, y):
        """
        Interpolates the value of the function at point y using the barycentric interpolation algorithm.

        Args:
            y (np.ndarray): The coordinates (x and y) for the point y.

        Returns:
            (float): Returns the interpolated value if the point y is inside either ABC triangle or the CDA triangle. 
            (np.ndarray): Returns NaN if the point y is not inside the ABC triangle or the CDA triangle.
        """
        if self.F is None:
            raise ValueError("Function values F are not set. Please set F using the set_F method.")
        
        A, B, C, D = self.find_the_triangle_points(y)
        
        if A is None or B is None or C is None or D is None:
            return np.nan
        
        rABC = self.barycentric_coordinates(A, B, C, D, y, triangle_type='ABC')
        rCDA = self.barycentric_coordinates(A, B, C, D, y, triangle_type='CDA')
        
        rABC = np.round(rABC, 2)
        rCDA = np.round(rCDA, 2)

        if self.inside_triangle_check(rABC):
            return rABC[0] * self.F[np.where((self.X == A).all(axis=1))[0][0]] + \
                   rABC[1] * self.F[np.where((self.X == B).all(axis=1))[0][0]] + \
                   rABC[2] * self.F[np.where((self.X == C).all(axis=1))[0][0]]
        elif self.inside_triangle_check(rCDA):
            return rCDA[0] * self.F[np.where((self.X == C).all(axis=1))[0][0]] + \
                   rCDA[1] * self.F[np.where((self.X == D).all(axis=1))[0][0]] + \
                   rCDA[2] * self.F[np.where((self.X == A).all(axis=1))[0][0]]
        else:
            return np.nan
