import numpy as np
from scipy.optimize import minimize
from scipy import optimize
import pandas as pd
import matplotlib.pyplot as plt
from types import SimpleNamespace
from scipy.optimize import minimize_scalar

class ProductionEconomyClass:
    def __init__(self):
        par = self.par = SimpleNamespace()

        # a. parameters
        par.A = 1.0
        par.gamma = 0.5
        par.alpha = 0.3
        par.nu = 1.0
        par.epsilon = 2.0
        par.tau = 0.0
        par.T = 0.0
        par.num_p1 = 10
        par.num_p2 = 10
        par.grid_p1 = np.linspace(0.1, 2.0, par.num_p1)
        par.grid_p2 = np.linspace(0.1, 2.0, par.num_p2)

        # b. solution
        sol = self.sol = SimpleNamespace()
        sol.w = 1  # numeraire

    def firm_behavior(self, p):
        """ Firm behavior """
        par = self.par
        w = self.sol.w
        l_star = (p * par.A * par.gamma / w) ** (1 / (1 - par.gamma))
        y_star = par.A * l_star ** par.gamma
        pi_star = ((1 - par.gamma) / par.gamma) * w * l_star
        return l_star, y_star, pi_star

    def consumer_behavior(self, l, p1, p2):
        """ Consumer behavior """
        par = self.par
        w = self.sol.w
        T = par.T
        pi1_star = self.firm_behavior(p1)[2]
        pi2_star = self.firm_behavior(p2)[2]
        c1 = par.alpha * (w * l + T + pi1_star + pi2_star) / p1
        c2 = (1 - par.alpha) * (w * l + T + pi1_star + pi2_star) / (p2 + par.tau)
        return c1, c2

    def labor_supply(self, p1, p2):
        """ Labor supply """
        par = self.par
        w = self.sol.w
        T = par.T
        l_values = np.linspace(0, 10, 100)
        utilities = [np.log((self.consumer_behavior(l, p1, p2)[0] ** par.alpha) * (self.consumer_behavior(l, p1, p2)[1] ** (1 - par.alpha))) - par.nu * (l ** (1 + par.epsilon)) / (1 + par.epsilon) for l in l_values]
        max_index = np.argmax(utilities)
        l = l_values[max_index]
        return l

    def evaluate_equilibrium(self, p1, p2):
        """ Evaluate equilibrium for given p1 and p2 """
        l = self.labor_supply(p1, p2)
        c1, c2 = self.consumer_behavior(l, p1, p2)
        l1, y1, _ = self.firm_behavior(p1)
        l2, y2, _ = self.firm_behavior(p2)
        eps1 = c1 - y1
        eps2 = c2 - y2
        return eps1, eps2

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
        errors = []

        for p1 in par.grid_p1:
            for p2 in par.grid_p2:
                eps1, eps2 = self.evaluate_equilibrium(p1, p2)
                errors.append((eps1, eps2))
                print(f'p1 = {p1:.2f}, p2 = {p2:.2f} -> excess demand 1 = {eps1:12.8f}, excess demand 2 = {eps2:12.8f}')

        return errors

# Problem 2: Career Choice Model
class CareerChoiceModelClass:
    def __init__(self, J=3, N=10, K=10000, sigma=2, v=np.array([1, 2, 3]), c=1):
        
        self.par = SimpleNamespace()

        self.J = J  # Number of career tracks
        self.N = N  # Number of graduates
        self.K = K  # Number of simulations
        self.sigma = sigma  # Standard deviation
        self.v = v  # Array of values for each career track
        self.c = c  # Switching cost
        np.random.seed(69) # Seed to reproduce results
    
    def simulate_career_choices(self):
        """


        """
        # Simulate the expected utility
        epsilon = np.random.normal(0, self.sigma, (self.J, self.K))
        
        # Calculate expected utility for each career track
        expected_utility = self.v[:, np.newaxis] + np.mean(epsilon, axis=1)[:, np.newaxis]
        average_expected_utility = np.mean(expected_utility, axis=1)
        
        # Realized utility
        realized_utility = self.v + np.mean(epsilon, axis=1)
        
        # Print results
        print("Average Expected Utility for each career track:")
        for j in range(self.J):
            print(f"Career choice {j+1}: {average_expected_utility[j]:.4f}")
        
        print("\nAverage Realized Utility for each career track:")
        for j in range(self.J):
            print(f"Career choice {j+1}: {realized_utility[j]:.4f}")
    
    def simulate_and_plot(self):
        """


        """
        # Storage for results
        career_choices = np.zeros((self.N, self.J))
        subjective_utilities = np.zeros(self.N)
        realized_utilities = np.zeros(self.N)

        # Simulation
        for k in range(self.K):
            for i in range(self.N):
                F_i = i + 1  # Number of friends for graduate i (1 to N)
                
                # Simulate the noise terms for friends
                epsilon_friends = np.random.normal(0, self.sigma, (self.J, F_i))
                
                # Calculate prior expected utility for each career track
                prior_expected_utility = (self.v[:, np.newaxis] + epsilon_friends).mean(axis=1)
                
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
        for j in range(self.J):
            plt.plot(range(1, self.N + 1), career_share[:, j], label=f'Career {j+1}')
        plt.xlabel('Graduate Type (i)')
        plt.ylabel('Share Choosing Career')
        plt.legend()
        plt.title('Share of Graduates Choosing Each Career')

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

    def simulate_and_plot_switching(self):
        """


        """
        # Storage for results
        initial_career_choices = np.zeros((self.N, self.J))
        final_career_choices = np.zeros((self.N, self.J))
        subjective_utilities = np.zeros(self.N)
        realized_utilities = np.zeros(self.N)
        switch_counts = np.zeros((self.N, self.J))  # Track switching counts by initial choice

        # Simulation
        for k in range(self.K):
            for i in range(self.N):
                F_i = i + 1  # Number of friends for graduate i (1 to N)

                # Simulate the noise terms for friends
                epsilon_friends = np.random.normal(0, self.sigma, (self.J, F_i))

                # Calculate prior expected utility for each career track
                prior_expected_utility = self.v + np.mean(epsilon_friends, axis=1)

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

        # Share of graduates choosing each career initially
        plt.subplot(2, 3, 1)
        for j in range(self.J):
            plt.plot(range(1, self.N + 1), initial_career_share[:, j], label=f'Career {j+1}')
        plt.xlabel('Graduate Type (i)')
        plt.ylabel('Share Choosing Career')
        plt.legend()
        plt.title('Share of Graduates Choosing Each Career Initially')

        # Share of graduates choosing each career after potential switch
        plt.subplot(2, 3, 2)
        for j in range(self.J):
            plt.plot(range(1, self.N + 1), final_career_share[:, j], label=f'Career {j+1}')
        plt.xlabel('Graduate Type (i)')
        plt.ylabel('Share Choosing Career')
        plt.legend()
        plt.title('Share of Graduates Choosing Each Career After Potential Switch')

        # Average subjective expected utility
        plt.subplot(2, 3, 3)
        plt.plot(range(1, self.N + 1), avg_subjective_utility, marker='o')
        plt.xlabel('Graduate Type (i)')
        plt.ylabel('Average Subjective Expected Utility')
        plt.title('Average Subjective Expected Utility')

        # Average realized utility
        plt.subplot(2, 3, 4)
        plt.plot(range(1, self.N + 1), avg_realized_utility, marker='o')
        plt.xlabel('Graduate Type (i)')
        plt.ylabel('Average Realized Utility')
        plt.title('Average Realized Utility')

        # Share of graduates switching careers
        plt.subplot(2, 3, 5)
        for j in range(self.J):
            plt.plot(range(1, self.N + 1), switch_share[:, j], label=f'Initial Career {j+1}')
        plt.xlabel('Graduate Type (i)')
        plt.ylabel('Share Switching Careers')
        plt.legend()
        plt.title('Share of Graduates Switching Careers')

        plt.tight_layout()
        plt.show()

# 
class BarycentricInterpolationClass:
    """
    A class which does barycentric interpolation and finds a point (y) inside a triangles using an algorithm.

    Attributes:
        X (np.ndarray): A NumPy array of random points in a unit square.
            x (float): The points which are (pseduo)-random generated using the uniform distribution (0, 1).
        F (np.ndarray): A NumPy array of the values of the function at the points given in the X set. 
            f (float): Each of the elements.
    """
    
    def __init__(self, X, F): 
        """
        Initializes the BarycentricInterpolationClass with the points and function values.

        Parameters:
            X (np.ndarray): The NumPy array of points.
                x (float): The points which are (pseduo)-random generated using the uniform distribution (0, 1).
            F (np.ndarray): The array of function values at the points.
                f (float): Each of the elements.
        """
        self.X = X
        self.F = F

    def find_the_nearest_point(self, y, condition):
        """
        Finds the nearest point in X to y that satisfies the given condition.

        Args:
            y (np.ndarray): The coordinates for the point y.
            condition: A function that takes a point and y, and returns True if the point satisfies the condition.

        Returns:
            (np.ndarray): The nearest point satisfying the condition.
            (None): If the condition (s.t.) is not met.
        """
        # i. Calculates the distances for the points
        distances = np.sqrt(np.sum((self.X - y) ** 2, axis=1))
        # ii. A list comprehension for the points the X set to check the condition (s.t.) x1 > y1 and x2 > y2  
        valid_indices = [i for i in range(len(self.X)) if condition(self.X[i], y)]
        # iii. If the points in the X set does not satisfy the condition, then the result will be None
        if not valid_indices:
            return None
        # iv. Minimize the distances to find the nearest point  
        nearest_index = min(valid_indices, key=lambda i: distances[i])
        return self.X[nearest_index]

    def find_the_triangle_points(self, y):
        """
        Finds the points A, B, C, and D as described in the algorithm.

        Args:
            y (np.ndarray): The point y.

        Returns:
            A (np.ndarray): A NumPy array that contains the coordinates for the point A. 
            B (np.ndarray): A NumPy array that contains the coordinates for the point B. 
            C (np.ndarray): A NumPy array that contains the coordinates for the point C. 
            D (np.ndarray): A NumPy array that contains the coordinates for the point D. 
        """
        # i. Finds the points using the minimization and the conditions (s.t.)
        A = self.find_the_nearest_point(y, lambda p, y: p[0] > y[0] and p[1] > y[1])
        B = self.find_the_nearest_point(y, lambda p, y: p[0] > y[0] and p[1] < y[1])
        C = self.find_the_nearest_point(y, lambda p, y: p[0] < y[0] and p[1] < y[1])
        D = self.find_the_nearest_point(y, lambda p, y: p[0] < y[0] and p[1] > y[1])

        return A, B, C, D

    def barycentric_coordinates(self, A, B, C, D, y, triangle_type):
        """
        Computes the three barycentric coordinates (r1, r2, and r3) of y in the ABC triangle.

        Args:
            A, 
            B, C (np.ndarray): The points of the triangle.
            y (np.ndarray): The coordinates for the point y.

        Returns:
            r1 (float): The first barycentric coordinate for the ABC triangle.
            r2 (float): The second barycentric coordinate for the ABC triangle.
            r3 (float): The third barycentric coordinate for the ABC triangle.
        """
        # ii. If the triangle type is ABC then it calculates the barycentric coordinates (r1 and r2) for the ABC triangle
        if triangle_type == 'ABC':
            denominator = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
            r1 = ((B[1] - C[1]) * (y[0] - C[0]) + (C[0] - B[0]) * (y[1] - C[1])) / denominator
            r2 = ((C[1] - A[1]) * (y[0] - C[0]) + (A[0] - C[0]) * (y[1] - C[1])) / denominator
        # iii. If the triangle type is CDA then it calculates the barycentric coordinates (r1 and r2) for the CDA triangle
        elif triangle_type == 'CDA':
            denominator = (D[1] - A[1]) * (C[0] - A[0]) + (A[0] - D[0]) * (C[1] - A[1])
            r1 = ((D[1] - A[1]) * (y[0] - A[0]) + (A[0] - D[0]) * (y[1] - A[1])) / denominator
            r2 = ((A[1] - C[1]) * (y[0] - A[0]) + (C[0] - A[0]) * (y[1] - A[1])) / denominator
        # iv. ValueError message if another type of triangle was typed
        else:
            raise ValueError("No, that is not a correct triangle. Only 'ABC' or 'CDA' can be entered.")
        
        # v. Calculates the third barycentric coordinate
        r3 = 1 - r1 - r2
        
        # iii. Rounds the barycentric coordinates to two decimals
        r1 = round(r1, 2)
        r2 = round(r2, 2)
        r3 = round(r3, 2)

        return r1, r2, r3

    def inside_triangle_check(self, r):
        """
        Checks if the point with barycentric coordinates (r1, r2, and r3) is inside the triangle.

        Args:
            r (tuple): The barycentric coordinates (r1, r2, r3).
                r1 (float): The first barycentric coordinate.
                r2 (float): The second barycentric coordinate.
                r3 (float): The third barycentric coordinate.

        Returns:
            (boolean): Returns true if the point y is inside the given triangle, returns false if the point y is outside the given triangle.
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
        # i. Finds the points of the triangle from previous method 
        A, B, C, D = self.find_the_triangle_points(y)
        
        # ii. If the points of the two triangles are none
        if A is None or B is None or C is None or D is None:
            return np.nan
        
        # iii.
        rABC = self.barycentric_coordinates(A, B, C, D, y, triangle_type='ABC')
        rCDA = self.barycentric_coordinates(A, B, C, D, y, triangle_type='CDA')
        
        # iv.
        rABC = np.round(rABC, 2)
        rCDA = np.round(rCDA, 2)

        # v.
        if self.inside_triangle_check(rABC):
            return rABC[0] * self.F[np.where((self.X == A).all(axis=1))[0][0]] + \
                   rABC[1] * self.F[np.where((self.X == B).all(axis=1))[0][0]] + \
                   rABC[2] * self.F[np.where((self.X == C).all(axis=1))[0][0]]
        # vi.
        elif self.inside_triangle_check(rCDA):
            return rCDA[0] * self.F[np.where((self.X == C).all(axis=1))[0][0]] + \
                   rCDA[1] * self.F[np.where((self.X == D).all(axis=1))[0][0]] + \
                   rCDA[2] * self.F[np.where((self.X == A).all(axis=1))[0][0]]
        # vii.
        else:
            return np.nan
