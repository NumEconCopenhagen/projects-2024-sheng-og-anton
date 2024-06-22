


# Problem 2: Career Choice Model


import numpy as np
from types import SimpleNamespace
import matplotlib.pyplot as plt
class CareerChoice:

    def __init__(self, J=3, N=10, K=10000, sigma=2, v=np.array([1, 2, 3]), c=1):
        self.par = SimpleNamespace()
        self.J = J  # Number of career tracks
        self.N = N  # Number of graduates
        self.K = K  # Number of simulations
        self.sigma = sigma  # Standard deviation
        self.v = v  # Array of values for each career track
        self.c = c  # Switching cost
    
    def simulate_career_choices(self):
        # Simulate the expected utility
        np.random.seed(42)  # For reproducibility
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
        # Storage for results
        career_choices = np.zeros((self.N, self.J))
        subjective_utilities = np.zeros(self.N)
        realized_utilities = np.zeros(self.N)

        # Simulation
        np.random.seed(42)  # For reproducibility
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

        
