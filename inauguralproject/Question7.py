# Exercise 7
# 1. First we set a seed, so that we can get the same pseduo-random numbers for the endowments everytime we run the code.
np.random.seed(69)

# 2. Now we need to set the pseduo-random number of elements (50) in the set of endowments W 
n = 50

# 3. We generate psedou-random numbers for the endowments omegaA1 and omegaA2
omegaA1 = np.random.uniform(0, 1, n)
omegaA2 = np.random.uniform(0, 1, n)

# 4. We now create the set of endowments based on our pseduo-random numbers
W = np.column_stack((omegaA1, omegaA2))

# 5. We can now begin plotting the total endowments
plt.figure(figsize=(8, 6))
plt.scatter(wA1, wA2, c='green')
plt.title('The set of endowments with 50 elements')

# a. We label the two axes in the diagram 
plt.xlabel('ωA1')
plt.ylabel('ωA2')

# b. We add grids to the plot
plt.grid(True)

# c. We show the plot
plt.show()