# 1. Import packages
import numpy as np # Used to store the values in DataFrames
from scipy import optimize # Used this to root maximize the steady state values
from types import SimpleNamespace
import matplotlib.pyplot as plt # Use this to plot graphs

# 2. Update the standard parameters in matplotlib
plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"--"})
plt.rcParams.update({'font.size': 15})

# 3. We create a class that defines all of the variables, parameters, and equations for the RBC model
class RealBusinessCycleModel(object):
    def __init__(self, params=None):
        # a. We define the number of parameters
        self.k_params = 4
        # b. Define the number of variables
        self.k_variables = 6
        # c. Checks if the parameter is equal to none
        if params is not None:
            # i. Update the instance with values from params
            self.update(params)
    
    # a. Create a new class that updates the elements in the tuple
    def update(self, params):
        # i. The first element in the tuple should be the discount rate, beta
        self.discount_rate = params[0]
        # ii. The second element is the disutility from labour, psi
        self.disutility_from_labour = params[1]
        # iii. The third element is the depreciation rate, delta
        self.depreciation_rate = params[2]
        # iv. The fourth element is the capital share, alpha
        self.capital_share = params[3]
        # v. The fifth element is technology, A
        self.technology = params[4]
    
    # b. Define the root-evaluated variables for both period t and period t+1
    def root_evaluated_variables(self, next_period_log_variables, this_period_log_variables):
        # i. The root-evaluated variables for period t
        (next_period_log_output, next_period_log_consumption, next_period_log_investment,
         next_period_log_labour, next_period_log_leisure, next_period_log_capital) = next_period_log_variables
        
        # ii. The root-evaluated variables for period t+1
        (this_period_log_output, this_period_log_consumption, this_period_log_investment, this_period_log_labour,
         this_period_log_leisure, this_period_log_capital) = this_period_log_variables
        
        # iii. We return a NumPy array with the five equations in the model
        return np.r_[
            self.log_first_order_condition(next_period_log_consumption, next_period_log_labour,
                next_period_log_capital
            ),
            self.log_euler_equation(
                next_period_log_consumption, next_period_log_labour, next_period_log_capital, 
                next_period_log_consumption
            ),
            self.log_production_function(
                next_period_log_output, next_period_log_labour, next_period_log_capital
            ),
            self.log_resource_constraint(
                next_period_log_output, next_period_log_consumption, next_period_log_investment
            ),
            self.log_capital_accumulation(
                next_period_log_capital, next_period_log_investment, next_period_log_capital
            ),
            self.log_labour_leisure_constraint(
                next_period_log_labour, next_period_log_leisure
            ),
        ]
    
    # c. We define the first equation in the model, which is the FOC (1)
    def log_first_order_condition(self, next_period_log_consumption, next_period_log_labour,
                       next_period_log_capital):
        # i. Returns the result from the FOC (logged) below
        return (
            np.log(self.disutility_from_labour) +
            next_period_log_consumption -
            np.log(1 - self.capital_share) -
            self.technology -
            self.capital_share * (next_period_log_capital - next_period_log_labour)
        )
    
    # d. We define the second equation in the RBC model, which is the consumption Euler equation (2)
    def log_euler_equation(self, next_period_log_consumption, next_period_log_labour,
                            next_period_log_capital, this_period_log_consumption):
        # i. Returns the result from the logged consumption euler equation below
        return (
            -this_period_log_consumption -
            np.log(self.discount_rate) +
            next_period_log_consumption -
            np.log(
                (self.capital_share *
                 np.exp((1 - self.capital_share) * next_period_log_labour) /
                 np.exp((1 - self.capital_share) * next_period_log_capital)) +
                (1 - self.depreciation_rate)
            )
        )
    
    # e. Define the third equation in the RBC model, which is the Cobb-Douglass production function
    def log_production_function(self, next_period_log_output, next_period_log_labour, next_period_log_capital):
        # i. Returns the result from the equation for the logged production function below
        return (
            next_period_log_output - self.capital_share * next_period_log_capital -
            (1 - self.capital_share) * next_period_log_labour
        )
    
    # f. Define the fourth equation in the RBC model, which is the resource constraint for the economy
    def log_resource_constraint(self, next_period_log_output, next_period_log_consumption,
                                          next_period_log_investment):
        # i. 
        return (
            next_period_log_output -
            np.log(np.exp(next_period_log_consumption) + np.exp(next_period_log_investment))
        )
    
    # g. Define the fifth equation in the RBC model, which is the capital accumulation
    def log_capital_accumulation(self, next_period_log_capital, this_period_log_investment, this_period_log_capital):
        # i. Returns the result from the equation for the capital accumulation
        return (
            next_period_log_capital -
            np.log(np.exp(this_period_log_investment) + (1 - self.depreciation_rate) * np.exp(this_period_log_capital))
        )
    
    # h. Define the sixth equation in the RBC model, which is the labour-leisure constraint
    def log_labour_leisure_constraint(self, next_period_log_labour, next_period_log_leisure):
        # i.
        return (
            -np.log(np.exp(next_period_log_labour) + np.exp(next_period_log_leisure))
        )

# 4. Next, we need to make a class that calculates the numerical solution to the RBC model
class NumericalSolution(RealBusinessCycleModel):
    # a. Define the 
    def steady_state_numeric(self):
            # i. Setup the starting parameters for the variables
            start_log_variable = [0.5] * self.k_variables

            # ii. Setup the function to finding the root
            root_evaluated_variables = lambda log_variable: self.root_evaluated_variables(log_variable, log_variable)

            # iii. Apply the root-finding algorithm
            solution = optimize.root(root_evaluated_variables, start_log_variable)
            
            # iv. Outputs the numerical solution to the Real Business Cycle Model 
            return np.exp(solution.x)

# 5. 
class SteadyStatePlot:
    # a.
    def __init__(self, variables, steady_state_values):
        # i.
        self.variables = variables
        # ii. 
        self.steady_state_values = steady_state_values
        
    # b.
    def simpleplot(self):
        # i. Sets the size of the figure
        plt.figure(figsize=(10, 6))
        # ii. Sets the bar diagram with the variables and steady state values as the input
        plt.bar(self.variables, self.steady_state_values['Value'], color='skyblue')
        # iii. 
        plt.title('Steady state')
        # iv. 
        plt.xlabel('Variables')
        # v.
        plt.ylabel('Steady State values')
        # vi.
        plt.xticks(rotation=45)
        # vii. 
        plt.grid(axis='y', linestyle='--')
        # viii. 
        plt.show()

# 6. Replacing production function with CES function
# Define the RBC_CES class
class RBC_CES(object):
    # 
    def __init__(self, params=None):
        #
        self.k_params = 5
        #
        self.k_variables = 6
        #
        if params is not None:
            #
            self.update(params)
    
    #
    def update(self, params):
        #
        self.discount_rate = params[0]
        #
        self.disutility_from_labour = params[1]
        #
        self.depreciation_rate = params[2]
        #
        self.capital_share = params[3]
        #
        self.technology = params[4]
        #
        self.rho = params[5]
    
    # 
    def root_evaluated_variables(self, next_period_log_variables, this_period_log_variables):
        # 
        (next_period_log_output, next_period_log_consumption, next_period_log_investment,
         next_period_log_labour, next_period_log_leisure, next_period_log_capital) = next_period_log_variables
        
        #
        (this_period_log_output, this_period_log_consumption, this_period_log_investment, this_period_log_labour,
         this_period_log_leisure, this_period_log_capital) = this_period_log_variables
        
        #
        return np.r_[
            self.log_first_order_condition(
                next_period_log_consumption, next_period_log_labour,
                next_period_log_capital
            ),
            self.log_euler_equation(
                next_period_log_consumption, next_period_log_labour,
                next_period_log_capital, next_period_log_consumption
            ),
            self.log_ces_function(
                next_period_log_output, next_period_log_labour, next_period_log_capital
            ),
            self.log_resource_constraint(
                next_period_log_output, next_period_log_consumption,
                next_period_log_investment
            ),
            self.log_capital_accumulation(
                next_period_log_capital, next_period_log_investment, next_period_log_capital
            ),
            self.log_labour_leisure_constraint(
                next_period_log_labour, next_period_log_leisure
            ),
        ]
    
    #
    def log_first_order_condition(self, next_period_log_consumption, next_period_log_labour,
                       next_period_log_capital):
        #
        return (
            np.log(self.disutility_from_labour) +
            next_period_log_consumption -
            np.log(1 - self.capital_share) -
            self.technology -
            self.capital_share * (next_period_log_capital - next_period_log_labour)
        )
    
    #
    def log_euler_equation(self, next_period_log_consumption, next_period_log_labour,
                            next_period_log_capital, this_period_log_consumption):
        #
        return (
            -this_period_log_consumption -
            np.log(self.discount_rate) +
            next_period_log_consumption -
            np.log(
                (self.capital_share * 
                 np.exp((1 - self.capital_share) * next_period_log_labour) /
                 np.exp((1 - self.capital_share) * next_period_log_capital)) +
                (1 - self.depreciation_rate)
            )
        )
    
    #
    def log_ces_function(self, next_period_log_output, next_period_log_labour, next_period_log_capital):
        #
        if self.rho == 0:
        # Cobb-Douglas production function
         return next_period_log_output - (self.capital_share * next_period_log_capital + (1 - self.capital_share) * next_period_log_labour)
        #
        else:
        # CES production function
         return next_period_log_output - np.log(
            (self.capital_share * np.exp(self.rho * next_period_log_capital) +
            (1 - self.capital_share) * np.exp(self.rho * next_period_log_labour)) ** (1 / self.rho)
        )

    # 
    def log_resource_constraint(self, next_period_log_output, next_period_log_consumption,
                                          next_period_log_investment):
        # 
        return (
            next_period_log_output -
            np.log(np.exp(next_period_log_consumption) + np.exp(next_period_log_investment))
        )
    
    #
    def log_capital_accumulation(self, next_period_log_capital, this_period_log_investment, this_period_log_capital):
        #
        return (
            next_period_log_capital -
            np.log(np.exp(this_period_log_investment) + (1 - self.depreciation_rate) * np.exp(this_period_log_capital))
        )
    
    #
    def log_labour_leisure_constraint(self, next_period_log_labour, next_period_log_leisure):
        #
        return (
            -np.log(np.exp(next_period_log_labour) + np.exp(next_period_log_leisure))
        )


class NumericalSolution(RealBusinessCycleModel):
    def steady_state_numeric(self):
            # Setup starting parameters
            start_log_variable = [0.5] * self.k_variables

            # Setup the function the evaluate
            root_evaluated_variables = lambda log_variable: self.root_evaluated_variables(log_variable, log_variable)

            # Apply the root-finding algorithm
            solution = optimize.root(root_evaluated_variables, start_log_variable)
            
            return np.exp(solution.x)