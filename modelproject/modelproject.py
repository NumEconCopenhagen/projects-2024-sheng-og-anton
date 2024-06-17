# 1. Import packages
import numpy as np # Used to store the values in numpy array
import pandas as pd #
from scipy import optimize # Used this to root maximize the steady state values
from types import SimpleNamespace
from scipy.optimize import fsolve
from ipywidgets import interact, FloatSlider
import matplotlib.pyplot as plt # Use this to plot graphs

# 2. Update the standard parameters in matplotlib
plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"--"})
plt.rcParams.update({'font.size': 15})

# 3. We create a class that defines all of the variables, parameters, and equations for the RBC model
class RealBusinessCycleModel(object):
    """
    A class for the Real Business Cycle (RBC) model with log-linearized equations.
    """
    def __init__(self, params=None):
        """
        Initializes the Real Business Cycle model with the given realistic parameters.

        Args: 
            params (tuple, optional): The tuple contains the parameters in the RBC mocel.
                discount_rate (float):
                disutility_from_labour (float):
                depreciation_rate (float):
                capital_share (float):  
                technology (float): 

        Returns:
            None     
        """
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
        """
        Updates the model parameters.

        Args:
            params (tuple): The tuple contains the parameters in the RBC model.
                discount_rate (float):
                disutility_from_labour (float):
                depreciation_rate (float):
                capital_share (float):
                technology (float): 

        Returns:
            None

        """
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
        """
        Calculates the variables which are root-evaluated for period t and period t+1.

        Args:
            next_period_log_variables (tuple): The tuple of the logged variables for period t+1.
                next_period_log_output (float):
                next_period_log_consumption (float):
                next_period_log_investment (float):
                next_period_log_labour (float):
                next_period_log_leisure (float):
                next_period_log_capital (float): 
            this_period_log_variables (tuple): The tuple of the logged variables for period t.
                this_period_log_output (float):
                this_period_log_consumption (float):
                this_period_log_investment (float):
                this_period_log_labour (float):
                this_period_log_leisure (float):
                this_period_log_capital (float): 
        Returns:
            np.ndarray: A NumPy array which contains the variables which are root-evaluated.
        """
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
        """
        Computes the logarithm of the first-order condition equation. 

        Args: 
            next_period_log_consumption (float): The logarithm of consumption in the next period (t+1).
            next_period_log_labour (float): The logarithm of labour in the next period (t+1). 
            next_period_log_capital (float): The logarithm of capital in the next period (t+1).

        Returns:
            float: The logarithm of the first-order condition equation.
        """
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
        """
        Calculates the logarithm of the consumption Euler equation.

        Args: 
            next_period_log_consumption (float): The logarithm of consumption in the next period (t+1).
            next_period_log_labour (float): The logarithm of labour in the next period (t+1).
            next_period_log_capital (float): The logarithm of capital in the next period (t+1).
            this_period_log_consumption (float): The logarithm of consumption in this period (t).
        Returns:
            float: The value of the logged consumption Euler equation.
            
        """
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
        """
        Calculate the logarithm of the Cobb-Douglas production function.

        Args:
            next_period_log_output (float): The logarithm of output in the next period (t+1).
            next_period_log_labour (float): The logarithm of labour in the next period (t+1).
            next_period_log_capital (float): The logarithm of capital in the next period (t+1).
        
        Returns:
            float: The value of the logged Cobb-Douglas production.
        """
        return ( next_period_log_output - (self.capital_share * next_period_log_capital + (1 - self.capital_share) 
         * next_period_log_labour)
         )
    
    # f. Define the fourth equation in the RBC model, which is the resource constraint for the economy
    def log_resource_constraint(self, next_period_log_output, next_period_log_consumption,
                                next_period_log_investment):
        """
        Calculate the logarithm of the resource constraint equation.
        
        Args:
            next_period_log_output (float): The logarithm of output in the next period (t+1).
            next_period_log_consumption (float): The logarithm of consumption in the next period (t+1).
            next_period_log_investment (float): The logarithm of investment in the next period (t+1).
        
        Returns:
            float: The value of the logged resource constraint equation.
        """
        return (
            next_period_log_output -
            np.log(np.exp(next_period_log_consumption) + np.exp(next_period_log_investment))
        )
    
    # g. Define the fifth equation in the RBC model, which is the capital accumulation
    def log_capital_accumulation(self, next_period_log_capital, this_period_log_investment, this_period_log_capital):
        """
        Calculates the logarithm of the capital accumulation equation.

        Args:
            next_period_log_capital (float): The logarithm of capital in the next period (t+1).
            this_period_log_investment (float): The logarithm of investment in this period (t).
            this_period_log_capital (float): The logarithm of capital in this period (t).

        Returns:
            float: The value of the logged capital accumulation equation.
        """
        return (
            next_period_log_capital -
            np.log(np.exp(this_period_log_investment) + (1 - self.depreciation_rate) * np.exp(this_period_log_capital))
        )
    
    # h. Define the sixth equation in the RBC model, which is the labour-leisure constraint
    def log_labour_leisure_constraint(self, next_period_log_labour, next_period_log_leisure):
        """
        Calculate the logarithm of the labour-leisure constraint equation.

        Args: 
            next_period_log_labour (float): The logarithm of labour in the next period (t+1).
            next_period_log_leisure (float): The logarithm of leisure in the next period (t+1).
        
        Returns:
            float: The value of the logged labour-leisure constraint.
        """
        return (
            -np.log(np.exp(next_period_log_labour) + np.exp(next_period_log_leisure))
        )

# 4. Next, we need to make a class that calculates the numerical solution to the RBC model
class NumericalSolution(RealBusinessCycleModel):
    """
    Class which calculates the numerical solution to the Real Business Cycle model.

    Inherits from:
        RealBusinessCycleModel: This class defines the equations of the RBC model.
    """
    # a. Define the numeric solution for the steady-state values
    def steady_state_numeric(self):
            """
            Calculates the numerical solution for the steady-state values in the RBC model.
            This function starts from an initial guess for the logged variables, and applies a root-finding algorithm to solve all of the equation simultanously.
            
            Returns:
                np.ndarray: The numerical solution of the RBC model in exponential form such that the values are not longer logged.

            """
            # i. Setup the starting parameters for the variables
            start_log_variable = [0.5] * self.k_variables

            # ii. Setup the function to finding the root
            root_evaluated_variables = lambda log_variable: self.root_evaluated_variables(log_variable, log_variable)

            # iii. Apply the root-finding algorithm
            solution = optimize.root(root_evaluated_variables, start_log_variable)
            
            # iv. Outputs the numerical solution to the Real Business Cycle Model, where we take the exponential to remove the logarithm of the values.
            return np.exp(solution.x)

# 5. Define a class that plot the steady state values.
class SteadyStatePlot:
    """
    Class which plots the steady state values of the variables in the simple Real Business Cycle model.
    
    """
    # a. Defining the variables and steady state
    def __init__(self, variables, steady_state_values):
        """
        Initializes the SteadyStatePlot class with the variables and the steady state values.


        """
        self.variables = variables
        self.steady_state_values = steady_state_values
        
    # b. Define a method that plots the steady state values
    def simpleplot(self):
        """
        Plots the steady state values of the variables in a bar chart with one bar for each variable. 

        """
        # i. Sets the size of the figure and sets the bar diagram with the variables as input
        plt.figure(figsize=(10, 6))
        plt.bar(self.variables, self.steady_state_values['Steady state value'], color='skyblue')

        # ii. Create a title and labels for plot
        plt.title('Steady state')
        plt.xlabel('Variables')
        plt.ylabel('Steady State values')

        # iii. Rotates the ticks on the x-axis such that they do not overlap and add grids
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--')

        # iv. Set the y-axis limit dynamically
        max_value = max(self.steady_state_values['Steady state value'])
        plt.ylim(0, max_value * 1.1)  # Adjust ylim to give some extra space

        # v. Shows the plot
        plt.show()
        
# 6. Define the Interactive Model class
class RBCModelInteractive:
    """


    """
    # a. Define the initializer method used when instances are created in the notebook
    def __init__(self):
        """


        """
        # i. Define the parameters of the model [beta, disutility from labour, depreciation rate, capital share, technology] 
        self.model = NumericalSolution(params=[0.9, 3, 0.1, 1/3, 1])
        # ii. Define the names of the variables in the model
        self.variables = ['Output', 'Consumption', 'Investment', 'Labour', 'Leisure', 'Capital']
    
    # b. Define a method that updates the values of the parameters
    def update_parameters_and_recompute(self, params):
        """


        """
        # i. Update the parameters in the model
        self.model.update(params)
        # ii. Create a variable that stores the steady state values in a pandas dataframe
        steady_state_values = pd.DataFrame({
            'Steady state value': self.model.steady_state_numeric()
        }, index=self.variables).round(2)
        # iii. Return the DataFrame with the steady state values
        return steady_state_values
    
    # c. Define a method that sets up the interactive plot
    def interactive_plot(self, discount_rate, disutility_from_labour, depreciation_rate, capital_share, technology):
        """


        """
        # i. Define the five parameters
        params = [discount_rate, disutility_from_labour, depreciation_rate, capital_share, technology]
        # ii. Call the DataFrame with the steady state values
        steady_state_values = self.update_parameters_and_recompute(params)
        # iii. Call the SteadyStatePlot class used to make the simple plot
        plot = SteadyStatePlot(variables=steady_state_values.index, steady_state_values=steady_state_values)
        # iv. Plot the simple plot from the SteadyStatePlot class
        plot.simpleplot()
    
    # d. Define a method that plots the interactive plot
    def create_interactive_plot(self):
        """


        """
        # i. Make a slider for the discount rate parameter
        discount_rate_slider = FloatSlider(min=0.1, max=1.0, step=0.1, value=0.9, description='discount_rate')
        # ii. Make a slider for the marginal disutility from labour parameter
        disutility_from_labour_slider = FloatSlider(min=0.1, max=10.0, step=0.1, value=3, description='disutility_from_labour')
        # iii. Make a slider for the depreciation rate parameter
        depreciation_rate_slider = FloatSlider(min=0.01, max=0.5, step=0.01, value=0.1, description='depreciation_rate')
        # iv. Make a slider for the capital share parameter
        capital_share_slider = FloatSlider(min=0.1, max=0.9, step=0.1, value=1/3, description='capital_share')
        # v. Make a slider for the technology parameter
        technology_slider = FloatSlider(min=0.1, max=2.0, step=0.1, value=1, description='technology')
        
        # vi. Combine the sliders and plot the interactive plot
        interact(self.interactive_plot, 
                 discount_rate=discount_rate_slider,
                 disutility_from_labour=disutility_from_labour_slider,
                 depreciation_rate=depreciation_rate_slider,
                 capital_share=capital_share_slider,
                 technology=technology_slider) 
    

# 6. Replacing production function with CES function and adding parameter rho
# Define the RBC_CES class
class RBC_CES(object):
    """


    """
    # a. Define params and variables
    def __init__(self, params=None):
        """


        """
        # i. We define the number of parameters
        self.k_params = 5
        # ii. Define the number of variables
        self.k_variables = 6
        # iii. Checks if the parameter is equal to none
        if params is not None:
            self.update(params)
    
    
    # a. Create a new class that updates the elements in the tuple
    def update(self, params):
        """


        """
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
        # vi. The sixth element is the substitution parameter, rho
        self.rho = params[5]
   
    # b. Define the root-evaluated variables for both period t and period t+1
    def root_evaluated_variables(self, next_period_log_variables, this_period_log_variables):
        """


        """
        # i. The root-evaluated variables for period t
        (next_period_log_output, next_period_log_consumption, next_period_log_investment,
         next_period_log_labour, next_period_log_leisure, next_period_log_capital) = next_period_log_variables
        
        # ii. The root-evaluated variables for period t+1
        (this_period_log_output, this_period_log_consumption, this_period_log_investment, this_period_log_labour,
         this_period_log_leisure, this_period_log_capital) = this_period_log_variables
        
        # iii. We return a NumPy array with the five equations in the model
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
    
    # c. We define the first equation in the model, which is the FOC
    def log_first_order_condition(self, next_period_log_consumption, next_period_log_labour,
                       next_period_log_capital):
        """


        """
        return (
            np.log(self.disutility_from_labour) +
            next_period_log_consumption -
            np.log(1 - self.capital_share) -
            self.technology -
            self.capital_share * (next_period_log_capital - next_period_log_labour)
        )
    
    # d. We define the second equation in the RBC model, which is the consumption Euler equation
    def log_euler_equation(self, next_period_log_consumption, next_period_log_labour,
                            next_period_log_capital, this_period_log_consumption):
        """

        
        """
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
    
    # e. Define the third equation for the extension of the RBC model, which is the CES production function
    def log_ces_function(self, next_period_log_output, next_period_log_labour, next_period_log_capital):
         """


         """
         return ( next_period_log_output - np.log(
            (self.capital_share * np.exp(self.rho * next_period_log_capital) +
            (1 - self.capital_share) * np.exp(self.rho * next_period_log_labour)) ** (1 / self.rho)
        )
         )

    # f. Define the fourth equation in the RBC model, which is the resource constraint for the economy
    def log_resource_constraint(self, next_period_log_output, next_period_log_consumption,
                                          next_period_log_investment):
        """


        """
        return (
            next_period_log_output -
            np.log(np.exp(next_period_log_consumption) + np.exp(next_period_log_investment))
        )
    
    # g. Define the fifth equation in the RBC model, which is the capital accumulation
    def log_capital_accumulation(self, next_period_log_capital, this_period_log_investment, this_period_log_capital):
        """

        
        """
        return (
            next_period_log_capital -
            np.log(np.exp(this_period_log_investment) + (1 - self.depreciation_rate) * np.exp(this_period_log_capital))
        )
    
    # h. Define the sixth equation in the RBC model, which is the labour-leisure constraint
    def log_labour_leisure_constraint(self, next_period_log_labour, next_period_log_leisure):
        """


        """
        return (
            -np.log(np.exp(next_period_log_labour) + np.exp(next_period_log_leisure))
        )


# 4. make a class that calculates the numerical solution to the RBC model
class NumericalSolutionCES(RBC_CES):
    """


    """
    def steady_state_numeric(self):
            """



            """
            # i. Setup the starting parameters for the variables
            start_log_variable = [0.5] * self.k_variables

            # ii. Setup the function to finding the root
            root_evaluated_variables = lambda log_variable: self.root_evaluated_variables(log_variable, log_variable)

            # iii. Apply the root-finding algorithm
            solution = optimize.root(root_evaluated_variables, start_log_variable)
            
            return np.exp(solution.x)

# 5. 
class SteadyStatePlotCES:
    """



    """
    # a. Define the variables and steady state
    def __init__(self, variables, steady_state_values):
        """


        """
        self.variables = variables
        self.steady_state_values = steady_state_values
    # b. Define a method that plots the steady state values
    def simpleplot_ces(self):
        """


        """
        # i. Sets the size of the figure and the bar diagram with inputs
        plt.figure(figsize=(10, 6))
        plt.bar(self.variables, self.steady_state_values['value'], color='skyblue')  # Access 'value' column
        # ii. Create a title and labels for the plot
        plt.title('Steady state values (CES)')
        plt.xlabel('Variables')
        plt.ylabel('Steady State values')

        # iii. Rotate the ticks and add grids
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--')

        # iv. Set the y-axis limit dynamically
        max_value = max(self.steady_state_values['value'])
        plt.ylim(0, max_value * 1.1)  # Adjust ylim to give some extra space
        # v. Show the plot
        plt.show()