# 1. Import the relevant packages
import numpy as np # Used to store the values in numpy array and making logarithic and exponential transformations
import pandas as pd # Uses this to update parameters
from scipy import optimize # Uses this to root maximize the steady state values
from ipywidgets import interact, FloatSlider # Uses this to make interactive plots
import matplotlib.pyplot as plt # Uses this to plot graphs

# 2. Update the standard parameters in matplotlib
plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"--"})
plt.rcParams.update({'font.size': 15})

# 3. We create a class that defines all of the variables, parameters, and equations for the RBC model, which is based on Chad Fulton (2015)
class RealBusinessCycleModelClass(object):
    """
    A class for the Real Business Cycle (RBC) model with log-linearized equations.
    """
    def __init__(self, params=None):
        """
        Initializes the Real Business Cycle model with the given empirically realistic parameters.

        Args: 
            params (tuple, optional): The tuple contains the parameters in the RBC mocel.
                discount_rate (float): The preference of the households for present consumption.
                disutility_from_labor (float): The marginal cost of working more for the households.
                depreciation_rate (float): How quickly the capital dimishes.
                capital_share (float): The share of output which is produced because of capital.
                technology (float): The production technologies which the firms have acces to.
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
                discount_rate (float): The preference of the households for present consumption.
                disutility_from_labor (float): The marginal cost of working more for the households.
                depreciation_rate (float): How quickly the capital dimishes.
                capital_share (float): The share of output which is produced because of capital.
                technology (float): The production technologies which the firms have acces to.
        """
        # i. The first element in the tuple is the discount rate
        self.discount_rate = params[0]
        # ii. The second element is the disutility from labor
        self.disutility_from_labor = params[1]
        # iii. The third element is the depreciation rate
        self.depreciation_rate = params[2]
        # iv. The fourth element is the capital share
        self.capital_share = params[3]
        # v. The fifth element is technology
        self.technology = params[4]
    
    # b. Define the root-evaluated variables for both period t and period t+1
    def root_evaluated_variables(self, next_period_log_variables, this_period_log_variables):
        """
        Calculates the variables which are root-evaluated for period t and period t+1.

        Args:
            next_period_log_variables (tuple): The tuple of the logged variables for period t+1.
                next_period_log_output (float): The logarithm of output in period t+1.
                next_period_log_consumption (float): The logarithm of consumption in period t+1.
                next_period_log_investment (float): The logarithm of investment in period t+1.
                next_period_log_labor (float): The logarithm of labor in period t+1.
                next_period_log_leisure (float): The logarithm of leisure in period t+1.
                next_period_log_capital (float): The logarithm of capital in period t+1.

            this_period_log_variables (tuple): The tuple of the logged variables for period t.
                this_period_log_output (float): The logarithm of output in period t.
                this_period_log_consumption (float): The logarithm of contumption in period t.
                this_period_log_investment (float): The logarithm of investment in period t.
                this_period_log_labor (float): The logarithm of labor in period t.
                this_period_log_leisure (float): The logarithm of leisure in period t.
                this_period_log_capital (float): The logarithm of capital in period t.

        Returns:
            (np.ndarray): A NumPy array which contains the variables which are root-evaluated.
        """
        # i. The root-evaluated variables for period t+1
        (next_period_log_output, next_period_log_consumption, next_period_log_investment,
         next_period_log_labor, next_period_log_leisure, next_period_log_capital) = next_period_log_variables
        
        # ii. The root-evaluated variables for period t
        (this_period_log_output, this_period_log_consumption, this_period_log_investment, this_period_log_labor,
         this_period_log_leisure, this_period_log_capital) = this_period_log_variables
        
        # iii. We return a NumPy array with the five equations in the model
        return np.r_[
            self.log_first_order_condition(next_period_log_consumption, next_period_log_labor, next_period_log_capital),
 
            self.log_euler_equation(next_period_log_consumption, next_period_log_labor, next_period_log_capital, 
                next_period_log_consumption),
  
            self.log_production_function(next_period_log_output, next_period_log_labor, next_period_log_capital),
      
            self.log_resource_constraint(
                next_period_log_output, next_period_log_consumption, next_period_log_investment),
           
            self.log_capital_accumulation(next_period_log_capital, next_period_log_investment, next_period_log_capital),
           
            self.log_labor_leisure_constraint(next_period_log_labor, next_period_log_leisure),
        ]
    
    # c. We define the first equation in the model, which is the FOC
    def log_first_order_condition(self, next_period_log_consumption, next_period_log_labor,
                       next_period_log_capital):
        """
        Computes the logarithm of the first-order condition equation. 

        Args: 
            next_period_log_consumption (float): The logarithm of consumption in the next period (t+1).
            next_period_log_labor (float): The logarithm of labor in the next period (t+1). 
            next_period_log_capital (float): The logarithm of capital in the next period (t+1).

        Returns:
            (float): The logarithm of the first-order condition equation.
        """
        return (np.log(self.disutility_from_labor) + next_period_log_consumption - np.log(1 - self.capital_share) -
                np.log(self.technology) - self.capital_share * (next_period_log_capital - next_period_log_labor))

    # d. We define the second equation in the RBC model, which is the consumption Euler equation
    def log_euler_equation(self, next_period_log_consumption, next_period_log_labor,
                            next_period_log_capital, this_period_log_consumption):
        """
        Calculates the logarithm of the consumption Euler equation.

        Args: 
            next_period_log_consumption (float): The logarithm of consumption in the next period (t+1).
            next_period_log_labor (float): The logarithm of labor in the next period (t+1).
            next_period_log_capital (float): The logarithm of capital in the next period (t+1).
            this_period_log_consumption (float): The logarithm of consumption in this period (t).

        Returns:
            (float): The value of the logged consumption Euler equation.
        """
        return (-this_period_log_consumption - np.log(self.discount_rate) + next_period_log_consumption -
                np.log(self.capital_share * np.exp(np.log(self.technology)) * np.exp((1 - self.capital_share) * 
                      (next_period_log_labor - next_period_log_capital)) + (1 - self.depreciation_rate)))

    # e. Define the third equation in the RBC model, which is the Cobb-Douglass production function
    def log_production_function(self, next_period_log_output, next_period_log_labor, next_period_log_capital):
        """
        Calculate the logarithm of the Cobb-Douglas production function.

        Args:
            next_period_log_output (float): The logarithm of output in the next period (t+1).
            next_period_log_labor (float): The logarithm of labor in the next period (t+1).
            next_period_log_capital (float): The logarithm of capital in the next period (t+1).
        
        Returns:
            (float): The value of the logged Cobb-Douglas production.
        """
        return (next_period_log_output - np.log(self.technology) - (self.capital_share * next_period_log_capital + (1 - self.capital_share) 
                * next_period_log_labor))

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
            (float): The value of the logged resource constraint equation.
        """
        return (next_period_log_output -
                np.log(np.exp(next_period_log_consumption) + np.exp(next_period_log_investment)))
    
    # g. Define the fifth equation in the RBC model, which is the capital accumulation
    def log_capital_accumulation(self, next_period_log_capital, this_period_log_investment, this_period_log_capital):
        """
        Calculates the logarithm of the capital accumulation equation.

        Args:
            next_period_log_capital (float): The logarithm of capital in the next period (t+1).
            this_period_log_investment (float): The logarithm of investment in this period (t).
            this_period_log_capital (float): The logarithm of capital in this period (t).

        Returns:
            (float): The value of the logged capital accumulation equation.
        """
        return (next_period_log_capital -
                np.log(np.exp(this_period_log_investment) + (1 - self.depreciation_rate) * np.exp(this_period_log_capital)))
    
    # h. Define the sixth equation in the RBC model, which is the labor-leisure constraint
    def log_labor_leisure_constraint(self, next_period_log_labor, next_period_log_leisure):
        """
        Calculate the logarithm of the labor-leisure constraint equation.

        Args: 
            next_period_log_labor (float): The logarithm of labor in the next period (t+1).
            next_period_log_leisure (float): The logarithm of leisure in the next period (t+1).
        
        Returns:
            (float): The value of the logged labor-leisure constraint.
        """
        return (-np.log(np.exp(next_period_log_labor) + np.exp(next_period_log_leisure)))

# 4. Next, we need to make a class that calculates the numerical solution to the RBC model, which is also based on Chad Fulton (2015)
class NumericalSolutionClass(RealBusinessCycleModelClass):
    """
    Class which calculates the numerical solution to the simple Real Business Cycle model.

    This class inherits from:
        RealBusinessCycleModelClass: This class defines the equations of the RBC model.
    """
    # a. Define the numeric solution for the steady-state values
    def steady_state_numeric(self):
            """
            Calculates the numerical solution for the steady-state values in the RBC model.
            This function starts from an initial guess for the logged variables, and applies a root-finding algorithm to solve all of the equation simultanously.
            
            Returns:
                (np.ndarray): The numerical solution of the RBC model in exponential form such that the values are not longer logged.
            """
            # i. Setup the starting parameters for the variables
            start_log_variable = [0.5] * self.k_variables

            # ii. Setup the function to finding the root
            root_evaluated_variables = lambda log_variable: self.root_evaluated_variables(log_variable, log_variable)

            # iii. Apply the root-finding algorithm
            solution = optimize.root(root_evaluated_variables, start_log_variable, method='hybr')
            
            # iv. Outputs the numerical solution to the Real Business Cycle Model, where we take the exponential to remove the logarithm of the values.
            return np.exp(solution.x)

# 5. Define the SteadyStatePlotClass that plot the steady state values.
class SteadyStatePlotClass:
    """
    Class which plots the steady state values of the variables in the simple Real Business Cycle model.
    """
    # a. Defining the variables and steady state
    def __init__(self, variables, steady_state_values):
        """
        Initializes the SteadyStatePlotClass with the variables and the steady state values.
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
class RBCModelInteractiveClass:
    """
    A class which creates an interactive plot for the steady state values with changing parameter values in the RBC model 

    Attributes:
        model (NumericalSolutionClass): This class numerically calculates the steady state values in the RBC model.
        variables (list): A list of the variable names in the model.
            Output (str): The amount which is produced by the firms.
            Consumption (str): The amount which is consumed by the households.
            Investment (str): The amount which is invested in new capital.
            Labor (str): How much of their time, households spend on labor.
            Leisure (str): How of their time, households spend on leisure.
            Capital (str): The capital stock in the firms.
    """
    # a. Define the initializer method used when instances are created in the notebook
    def __init__(self):
        """
        Initializes the RBCModelInteractiveClass with the parameters and variables.
        """
        # i. Define the parameters of the model [beta, disutility from labor, depreciation rate, capital share, technology] 
        self.model = NumericalSolutionClass(params=[0.9, 3, 0.1, 1/3, 1])
        # ii. Define the names of the variables in the model
        self.variables = ['Output', 'Consumption', 'Investment', 'Labor', 'Leisure', 'Capital']
    
    # b. Define a method that updates the values of the parameters
    def update_parameters_and_recompute(self, params):
        """
        Updates the parameters of the RBC model and computes the steady state values.

        Args:
            params (list): A list of the parameters of the model, which will be updated.
                discount_rate (float): The preference of the households for present consumption.
                disutility_from_labor (float): The marginal cost of working more for the households.
                depreciation_rate (float): How quickly the capital dimishes.
                capital_share (float): The share of output which is produced because of capital.
                technology (float): The production technologies which the firms have acces to.  
        
        Returns:
            (pd.DataFrame): A Pandas DataFrame which contains the steady state values.
        """
        # i. Update the parameters in the model
        self.model.update(params)
        # ii. Create a variable that stores the steady state values in a pandas dataframe
        steady_state_values = pd.DataFrame({'Steady state value': self.model.steady_state_numeric()
                                }, index=self.variables).round(2)
        # iii. Return the DataFrame with the steady state values
        return steady_state_values
    
    # c. Define a method that sets up the interactive plot
    def interactive_plot(self, discount_rate, disutility_from_labor, depreciation_rate, capital_share, technology):
        """
        Sets up and shows the interactive plot for the steady state values for changing parameter values of the RBC model.

        Args: params (list): A list which contains the parameters of the simple RBC model.
            discount_rate (float): The preference of the households for present consumption.
            disutility_from_labor (float): The marginal cost of working more for the households.
            depreciation_rate (float): How quickly the capital dimishes.
            capital_share (float): The share of output which is produced because of capital.
            technology (float): The production technologies which the firms have acces to.
        """
        # i. Define the five parameters
        params = [discount_rate, disutility_from_labor, depreciation_rate, capital_share, technology]
        # ii. Call the DataFrame with the steady state values
        steady_state_values = self.update_parameters_and_recompute(params)
        # iii. Call the SteadyStatePlotClass used to make the simple plot
        plot = SteadyStatePlotClass(variables=steady_state_values.index, steady_state_values=steady_state_values)
        # iv. Plot the simple plot from the SteadyStatePlotClass
        plot.simpleplot()
    
    # d. Define a method that plots the interactive plot
    def create_interactive_plot(self):
        """
        Creates and shows the interactive plot, where the parameter values can be adjusted with sliders.
        """
        # i. Make a slider for the discount rate parameter
        discount_rate_slider = FloatSlider(min=0.1, max=1.0, step=0.1, value=0.9, description='discount_rate')
        # ii. Make a slider for the marginal disutility from labor parameter
        disutility_from_labor_slider = FloatSlider(min=0.1, max=10.0, step=0.1, value=3, description='disutility_from_labor')
        # iii. Make a slider for the depreciation rate parameter
        depreciation_rate_slider = FloatSlider(min=0.01, max=0.5, step=0.01, value=0.1, description='depreciation_rate')
        # iv. Make a slider for the capital share parameter
        capital_share_slider = FloatSlider(min=0.1, max=0.9, step=0.1, value=1/3, description='capital_share')
        # v. Make a slider for the technology parameter
        technology_slider = FloatSlider(min=0.1, max=2.0, step=0.1, value=1, description='technology')
        
        # vi. Combine the sliders and plot the interactive plot
        interact(self.interactive_plot, 
                 discount_rate = discount_rate_slider,
                 disutility_from_labor = disutility_from_labor_slider,
                 depreciation_rate = depreciation_rate_slider,
                 capital_share = capital_share_slider,
                 technology = technology_slider) 

# 6. Replacing production function with CES function and adding substitution parameter rho
class RBCCESClass(object):
    """
    A class for the RBC model with a CES production function.

    Attributes:
        k_params (int): The number of parameters in the RBC model.
        k_variables (int): The number of variables in the RBC model.
        discount_rate (float): The preference of the households for present consumption.
        disutility_from_labor (float): The marginal cost of working more for the households.
        depreciation_rate (float): How quickly the capital dimishes.
        capital_share (float): The share of output which is produced because of capital.
        technology (float): The production technologies which the firms have acces to.
        rho (float): The substitution parameter.
    """
    # a. Define params and variables
    def __init__(self, params=None):
        """
        Initializes the RBCCESClass with the parameters of the RBC model with a CES production function.
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
        Updates the parameters of the RBC model with CES production function.

        Args:
            params (tuple): A tuple which contains the paramters of the RBC model with a CES production function.
                discount_rate (float): The preference of the households for present consumption.
                disutility_from_labor (float): The marginal cost of working more for the households.
                depreciation_rate (float): How quickly the capital dimishes.
                capital_share (float): The share of output which is produced because of capital.
                technology (float): The production technologies which the firms have acces to.
                rho (float): The substitution parameter.
        """
        # i. The first element in the tuple should be the discount rate, beta
        self.discount_rate = params[0]
        # ii. The second element is the disutility from labor, psi
        self.disutility_from_labor = params[1]
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
        Defines the root-evaluated variables for period t and period t+1.

        Args:
            next_period_log_variables (tuple): The tuple of the logged variables for period t+1.
                next_period_log_output (float): The logarithm of output in period t+1.
                next_period_log_consumption (float): The logarithm of consumption in period t+1.
                next_period_log_investment (float): The logarithm of investment in period t+1.
                next_period_log_labor (float): The logarithm of labor in period t+1.
                next_period_log_leisure (float): The logarithm of leisure in period t+1.
                next_period_log_capital (float): The logarithm of capital in period t+1.

            this_period_log_variables (tuple): The tuple of the logged variables for period t.
                this_period_log_output (float): The logarithm of output in period t.
                this_period_log_consumption (float): The logarithm of contumption in period t.
                this_period_log_investment (float): The logarithm of investment in period t.
                this_period_log_labor (float): The logarithm of labor in period t.
                this_period_log_leisure (float): The logarithm of leisure in period t.
                this_period_log_capital (float): The logarithm of capital in period t.

        Returns:
            (np.ndarray): A NumPy array which contains the evaluated the roots for the equations of the RBC model with a CES production function.
        """
        # i. The root-evaluated variables for period t
        (next_period_log_output, next_period_log_consumption, next_period_log_investment,
         next_period_log_labor, next_period_log_leisure, next_period_log_capital) = next_period_log_variables

        # ii. The root-evaluated variables for period t+1
        (this_period_log_output, this_period_log_consumption, this_period_log_investment, this_period_log_labor,
         this_period_log_leisure, this_period_log_capital) = this_period_log_variables

        # iii. We return a NumPy array with the five equations in the model
        return np.r_[
            self.log_first_order_condition(
                next_period_log_consumption, next_period_log_labor, next_period_log_capital),
            self.log_euler_equation(
                next_period_log_consumption, next_period_log_labor, 
                next_period_log_capital, next_period_log_consumption),
            self.log_ces_function(
                next_period_log_output, next_period_log_labor, next_period_log_capital),
            self.log_resource_constraint(
                next_period_log_output, next_period_log_consumption, next_period_log_investment),
            self.log_capital_accumulation(
                next_period_log_capital, next_period_log_investment, next_period_log_capital),
            self.log_labor_leisure_constraint(next_period_log_labor, next_period_log_leisure)]
    
    # c. We define the first equation in the model, which is the FOC
    def log_first_order_condition(self, next_period_log_consumption, next_period_log_labor, next_period_log_capital):
        """
        Defines the logged first order condition of the RBC model with a CES production function.

        Args:
            next_period_log_consumption (float): The logged consumption in period t+1.
            next_period_log_labor (float): The logged labor in period t+1.
            next_period_log_capital (float): The logged capital in period t+1.
        
        Returns:
            (float): The logged first-order condition equation which have been evaluated.
        """
        return (
            np.log(self.disutility_from_labor) + next_period_log_consumption - np.log(1 - self.capital_share) 
            - np.log(self.technology) - self.capital_share * (next_period_log_capital - next_period_log_labor))
    
    # d. We define the second equation in the RBC model, which is the consumption Euler equation
    def log_euler_equation(self, next_period_log_consumption, next_period_log_labor,
                            next_period_log_capital, this_period_log_consumption):
        """
        Defines the consumption Euler equation for the RBC model with a CES production function.
        
        Args: 
            next_period_log_consumption (float): The logged consumption in period t+1.
            next_period_log_labor (float): The logged labor in period t+1.
            next_period_log_capital (float): The logged capital in period t+1.
            this_period_log_consumption (float): The logged consumption period t.
        """
        return (-this_period_log_consumption - np.log(self.discount_rate) + next_period_log_consumption -
                np.log(self.capital_share * np.exp(np.log(self.technology)) * np.exp((1 - self.capital_share) * 
                    (next_period_log_labor - next_period_log_capital)) + (1 - self.depreciation_rate)))

    # e. Define the third equation for the extension of the RBC model, which is the CES production function
    def log_ces_function(self, next_period_log_output, next_period_log_labor, next_period_log_capital):
         """
        Defines the CES production function equation for the RBC model.

        Args:
            next_period_log_output (float): The logged output in period t+1.
            next_period_log_labor (float): The logged labor in period t+1.
            next_period_log_capital (float): The logged cpaital in period t+1.
        
        Returns:
            (float): The CES production function which have been evaluated.
         """
         return (next_period_log_output - np.log(self.technology) - np.log((self.capital_share * np.exp(self.rho * next_period_log_capital) +
            (1 - self.capital_share) * np.exp(self.rho * next_period_log_labor)) ** (1 / self.rho)))

    # f. Define the fourth equation in the RBC model, which is the resource constraint for the economy
    def log_resource_constraint(self, next_period_log_output, next_period_log_consumption,
                                          next_period_log_investment):
        """
        Defines the resource constraint equation for the RBC model with a CES production function.

        Args:
            next_period_log_output (float): Logged output in period t+1.
            next_period_log_labor (float): Logged labor in period t+1.
            next_period_log_capital (float): Logged capital in period t+1.

        Returns:
            (float): The CES production function which have been evaluated.
        """
        return (next_period_log_output -
                np.log(np.exp(next_period_log_consumption) + np.exp(next_period_log_investment)))

    # g. Define the fifth equation in the RBC model, which is the capital accumulation
    def log_capital_accumulation(self, next_period_log_capital, this_period_log_investment, this_period_log_capital):
        """
        Defines the logged capital accumulation equation for the RBC model with a CES production function.
        
        Args:
            next_period_log_capital (float): Logged capital in period t+1.
            this_period_log_investment (float): Logged investment in period t.
            this_period_log_capital (float): Logged capital in period t.
        
        Returns:
            (float): The logged capital accumulation equation which have been evaluated.
        """
        return (next_period_log_capital -
                np.log(np.exp(this_period_log_investment) + (1 - self.depreciation_rate) * np.exp(this_period_log_capital)))
    
    # h. Define the sixth equation in the RBC model, which is the labor-leisure constraint
    def log_labor_leisure_constraint(self, next_period_log_labor, next_period_log_leisure):
        """
        Defines the logged labor-leisure constraint equation for the RBC model with a CES production function.

        Args:
            next_period_log_labor (float): Logged labor in period t+1.
            next_period_log_leisure (float): Logged leisure in period t+1.

        Returns:
            (float): The logged labor-leisure constraint equation which have been evaluated.
        """
        return (-np.log(np.exp(next_period_log_labor) + np.exp(next_period_log_leisure)))

# 4. make a class that calculates the numerical solution to the RBC model
class NumericalSolutionCESClass(RBCCESClass):
    """
    A class which numerically calculates the steady state for a RBC model with a CES production function.

    This class inherits from:
        RBCCESClass: This class contains the equations for the RBC model with a CES production function. 
    """
    def steady_state_numeric(self):
            """
            Calculates the numerical steady state values for the RBC model with a CES production function.

            It does so by setting up the starting parameters for the variables defines a function.
            In this function, the roots should be evaluated, and then a root-finding algorithm is used.

            Returns:
                (np.ndarray): A Numpy array of steady state values for the variables.
            """
            # i. Setup the starting parameters for the variables
            start_log_variable = [0.5] * self.k_variables

            # ii. Setup the function to finding the root
            root_evaluated_variables = lambda log_variable: self.root_evaluated_variables(log_variable, log_variable)

            # iii. Apply the root-finding algorithm
            solution_to_rbc_model_ces = optimize.root(root_evaluated_variables, start_log_variable, method='hybr')

            return np.exp(solution_to_rbc_model_ces.x)

# 5. Create a class that plot the steady state in the RBC model with a CES production function
class SteadyStatePlotCESClass:
    """
    A class that plot the steady state values for the RBC model with a CES production function.

    Attributes:
        variables (list): A list of variable names.
            Output (str): The amount which is produced by the firms.
            Consumption (str): The amount which is consumed by the households.
            Investment (str): The amount which is invested in new capital.
            Labor (str): How much of their time, households spend on labor.
            Leisure (str): How of their time, households spend on leisure.
            Capital (str): The capital stock in the firms.
        steady_state_values (pd.DataFrame): A DataFrame that contains the steady state values.
    """
    # a. Define the variables and steady state
    def __init__(self, variables, steady_state_values):
        """
        Initialize the SteadyStatePlotCESClass with variables.

        Args:
            variables (list): A list of variable names.
                Output (str): The amount which is produced by the firms.
                Consumption (str): The amount which is consumed by the households.
                Investment (str): The amount which is invested in new capital.
                Labor (str): How much of their time, households spend on labor.
                Leisure (str): How of their time, households spend on leisure.
                Capital (str): The capital stock in the firms.
            steady_state_values (pd.DataFrame): A Pandas DataFrame which contains the steady state values.
        """
        self.variables = variables
        self.steady_state_values = steady_state_values

    # b. Define a method that plots the steady state values
    def simpleplot_ces(self):
        """
        Plots the steady state values for the RBC model with a CES production function.
        """
        # i. Sets the size of the figure and the bar diagram with inputs
        plt.figure(figsize=(10, 6))
        plt.bar(self.variables, self.steady_state_values['value'], color='skyblue')  # Access 'value' column
        # ii. Create a title and labels for the plot
        plt.title('Steady state values (CES)')
        plt.xlabel('Variables')
        plt.ylabel('Steady State values')

        # iii. Rotate the ticks and add grids to the plot
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--')

        # iv. Set the y-axis limit dynamically
        max_value = max(self.steady_state_values['value'])
        plt.ylim(0, max_value * 1.1)  # Adjust ylim to give some extra space

        # v. Show the plot
        plt.show()