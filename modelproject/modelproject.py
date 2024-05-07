
import numpy as np
from scipy import optimize
from types import SimpleNamespace
from scipy import optimize
import matplotlib.pyplot as plt
plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"--"})
plt.rcParams.update({'font.size': 14})
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import ipywidgets as widgets

class RealBusinessCycleModel(object):
    def __init__(self, params=None):
        # a. We define the number of parameters
        self.k_params = 4
        # b. Define the number of variables
        self.k_variables = 6
        # c. Check if the parameter is equal to none.
        if params is not None:
            # i. Update the instance with values from params
            self.update(params)
    
    # 2. Create a new class that updates the elements in the tuple
    def update(self, params):
        # The first element in the tuple should be the discount rate
        self.discount_rate = params[0]
        # The second element is the disutility from labour
        self.disutility_from_labour = params[1]
        # The third element is the depreciation rate
        self.depreciation_rate = params[2]
        # The fourth element is the capital share
        self.capital_share = params[3]
        # Fifth element is technology
        self.technology = params[4]
        
    def logged_variables(self, next_period_log_variables, this_period_log_variables):
        (next_period_log_output, next_period_log_consumption, next_period_log_investment,
         next_period_log_labour, next_period_log_leisure, next_period_log_capital) = next_period_log_variables
        
        (this_period_log_output, this_period_log_consumption, this_period_log_investment, this_period_log_labour,
         this_period_log_leisure, this_period_log_capital) = this_period_log_variables
        
        return np.r_[
            self.log_first_order_condition(
                next_period_log_consumption, next_period_log_labour,
                next_period_log_capital
            ),
            self.log_euler_equation(
                next_period_log_consumption, next_period_log_labour,
                next_period_log_capital, next_period_log_consumption
            ),
            self.log_production_function(
                next_period_log_output, next_period_log_labour, next_period_log_capital
            ),
            self.log_aggregate_resource_constraint(
                next_period_log_output, next_period_log_consumption,
                next_period_log_investment
            ),
            self.log_capital_accumulation(
                next_period_log_capital, next_period_log_investment, next_period_log_capital
            ),
            self.log_labor_leisure_constraint(
                next_period_log_labour, next_period_log_leisure
            ),
        ]
    
    def log_first_order_condition(self, next_period_log_consumption, next_period_log_labour,
                       next_period_log_capital):
        return (
            np.log(self.disutility_from_labour) +
            next_period_log_consumption -
            np.log(1 - self.capital_share) -
            self.technology -
            self.capital_share * (log_lead_capital - log_lead_labor)
        )
        
    def log_euler_equation(self, next_period_log_consumption, next_period_log_labour,
                            next_period_log_capital, next_period_log_consumption):
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
        
    def log_production_function(self, next_period_log_output, next_period_log_labour, next_period_log_capital):
        return (
            next_period_log_output - self.capital_share * next_period_log_capital -
            (1 - self.capital_share) * next_period_log_labour
        )
        
    def log_aggregate_resource_constraint(self, next_period_log_output, next_period_log_consumption,
                                          next_period_log_investment):
        return (
            next_period_log_output -
            np.log(np.exp(next_period_log_consumption) + np.exp(next_period_log_investment))
        )
    
    def log_capital_accumulation(self, next_period_log_capital, this_period_log_investment, this_period_log_capital):
        return (
            log_lead_capital -
            np.log(np.exp(this_period_log_investment) + (1 - self.depreciation_rate) * np.exp(this_period_log_capital))
        )
    
    def log_labour_leisure_constraint(self, next_period_log_labour, log_lead_leisure):
        return (
            -np.log(np.exp(next_period_log_labour) + np.exp(next_period_log_leisure))
        )
class NumericalSolution(RealBusinessCycleModel):
    def steady_state_numeric(self):
            # Setup starting parameters
            log_start_vars = [0.5] * self.k_variables  # very arbitrary

            # Setup the function the evaluate
            eval_logged = lambda log_vars: self.eval_logged(log_vars, log_vars)

            # Apply the root-finding algorithm
            result = optimize.root(eval_logged, log_start_vars)
            
            return np.exp(result.x)