
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
        
    def logged_(self, next_period_log, this_period_log):
        (next_period_log_output, next_period_log_consumption, next_period_log_investment,
         log_lead_labor, log_lead_leisure, log_lead_capital) = next_period_log
        
        (log_output, log_consumption, log_investment, log_labor,
         log_leisure, log_capital) = this_period_log
        
        return np.r_[
            self.log_static_foc(
                next_period_log_consumption, next_period_log_labour,
                next_period_log_capital
            ),
            self.log_euler_equation(
                log_lead_consumption, next_period_labor,
                log_lead_capital, next_period_log_consumption
            ),
            self.log_production(
                log_lead_output, log_lead_labor, log_lead_capital
            ),
            self.log_aggregate_resource_constraint(
                log_lead_output, log_lead_consumption,
                log_lead_investment
            ),
            self.log_capital_accumulation(
                log_lead_capital, log_investment, log_capital
            ),
            self.log_labor_leisure_constraint(
                log_lead_labor, log_lead_leisure
            ),
        ]
    
    def log_static_foc(self, log_lead_consumption, log_lead_labor,
                       log_lead_capital):
        return (
            np.log(self.disutility_labor) +
            log_lead_consumption -
            np.log(1 - self.capital_share) -
            log_lead_technology_shock -
            self.capital_share * (log_lead_capital - log_lead_labor)
        )
        
    def log_euler_equation(self, log_lead_consumption, log_lead_labor,
                           log_lead_capital, log_consumption):
        return (
            -log_consumption -
            np.log(self.discount_rate) +
            log_lead_consumption -
            np.log(
                (self.capital_share * 
                 np.exp((1 - self.capital_share) * log_lead_labor) /
                 np.exp((1 - self.capital_share) * log_lead_capital)) +
                (1 - self.depreciation_rate)
            )
        )
        
    def log_production(self, log_lead_output, log_lead_labor, log_lead_capital):
        return (
            log_lead_output - self.capital_share * log_lead_capital -
            (1 - self.capital_share) * log_lead_labor
        )
        
    def log_aggregate_resource_constraint(self, log_lead_output, log_lead_consumption,
                                          log_lead_investment):
        return (
            log_lead_output -
            np.log(np.exp(log_lead_consumption) + np.exp(log_lead_investment))
        )
    
    def log_capital_accumulation(self, log_lead_capital, log_investment, log_capital):
        return (
            log_lead_capital -
            np.log(np.exp(log_investment) + (1 - self.depreciation_rate) * np.exp(log_capital))
        )
    
    def log_labor_leisure_constraint(self, log_lead_labor, log_lead_leisure):
        return (
            -np.log(np.exp(log_lead_labor) + np.exp(log_lead_leisure))
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