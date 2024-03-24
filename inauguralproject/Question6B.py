# 1. We import the matplotlib
import matplotlib.pyplot as plt

# 2. We create the class
class UtilitarianSocialPlannerEdgeworthBox:
    # a. We define the values for w1bar and w2bar
    def __init__(self, w1bar=1.0, w2bar=1.0):
        self.w1bar = w1bar
        self.w2bar = w2bar

    # b. We plot the Edgeworth box
    def plot_edgeworth_box(self, xA1_optimal, xA2_optimal, xB1_optimal, xB2_optimal):
        fig = plt.figure(frameon=False, figsize=(6, 6), dpi=100)
        ax_A = fig.add_subplot(1, 1, 1)
        ax_A.set_xlabel("$x_1^A$") # Sets the label of the first x-axis
        ax_A.set_ylabel("$x_2^A$") # Sets the label of the first y-axis
        
        temp = ax_A.twinx()
        temp.set_ylabel("$x_2^B$") # Sets the label of the second y-axis
        
        ax_B = temp.twiny()
        ax_B.set_xlabel("$x_1^B$") # Sets the label of the second x-axis
        
        # i. We invert the second axes
        ax_B.invert_xaxis()
        ax_B.invert_yaxis()
        
        # ii. We create the scatter plots
        ax_A.scatter(xA1_optimal, xA2_optimal, marker='s', color='black', label='endowment')
        ax_A.scatter(xB1_optimal, xB2_optimal, marker='s', color='black')
        
        # iii. We plot the scatter plots
        ax_A.plot([0, self.w1bar], [0, 0], lw=2, color='black')
        ax_A.plot([0, self.w1bar], [self.w2bar, self.w2bar], lw=2, color='black')
        ax_A.plot([0, 0], [0, self.w2bar], lw=2, color='black')
        ax_A.plot([self.w1bar, self.w1bar], [0, self.w2bar], lw=2, color='black')
        
        # iv. We set the limits of the axes
        ax_A.set_xlim([-0.1, self.w1bar + 0.1])
        ax_A.set_ylim([-0.1, self.w2bar + 0.1])
        ax_B.set_xlim([self.w1bar + 0.1, -0.1])
        ax_B.set_ylim([self.w2bar + 0.1, -0.1])
        
        ax_A.legend(frameon=True, loc='upper right', bbox_to_anchor=(1.6, 1.0))
        plt.show() # Shows the figure
