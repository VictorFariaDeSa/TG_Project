import matplotlib.pyplot as plt
from IPython import display
import numpy as np

plt.ion()
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
ax1, ax2 = axes
fig.suptitle("Training...", fontsize=16, fontweight="bold")
def plot(scores,last_scores,mean_scores,mse=None):
    display.clear_output(wait=True)
    ax1.clear()
    ax2.clear()
    ax1.set_title("Scores")
    ax1.set_xlabel("Number of games")
    ax1.set_ylabel("Score")
    ax1.plot(scores,alpha = 0.5, color = "grey")
    ax1.plot(mean_scores, color=(50/255, 250/255, 3/255))
    ax1.grid(True, which='both', linestyle='--', linewidth=1, alpha=0.7)
    if mse:
        ax2.plot(mse)
        ax2.text(len(mse)-1,mse[-1],f"{mse[-1]:.2f}")
        ax2.set_title("Mean MSE Critic network error")
        ax2.set_xlabel("Number of games")
        ax2.set_ylabel("MSE error") 
    ax1.text(len(scores)-1,scores[-1],f"{scores[-1]:.2f}")

    ax1.text(len(mean_scores)-1,mean_scores[-1],f"{mean_scores[-1]:.2f}")
    dt_mean = max(mean_scores)-min(mean_scores)
    dt_last = max(last_scores) - min(last_scores)
    lower_limit = min(min(mean_scores),min(last_scores))
    upper_limit = max(max(mean_scores),max(last_scores))
    delta = upper_limit - lower_limit
    # ax1.set_ylim([lower_limit-0.1*delta,upper_limit+0.1*delta])
    plt.show(block=False)
    plt.pause(.1)
