import matplotlib.pyplot as plt
from IPython import display
import numpy as np

plt.ion()
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
ax1, ax2 = axes
fig.suptitle("Training...", fontsize=16, fontweight="bold")
def plot(scores,mean_scores,mse=None):
    display.clear_output(wait=True)
    ax1.clear()
    ax2.clear()
    ax1.set_title("Scores")
    ax1.set_xlabel("Number of games")
    ax1.set_ylabel("Score")
    ax1.plot(scores)
    ax1.plot(mean_scores)
    if mse:
        ax2.plot(mse)
        ax2.text(len(mse)-1,mse[-1],f"{mse[-1]:.2f}")
        ax2.set_title("Mean MSE Critic network error")
        ax2.set_xlabel("Number of games")
        ax2.set_ylabel("MSE error") 
    ax1.text(len(scores)-1,scores[-1],f"{scores[-1]:.2f}")

    ax1.text(len(mean_scores)-1,mean_scores[-1],f"{mean_scores[-1]:.2f}")
    plt.show(block=False)
    plt.pause(.1)
