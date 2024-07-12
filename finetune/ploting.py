
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

def live_plot(data1, data2, data3, data4, figsize=(10, 20), title1='Train loss on whole sequence (EMA)', title2='Train loss on completion (EMA)',
             title3='Val loss on whole sequence (EMA)', title4='Val loss on completion (EMA)'):
    clear_output(wait=True)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=figsize)  # Creates 4 subplots vertically aligned

    # Plotting data1 on the first subplot
    ax1.plot(data1, marker='o')
    ax1.set_title(title1)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.grid(True)

    # Plotting data2 on the second subplot
    ax2.plot(data2, marker='o')
    ax2.set_title(title2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.grid(True)

    # Plotting data1 on the first subplot
    ax3.plot(data3, marker='o')
    ax3.set_title(title3)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Loss')
    ax3.grid(True)

    # Plotting data2 on the second subplot
    ax4.plot(data4, marker='o')
    ax4.set_title(title4)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Loss')
    ax4.grid(True)

    plt.tight_layout()
    plt.show()


