"""Visualization of Normal Distribution in different forms
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


if __name__ == "__main__":

    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    ax = axs
    ax.set_axisbelow(True)

    x = np.linspace(-4, 4, 100)
    pdf = norm.pdf(x)
    cdf = norm.cdf(x)

    ax.plot(x, pdf, 'k-', lw=1, label=r'PDF $\mathcal{N}\left(\mu=0, \sigma^{2}=1\right)$')
    ax.plot(x, cdf, label='CDF')

    ax.set_xlabel(r'X')
    ax.grid()
    ax.legend()

    
    

    plt.show()
