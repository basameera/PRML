import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs

"""Reference : https://www.youtube.com/watch?v=kKpqGcCUFF8
"""

if __name__ == "__main__":
    np.random.seed(123)

    ITER = 1

    N = 100

    data = np.random.uniform(0, 1, (N, 2))

    fig, axs = plt.subplots(1, 2)

    # plot 1
    ax = axs[0]
    ax.set_axisbelow(True)

    ax.scatter(data.T[0], data.T[1])

    ax.set_xlabel('$n_x$')
    ax.set_ylabel('$n_y$')

    ax.grid()
    
    # plot 2
    num = abs(data[:, 0] - data[:, 1])
    den = abs(data[:, 0] + data[:, 1])
    delta_xy = num / den

    ax = axs[1]
    ax.plot(delta_xy)

    plt.show()
    plt.close()
