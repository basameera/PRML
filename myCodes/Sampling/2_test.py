import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs

"""Reference : https://www.youtube.com/watch?v=kKpqGcCUFF8
"""

if __name__ == "__main__":
    np.random.seed(123)

    ITER = 1

    N = 20

    data = np.random.uniform(0, 1, (N, 2))

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))

    # plot 1
    ax = axs[0]
    ax.set_axisbelow(True)

    ax.scatter(data.T[0], data.T[1])

    ax.set_xlabel('$n_x$')
    ax.set_ylabel('$n_y$')
    ax.set_title('Uniform')

    ax.grid()

    # plot 2
    # N = 10
    data_lhs = lhs(2, samples=N)

    ticks = np.arange(N)/N + (1/N)

    ax = axs[1]
    ax.set_axisbelow(True)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.scatter(data_lhs.T[0], data_lhs.T[1])
    ax.grid(linestyle='--')
    ax.set_title('LHS')

    # plot 3
    # fig, axs = plt.subplots(1, 1)
    # num = abs(data[:, 0] - data[:, 1])
    # den = abs(data[:, 0] + data[:, 1])
    # delta_xy = num / den

    # ax = axs
    # ax.plot(delta_xy)

    plt.show()
    plt.close()
