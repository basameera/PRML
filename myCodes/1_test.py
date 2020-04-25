import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # generate sine wave
    def func(x):
        return np.sin(2 * np.pi * x)

    x_sine = np.linspace(0, 1, 100)
    y_sine = func(x_sine)

    fig, ax = plt.subplots(1, 1)
    ax.set_axisbelow(True)
    ax.plot(x_sine, y_sine, c='b', label="$\sin(2\pi x)$")
    
    def create_toy_data(func, sample_size, std):
        x = np.linspace(0, 1, sample_size)
        t = func(x) + np.random.normal(scale=std, size=x.shape)
        return x, t


    std = 1

    # std * 3
    scope = std * 3
    ax.fill_between(x_sine, y_sine - scope, y_sine + scope, color="b", label="std * 3", alpha=0.1)

    # std * 2
    scope = std * 2
    ax.fill_between(x_sine, y_sine - scope, y_sine + scope, color="g", label="std * 2", alpha=0.3)

    # std * 1
    scope = std * 1
    ax.fill_between(x_sine, y_sine - scope, y_sine + scope, color="r", label="std.", alpha=0.5)

    x_sine_noise, y_sine_noise = create_toy_data(func, 50, std)


    ax.scatter(x_sine_noise, y_sine_noise, c='k', label="$\sin(2\pi x)$ w/std noise")

    plt.xlabel('$\pi$')
    plt.title('Sine wave with $\sigma=1$ of noise')


    plt.grid()
    plt.legend()
    plt.show()
