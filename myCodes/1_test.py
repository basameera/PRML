import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def create_toy_data(func, sample_size, std):
    x = np.linspace(0, 1, sample_size)
    t = func(x) + np.random.normal(scale=std, size=x.shape)
    return x, t


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor,
                      **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def get_correlated_dataset(n, dependency, mu, scale):
    latent = np.random.randn(n, 2)
    dependent = latent.dot(dependency)
    scaled = dependent * scale
    scaled_with_offset = scaled + mu
    # return x and y of the new, correlated dataset
    return scaled_with_offset[:, 0], scaled_with_offset[:, 1]


if __name__ == "__main__":
    # generate sine wave
    def func(x):
        return np.sin(2 * np.pi * x)

    x_sine = np.linspace(0, 1, 100)
    y_sine = func(x_sine)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    ax = axs[0]
    ax.set_axisbelow(True)
    ax.plot(x_sine, y_sine, c='b', label="$\sin(2\pi x)$")

    std = 1

    # std * 3
    scope = std * 3
    ax.fill_between(x_sine, y_sine - scope, y_sine + scope,
                    color="b", label="std * 3", alpha=0.1)

    # std * 2
    scope = std * 2
    ax.fill_between(x_sine, y_sine - scope, y_sine + scope,
                    color="g", label="std * 2", alpha=0.3)

    # std * 1
    scope = std * 1
    ax.fill_between(x_sine, y_sine - scope, y_sine + scope,
                    color="r", label="std.", alpha=0.5)

    x_sine_noise, y_sine_noise = create_toy_data(func, 50, std)
    ax.scatter(x_sine_noise, y_sine_noise, c='k',
               label="$\sin(2\pi x)$ w/std noise")

    ax.set_xlabel('$\pi$')
    ax.set_title('Sine wave with $\sigma=1$ of noise')

    ax.grid()
    ax.legend()

    # fig, ax = plt.subplots(1, 1)
    ax = axs[1]
    x = np.linspace(norm.ppf(0.0001),
                    norm.ppf(0.9999), 100)

    ax.plot(x, norm.pdf(x), 'k-', lw=2, label='Norm PDF')

    ax.axvline(x=-1, c='r', linestyle='--', label='$\sigma$')
    ax.axvline(x=1, c='r', linestyle='--')

    ax.axvline(x=-2, c='g', linestyle='--', label='2$\sigma$')
    ax.axvline(x=2, c='g', linestyle='--')

    ax.axvline(x=-3, c='b', linestyle='--', label='3$\sigma$')
    ax.axvline(x=3, c='b', linestyle='--')

    ax.set_xlabel('$\sigma$')
    ax.grid()
    ax.legend()

    # fig, ax_nstd = plt.subplots(figsize=(6, 6))
    ax_nstd = axs[2]

    dependency_nstd = np.array([
        [0.8, 0.75],
        [-0.2, 0.35]
    ])
    mu = 0, 0
    scale = 8, 5

    ax_nstd.axvline(c='grey', lw=1)
    ax_nstd.axhline(c='grey', lw=1)

    x, y = get_correlated_dataset(500, dependency_nstd, mu, scale)
    ax_nstd.scatter(x, y, s=0.5)

    confidence_ellipse(x, y, ax_nstd, n_std=1,
                       label=r'$1\sigma$', edgecolor='firebrick')
    confidence_ellipse(x, y, ax_nstd, n_std=2,
                       label=r'$2\sigma$', edgecolor='fuchsia', linestyle='--')
    confidence_ellipse(x, y, ax_nstd, n_std=3,
                       label=r'$3\sigma$', edgecolor='blue', linestyle=':')

    ax_nstd.scatter(mu[0], mu[1], c='red', s=3)
    ax_nstd.set_title('Different standard deviations')
    ax_nstd.legend()
    plt.show()
