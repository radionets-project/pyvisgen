import numpy as np
import matplotlib.pyplot as plt


def plot_baselines(baselines):
    """Takes simulated baselines and creates uv coverage plot.

    Parameters
    ----------
    baseline : dataclass object
        baseline object containing all baselines between individual telescopes
    """
    weight = np.array([b.valid for b in baselines])
    u = np.array([b.u for b in baselines])
    v = np.array([b.v for b in baselines])
    u = u[weight]
    v = v[weight]
    plt.plot(np.append(u, -u), np.append(v, -v), "x")
