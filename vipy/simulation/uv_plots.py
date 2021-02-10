import numpy as np
import matplotlib.pyplot as plt


def plot_baselines(baseline):
    """Takes simulated baselines and creates uv coverage plot.

    Parameters
    ----------
    baseline : dataclass object
        baseline object containing all baselines between individual telescopes
    """
    weight = np.array([b.valid for b in baseline])
    u = np.array([b.u for b in baseline])
    v = np.array([b.v for b in baseline])
    u = u[weight]
    v = v[weight]
    plt.plot(np.append(u, -u), np.append(v, -v), "x")
