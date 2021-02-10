import numpy as np
import matplotlib.pyplot as plt


def plot_baselines(baselines):
    # plots uv-plane using array of baselines
    weight = np.array([b.valid for b in baselines])
    u = np.array([b.u for b in baselines])
    v = np.array([b.v for b in baselines])
    u = u[weight]
    v = v[weight]
    plt.plot(np.append(u, -u), np.append(v, -v), "x")
