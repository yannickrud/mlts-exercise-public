#!/usr/bin/env python
# author: "Noah Maul"
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse


def get_ellipse_from_cov(cov, nstd):
    """
    Returns ellipse parameters width, height, angle for given covariance matrix and std.

    :param cov: Covariance matrix 2x2
    :type cov: np.ndarray
    :param nstd: Number of standard deviations
    :type nstd: int
    :return: (width, height, angle)
    """
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * nstd * np.sqrt(vals)
    return (w, h, theta)


def stepwise_update(num, data, lnMeasure, lnAlgo, esAlgo, config):
    def updateEllipses(cov, ellipses, x, y):
        w, h, theta = get_ellipse_from_cov(cov, 2)
        ellipses[idxFrame].center = (x, y)
        ellipses[idxFrame].width = w
        ellipses[idxFrame].height = h
        ellipses[idxFrame].angle = theta
        ellipses[idxFrame].set_visible(True)

    (dataMeasured, dataAlgo) = data
    isAlreadyPlotted = lambda line: len(line.get_xdata()) == config['numMeasurements']

    idxFrame = np.floor_divide(num, 2)

    if num == 0:
        lnMeasure.set_data([], [])
        lnAlgo.set_data([], [])
        for ellip in esAlgo:
            ellip.set_visible(False)
    if not isAlreadyPlotted(lnMeasure):
        if num % 2 == 0:
            lnMeasure.set_data(dataMeasured[:, 0][:idxFrame + 1], dataMeasured[:, 1][:idxFrame + 1])
            return [lnAlgo, lnMeasure] + esAlgo
    if not isAlreadyPlotted(lnAlgo):
        lnAlgo.set_data(dataAlgo[0][:, 0][:idxFrame + 1], dataAlgo[0][:, 1][:idxFrame + 1])
        updateEllipses(dataAlgo[1][idxFrame][:2, :2], esAlgo, dataAlgo[0][:, 0][idxFrame], dataAlgo[0][:, 1][idxFrame])
        return [lnAlgo, lnMeasure] + esAlgo
    else:
        return [lnMeasure, lnAlgo] + esAlgo


def setup_animations(data, config):
    def create_lines_filtered(trueMovement, ax):
        lnActual, = ax.plot(trueMovement[:, 0], trueMovement[:, 1], '#000000', label='true movement')
        lnMeasure, = ax.plot([], [], 'bo', animated=True, label='noisy observations')
        lnFiltered, = ax.plot([], [], 'r', marker='X', animated=True, label='filtered')
        return lnActual, lnMeasure, lnFiltered

    def create_lines_smoothed(trueMovement, ax):
        lnActual, = ax.plot(trueMovement[:, 0], trueMovement[:, 1], '#000000', label='true movement')
        lnMeasure, = ax.plot([], [], 'go', animated=True, label='noisy observations')
        lnSmoothed, = ax.plot([], [], 'b', marker='X', animated=True, label='smoothed')
        return lnActual, lnMeasure, lnSmoothed

    def create_ellipses(ax):
        esFiltered = [Ellipse(xy=(0, 0), width=0, height=0, visible=False, fill=False,
                              animated=True, edgecolor='r')
                      for i in range(config['numMeasurements'])]
        esSmoothed = [Ellipse(xy=(0, 0), width=0, height=0, visible=False, fill=False,
                              animated=True, edgecolor='b', label='variance of smoothing densities')
                      for i in range(config['numMeasurements'])]
        for ellip in esFiltered + esSmoothed:
            ax.add_artist(ellip)
        return esFiltered, esSmoothed

    (resFiltered, trueMovement, measurements) = data

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    img = plt.imread("./utils/Map.PNG")

    ax1.imshow(img, extent=[8, 32, -2, 22])
    ax1.set_xlim(8, 32)
    ax1.set_ylim(-2, 22)
    ax1.grid()
    ax1.set_xlabel("X-Position")
    ax1.set_ylabel("Y-Position")
    lnActual, lnMeasure, lnFiltered = create_lines_filtered(trueMovement, ax1)
    esFiltered, esSmoothed = create_ellipses(ax1)
    cov_filtering = Line2D([0], [0], marker='o', color='w', label='variance of filtering densities',
                           markeredgecolor='r', markerfacecolor='white', markersize=15)
    ax1.legend(handles=[lnActual, lnMeasure, lnFiltered, cov_filtering])
    ani = animation.FuncAnimation(fig, stepwise_update, frames=config['numMeasurements'] * 2,
                                  fargs=((measurements, (resFiltered[0], resFiltered[1])), lnMeasure,
                                         lnFiltered, esFiltered, config),
                                  blit=True, interval=config['msPerTimestep'], repeat_delay=5000.0, repeat=True)

    return ani
