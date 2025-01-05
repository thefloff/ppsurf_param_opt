
import json
import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt 


def val_to_ind(possibilities, val):
    dist = 9999999999
    for i in range(len(possibilities)):
        if abs(possibilities[i] - val) > dist:
            return i-1
        else:
            dist = abs(possibilities[i] - val)
    return len(possibilities) - 1

def vis_line(results, par_name, possibilities):
    counter = np.zeros((len(possibilities), len(possibilities)))
    appearances = np.zeros(len(possibilities))
    for result in results:
        should = result['gt'][par_name]
        pred = result['pred'][par_name]
        counter[val_to_ind(possibilities, should), val_to_ind(possibilities, pred)] += 1
        appearances[val_to_ind(possibilities, should)] += 1

    plt.figure(figsize=(6,6))
    for i in range(len(possibilities)):
        if appearances[i] > 0:
            plt.plot(possibilities, counter[i, :], label = str(round(possibilities[i], 2)) + " (" + str(int(appearances[i])) + ")")
    plt.legend()
    plt.title(par_name)
    plt.xticks(possibilities)
    plt.show()

def vis_scatter(results, par_name, possibilities):
    should = []
    pred = []
    for result in results:
        should.append(result['gt'][par_name])
        pred.append(result['pred'][par_name])

    dots, sizes = np.unique(np.c_[should, pred], return_counts=True, axis=0)
    sizes = sizes * 5

    ax_min = possibilities[0] - (possibilities[1] - possibilities[0])
    ax_max = possibilities[-1] + (possibilities[-1] - possibilities[-2])

    b, m = polyfit(dots[:,0], dots[:,1], 1)

    plt.figure(figsize=(6,6))
    plt.scatter(dots[:,0], dots[:,1], s=sizes)
    plt.axline(xy1=(0, b), slope=m, color='r', label=f'linear regression, slope = {m:.2f}')
    plt.legend()
    ax = plt.gca()
    ax.set_xlim([ax_min, ax_max])
    ax.set_ylim([ax_min, ax_max])
    plt.title(par_name)
    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction")
    plt.show()

def visualize(results):
    # vis_line(results, "depth", range(4, 9))
    # vis_line(results, "cgDepth", range(0, 2))
    # vis_line(results, "fullDepth", range(5, 9))
    # vis_line(results, "iters", range(6, 11))
    # vis_line(results, "pointWeight", np.arange(2, 9, 1))
    # vis_line(results, "samplesPerNode", np.arange(1, 10, 1))
    # vis_line(results, "scale", np.arange(0.9, 1.8, 0.1))
    vis_scatter(results, "depth", range(4, 9))
    vis_scatter(results, "cgDepth", range(0, 2))
    vis_scatter(results, "fullDepth", range(5, 9))
    vis_scatter(results, "iters", range(6, 11))
    vis_scatter(results, "pointWeight", np.arange(2, 9, 1))
    vis_scatter(results, "samplesPerNode", np.arange(1, 10, 1))
    vis_scatter(results, "scale", np.arange(0.9, 1.8, 0.1))

    sum_e = 0
    for result in results:
        gt = result['gt']
        pred = result['pred']
        e = 0
        for key in ["depth", "cgDepth", "fullDepth", "iters", "pointWeight", "samplesPerNode", "scale"]:
            e += abs(gt[key] - pred[key]) * abs(gt[key] - pred[key])
        sum_e += e / 7
    sum_e /= len(results)
    print("Average MSE = " + str(sum_e))


if __name__ == "__main__":
    with open('all_results.json', 'r') as openfile:
        visualize(json.load(openfile))