
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import numpy as np
from numpy import asarray
from numpy.random import randn
from numpy.random import rand
import matplotlib
from matplotlib import pyplot
import random



def objective(x):
    # r1 is equal to x[2], count as a variable
    f1 = 1/2 * x[0]**2 * x[1]
    f2 = x[0]**2 + 3 * x[0] * x[1]
    w2 = 1-W1
    # minmize
    return -np.abs(f1-x[2]) * W1 + w2 * np.abs(f2-x[3])

# black-box optimization software


def local_hillclimber(objective, bounds, n_iterations, step_size, init):
    # generate an initial point
    best = init
    # evaluate the initial point
    best_eval = objective(best)
    curr, curr_eval = best, best_eval  # current working solution
    scores = list()
    points = list()
    for i in range(n_iterations):  # take a step
        candidate = [curr[0] + rand()*step_size[0]-step_size[0]/2.0,
                     curr[1]+rand()*step_size[1]-step_size[1]/2.0,
                     curr[2] + rand()*step_size[2]-step_size[2]/2.0,
                     curr[3] + rand()*step_size[3]-step_size[3]/2.0,]
        points.append(candidate)
        print('>%d f(%s) = %.5f, %s' % (i, best, best_eval, candidate))
        # evaluate candidate point
        candidate_eval = objective(candidate)  # check for new best solution
        if candidate_eval < best_eval:  # store new best point
            best, best_eval = candidate, candidate_eval
            # keep track of scores
            scores. append(best_eval)
            # current best
            curr = candidate
    return [best, best_eval, points, scores]

# pyrameter weight
W1 = 0.5
bounds = asarray([[0, 5], [0, 10],[0,250],[0,250]])
step_size = [0.5, 0.5, 7,7]
n_iterations = 200
init = [2.4, 5.0,20.0,20.0]
best, score, points, scores, = local_hillclimber(
    objective, bounds, n_iterations, step_size, init)

# plot
n, m = 7, 7
start = -3
x_vals = np.arange(start, start+7, 0.1)
y_vals = np.arange(start, start+7, 0.1)
X, Y = np.meshgrid(x_vals, y_vals)
print(X)
print(Y)
fig = plt.figure(figsize=(6, 5))
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height])
Z = -W1 * (1/2 * X**2 * Y) + (1-W1) * (X**2 + 3 * X * Y)
cp = ax.contour(X, Y, Z)
ax.clabel(cp, inline=True, fontsize=10)
ax.set_title('Contour Plot')
ax.set_xlabel('x[0]')
ax.set_ylabel('x[1]')
for i in range(n_iterations):
    plt.plot(points[i][0], points[i][1], "o")
plt.savefig('test22.png')
# plt.show()
