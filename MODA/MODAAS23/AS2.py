# # Basics of desdeo-emo
from numpy import random
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px
import plotly.graph_objects as go

import numpy as np
import pandas as pd

from desdeo_problem import variable_builder, ScalarObjective, MOProblem
from desdeo_emo.EAs import NSGAIII

from sklearn.preprocessing import MinMaxScaler

import plotly.express as px
import plotly.graph_objects as go

np.random.seed(0)

# Define Objective function


def f_1(x):
    #  surface to be minimized
    # r1 is long one
    r1 = x[:, 0]
    r2 = x[:, 1]
    h = x[:, 2]
    perimeter = np.pi * \
        (3 * (r1 + r2) - np.sqrt((3 * r1 + r2) * (3 * r2 + r1)))
    area_cylinder = 2 * np.pi * r1 * r2 + perimeter * h

    return area_cylinder


def f_2(x):
    # volumn to be maxmized
    r1 = x[:, 0]
    r2 = x[:, 1]
    h = x[:, 2]
    volume_cylinder = (np.pi) * r1 * r2 * h
    return -volume_cylinder

# Note that the expected input `x` is two dimensional. It should be a 2-D numpy array.


list_vars = variable_builder(['x1', 'x2', 'y'],
                             initial_values=[0, 0, 0],
                             lower_bounds=[0, 0, 0],
                             upper_bounds=[10, 10, 10])


f1 = ScalarObjective(name='f1', evaluator=f_1)
f2 = ScalarObjective(name='f2', evaluator=f_2)
list_objs = [f1, f2]

problem = MOProblem(variables=list_vars, objectives=list_objs)

# ## Using the EAs
#
# Pass the problem object to the EA, pass parameters as arguments if required.
evolver = NSGAIII(problem,
                  n_iterations=10,
                  n_gen_per_iter=100,
                  population_size=100)

while evolver.continue_evolution():
    evolver.iterate()

# ## Visualization of optimized decision variables and objective values using Plotly

# individuals: decision variable vectors
# solutions: points in objective space that approximate Parto front
individuals, solutions, others = evolver.end()

# pd.DataFrame(individuals).to_csv("individuals.csv")
pd.DataFrame(solutions).to_csv("ParetoFront.csv")

# --------------------------plot-----------------------

# Add a random sample to the plot
X1 = random.rand(1000, 1)*(10.0)
X2 = random.rand(1000, 1)*(10.0)
Y = random.rand(1000, 1)*(10.0)

len_data = 1000+len(solutions)
labels = np.zeros(len_data)

# Scatterplot F1, F2

randomsample = np.hstack((X1, X2, Y))  # matrix 1000x2

for i in range(1000):
    F1randomsample = f_1(randomsample)
    F2randomsample = f_2(randomsample)

plt.scatter(F1randomsample, -F2randomsample)
plt.scatter(solutions[:, 0], -solutions[:, 1])
plt.savefig('F2.png')

s1 = solutions[:, 0]
F1 = np.concatenate((F1randomsample, s1))
F1 = F1/np.max(F1)

s2 = solutions[:, 1]
F2 = np.concatenate((F2randomsample, s2))
F2 = F2/np.max(-F2)

d1 = individuals[:, 0]
X1 = np.concatenate((X1.flatten(), d1))
X1 = X1/np.max(X1)

d2 = individuals[:, 1]
X2 = np.concatenate((X2.flatten(), d2))
X2 = X2/np.max(X2)

d3 = individuals[:, 2]
X3 = np.concatenate((Y.flatten(), d3))
X3 = X3/np.max(X3)

for i in range(1000):
    labels[i] = '0'
for i in range(1001, 1000+len(d2)):
    labels[i] = '1'

data = pd.read_csv(r'ParetoFront.csv', sep=',')
df = pd.DataFrame([labels, F1, -F2, X1, X2, X3],
                  ["Label", "F1", "F2", "X1", "X2", "X3"])
dft = df.transpose()
print(dft)

pd.plotting.parallel_coordinates(dft, 'Label', color=["lime", "tomato"])
plt.ylim(0, 1)
plt.savefig('unknow.png')

cols = ["Label", "F1", "F2", "X1", "X2", "X3"]

fig = px.parallel_coordinates(dft, color="Label", dimensions=cols,
                              title="Geomertrical Shape Pareto Parallel Coorinates Plot")

fig.show()
