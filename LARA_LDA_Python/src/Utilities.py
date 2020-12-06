import numpy as np

def randomize(v):
    for i in range(len(v)):
        v[i] = (2 * np.random.rand() - 1) / 10


