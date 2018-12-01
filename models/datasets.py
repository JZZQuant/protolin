import numpy as np
from scipy import stats


def toric_knots(size=10000):
    points = np.random.uniform(low=-10, high=10, size=(size, 2))
    distance_from_origin = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    condition = np.where((np.floor(distance_from_origin) % 2 == 1) & (distance_from_origin <= 10))
    X = points[condition]
    y = np.apply_along_axis(lambda x: int((np.floor(np.sqrt(x[0] ** 2 + x[1] ** 2)) % 4 == 1)), 1, X)[np.newaxis].T
    X = stats.zscore(X)
    full = np.hstack((X, y))
    return full, X, y
