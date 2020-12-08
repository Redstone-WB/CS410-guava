import numpy as np


def randomize(v):
    for i in range(len(v)):
        v[i] = (2 * np.random.rand() - 1) / 10


def exp_sum(values):
    _sum = 0
    for v in values:
        _sum += np.exp(v)
    return _sum


def MSE(pred, answer, offset):
    mse = 0
    for i in range(len(pred)):
        mse += (pred[i]-answer[i+offset]) * (pred[i]-answer[i+offset])
    return mse/len(pred)


def correlation(pred, answer, offset):
    m_x = 0
    m_y = 0
    s_x = 0
    s_y = 0

    # first order moment
    for i in range(len(pred)):
        m_x += pred[i]
        m_y += answer[i+offset]

    m_x /= len(pred)
    m_y /= len(pred)

    # second order moment
    for i in range(len(pred)):
        s_x += (pred[i]-m_x) * (pred[i]-m_x)
        s_y += (answer[offset+i]-m_y) * (answer[offset+i]-m_y)

    # handle special cases
    if s_x == 0 and s_y == 0:
        return 1
    elif s_x == 0 or s_y == 0:
        return 0

    s_x = np.sqrt(s_x / (len(pred) - 1))
    s_y = np.sqrt(s_y / (len(pred) - 1))

    # Pearson correlation
    correlation = 0
    for i in range(len(pred)):
        correlation += (pred[i] - m_x) / s_x * (answer[offset + i] - m_y) / s_y

    return correlation / (len(pred) - 1.0)
