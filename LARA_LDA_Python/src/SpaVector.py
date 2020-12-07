import numpy as np

class SpaVector:
    def __init__(self, spa_index, spa_value):
        self.m_index = spa_index    # index must start from 1 (word index)
        self.m_value = spa_value    # word count

    def L1Norm(self):
        _sum = 0
        for v in self.m_value:
            _sum += np.abs(v)
        return _sum

    def normalize(self, norm):
        for i in range(len(self.m_value)):
            self.m_value[i] /= norm

    def get_length(self):
        i = len(self.m_index)
        return self.m_index[i-1]

    def dot_product(self, weight):
        _sum = weight[0]
        for i in range(len(self.m_index)):
            _sum += self.m_value[i] * weight[self.m_index[i]]
        return _sum

    def dot_product_with_offset(self, weight, offset):
        _sum = weight[offset]
        for j in range(len(self.m_index)):
            _sum += self.m_value[j] * weight[self.m_index[j]]  # index starts from one
        return _sum

