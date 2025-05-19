import numpy as np

class AdjTM:
    def __init__(self, vector):
        self.index_map = {val: i for i, val in enumerate(set(vector))}
        self.TM = np.zeros((len(self.index_map), len(self.index_map)), dtype = int)
        for i in range(len(vector) - 1):
            m = self.index_map[vector[i]]
            n = self.index_map[vector[i + 1]]
            self.TM[m, n] += 1
            self.TM[n, m] += 1
        
    def add_vector(self, vector):
        old_length = len(self.index_map)
        for val in set(vector):
            if val not in self.index_map:
                self.index_map[val] = len(self.index_map)
        if old_length !=  len(self.index_map):
            newTM = np.zeros((len(self.index_map), len(self.index_map)), dtype=int)
            newTM[:self.TM.shape[0], :self.TM.shape[1]] = self.TM
            self.TM = newTM
        for i in range(len(vector) - 1):
            m = self.index_map[vector[i]]
            n = self.index_map[vector[i + 1]]
            self.TM[m, n] += 1
            self.TM[n, m] += 1
            
    @staticmethod
    def add_TM(adjTM0, adjTM1):
        adjTM = AdjTM([])
        keys = set(adjTM0.index_map) | set(adjTM1.index_map)
        index_map = {val: i for i, val in enumerate(keys)}
        TM = np.zeros((len(index_map), len(index_map)), dtype = int)
        removed = set()
        for key in keys:
            removed.add(key)
            m = index_map[key]
            for tm in (adjTM0, adjTM1):
                for val in tm.index_map:
                    if val not in removed:
                        n = index_map[val]
                        paths = tm.TM[tm.index_map[key], tm.index_map[val]]
                        TM[m, n] += paths
                        TM[n, m] += paths
        adjTM.index_map = index_map
        adjTM.TM = TM
        return adjTM
        