import numpy as np

class AdjTM:
    def __init__(self, vector):
        self.index_map = {val: i for i, val in enumerate(set(vector))}
        self.reverse_map = {i: val for val, i in self.index_map.items()}
        self.TM = np.zeros((len(self.index_map), len(self.index_map)), dtype = float)
        for i in range(len(vector) - 1):
            m = self.index_map[vector[i]]
            n = self.index_map[vector[i + 1]]
            self.TM[m, n] += 1
            self.TM[n, m] += 1
    
    @classmethod        
    def from_parts(cls, index_map, reverse_map, TM):
        adjTM = AdjTM([])
        adjTM.index_map = index_map 
        adjTM.reverse_map = reverse_map
        adjTM.TM = TM
        return adjTM
    
    @classmethod        
    def copy(cls, other):
        adjTM = AdjTM([])
        adjTM.index_map = other.index_map 
        adjTM.reverse_map = other.reverse_map
        adjTM.TM = other.TM
        return adjTM
            
    def __add__(self, other):
        keys = set(self.index_map) | set(other.index_map)
        index_map = {val: i for i, val in enumerate(keys)}
        reverse_map = {i: val for val, i in index_map.items()}
        TM = np.zeros((len(index_map), len(index_map)), dtype = float)
        for key in keys:
            m = index_map[key]
            for tm in (self, other):
                for val in tm.index_map:
                    if key in tm.index_map:
                        n = index_map[val]
                        TM[m, n] += tm.get_entry(key, val)
        return AdjTM.from_parts(index_map, reverse_map, TM)
    
    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return AdjTM.from_parts(self.index_map, self.reverse_map, self.TM * other)
        if isinstance(other, AdjTM):
            if list(other.index_map.keys()) == list(self.index_map.keys()):
                return AdjTM.from_parts(self.index_map, self.reverse_map, self.TM @ other.TM)
            else:
                raise ValueError("Arguments with different nodes")
        raise TypeError("Unrecognized argument type")
    
    def stochastic(self):
        col_sums = self.TM.sum(axis = 0, keepdims = True)
        return AdjTM.from_parts(self.index_map, self.reverse_map, self.TM/ col_sums)
            
    def get_entry(self, nodeA, nodeB):
        return self.TM[self.index_map[nodeA], self.index_map[nodeB]]
    
    def get_row(self, node):
        return self.TM[self.index_map[node], :]
    
    def get_column(self, node):
        return self.TM[:, self.index_map[node]]
    
    def get_node(self, index):
        return self.reverse_map[index]
        
    def add_vector(self, vector):
        old_length = len(self.index_map)
        for val in set(vector):
            if val not in self.index_map:
                self.reverse_map[len(self.index_map)] = val
                self.index_map[val] = len(self.index_map)
        if old_length != len(self.index_map):
            newTM = np.zeros((len(self.index_map), len(self.index_map)), dtype = float)
            newTM[:self.TM.shape[0], :self.TM.shape[1]] = self.TM
            self.TM = newTM
        for i in range(len(vector) - 1):
            m = self.index_map[vector[i]]
            n = self.index_map[vector[i + 1]]
            self.TM[m, n] += 1
            self.TM[n, m] += 1
    
    def remove_vector(self, vector):
        old_length = len(self.index_map)
        old_index_map = self.index_map.copy()
        for i in range(len(vector) - 1):
            try: 
                m = self.index_map[vector[i]]
                n = self.index_map[vector[i + 1]]
            except KeyError:
                continue
            self.TM[m, n] -= 1
            self.TM[n, m] -= 1
            if np.sum(self.TM[m, :]) == 0:
                del self.index_map[self.reverse_map[m]]
                del self.reverse_map[m]
            if np.sum(self.TM[:, n]) == 0:
                del self.index_map[self.reverse_map[n]]
                del self.reverse_map[n]
        if old_length != len(self.index_map):
            newTM = np.zeros((len(self.index_map), len(self.index_map)), dtype = int)
            for key, i in self.index_map.items():
                if i >= len(self.index_map):
                    self.index_map[key] = i + len(self.index_map) - old_length
                newTM[self.index_map[key], :] = self.TM[old_index_map[key], [old_index_map[k] for k in self.index_map]]
            self.TM = newTM