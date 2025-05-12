class AdjTM:
    def __init__(val1, val2, TM):
        self.adjacents = tuple(val1, val2)
        self.TM = set(TM)
        # left and right are lists of AdjTM objects
        self.adj0 = set()
        self.adj1 = set()
        
    def add_adj(self, aTM):
        if aTM in self.adj1 or aTM in self.adj2: # Checks if aTM object aleady added
            return False
        a0, a1 = self.adjacents
        b0, b1 = aTM.adjacents
        if {a0, a1} == {b0, b1}: # Checks if same adjacents as aTM
            self.TM.update(aTM.TM)
            del aTM
            return None
        adj_pairs = [
            (a0 == b0, self.adj0, aTM.adj0),
            (a0 == b1, self.adj0, aTM.adj1),
            (a1 == b0, self.adj1, aTM.adj0),
            (a1 == b1, self.adj1, aTM.adj1)
        ]
        for cond, s_adj, t_adj in adj_pairs:
            # Creates an edge if shares an adjacent with aTM
            if cond:
                s_adj.add(aTM)
                t_adj.add(self)
                return True
        return False