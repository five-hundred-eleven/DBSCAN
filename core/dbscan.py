import numpy as np
from collections import defaultdict


class DBSCAN:
    
    def __init__(self, eps=0.2, minpts=4, distfunc=(lambda p1, p2: np.linalg.norm(p1-p2))):
        # TODO behavior for eps="auto"
        assert eps == "auto" or type(eps) in (int, float)
        assert type(minpts) is int
        assert callable(distfunc)
        self.__eps = eps
        self.__distfunc = distfunc
        self.__minpts = minpts
    
    def fit(self, points):
        count = 0
        self.__labels = defaultdict(lambda: None)
        for P in points:
            # the point has already been processed in the inner loop
            if self.__getLabel(P) is not None:
                continue
            neighbors = self.__rangeQuery(points, P)
            if len(neighbors) < self.__minpts:
                self.__setLabel(P, -1) # point is noise
                continue
            
            # expand the cluster
            count += 1
            self.__setLabel(P, count)
            j = 0
            S = set(tuple(Q) for Q in neighbors)
            N = list(neighbors)
            while j < len(N):
                Q = N[j]
                j += 1
                if self.__getLabel(Q) == -1:
                    self.__setLabel(Q, count)
                if self.__getLabel(Q) is not None:
                    continue
                self.__setLabel(Q, count)
                neighbors = self.__rangeQuery(points, Q)
                if len(neighbors) >= self.__minpts:
                    for R in neighbors:
                        if tuple(R) in S:
                            continue
                        S.add(tuple(R))
                        N.append(R)
            
        self.labels_ = [self.__getLabel(P) for P in points]
        self.labels_ = np.array([P if P is not None else np.nan for P in self.labels_])
            
        return self
    
    def __getLabel(self, P):
        return self.__labels[tuple(P)]
    
    def __setLabel(self, P, val):
        self.__labels[tuple(P)] = val
    
    def __rangeQuery(self, points, P):
        neighbors = []
        for Q in points:
            if self.__distfunc(P, Q) <= self.__eps:
                neighbors.append(Q)
        return neighbors
