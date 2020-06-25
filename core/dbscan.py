import numpy as np
from collections import defaultdict
from sklearn.neighbors import KDTree


class BaseDBSCAN:
    
    def __init__(self, eps=0.2, minpts=4, distfunc=(lambda p1, p2: np.linalg.norm(p1-p2))):
        assert type(eps) in (int, float)
        assert type(minpts) is int
        assert callable(distfunc)
        self._eps = eps
        self.__distfunc = distfunc
        self._minpts = minpts+1
    
    def fit(self, points):
        count = 0
        self.__labels = defaultdict(lambda: None)
        for P in points:
            # the point has already been processed in the inner loop
            if self.__getLabel(P) is not None:
                continue
            neighbors = self._rangeQuery(points, P)
            if len(neighbors) < self._minpts:
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
                Q_tuple = tuple(Q)
                j += 1
                if self.__getLabel(Q_tuple) == -1:
                    self.__setLabel(Q_tuple, count)
                if self.__getLabel(Q_tuple) is not None:
                    continue
                self.__setLabel(Q_tuple, count)
                neighbors = self._rangeQuery(points, Q)
                if len(neighbors) >= self._minpts:
                    for R in neighbors:
                        R_tuple = tuple(R)
                        if R_tuple in S:
                            continue
                        S.add(R_tuple)
                        N.append(R)
            
        self.labels_ = [self.__getLabel(P) for P in points]
        self.labels_ = np.array([P if P is not None else np.nan for P in self.labels_])
            
        return self
    
    def __getLabel(self, P):
        return self.__labels[tuple(P)]
    
    def __setLabel(self, P, val):
        self.__labels[tuple(P)] = val

class DBSCAN(BaseDBSCAN):
    def _rangeQuery(self, points, P):
        """
            Naive implementation of a rangescan function which calculates the distance between
            the point and every other point.
        """
        neighbors = []
        for Q in points:
            if self.__distfunc(P, Q) <= self._eps:
                neighbors.append(Q)
        return neighbors

class OptimizedDBSCAN(BaseDBSCAN):
    
    def __init__(self, eps=0.2, minpts=4):
        super().__init__(eps=eps, minpts=minpts, distfunc=lambda: None)
    
    def fit(self, points):
        self.__tree = KDTree(points)
        return super().fit(points)
    
    def _rangeQuery(self, points, P):
        """
            Optimized version of the rangeQuery function which uses a kd tree.
        """
        ind, = self.__tree.query_radius([P], self._eps)
        return points[ind]
