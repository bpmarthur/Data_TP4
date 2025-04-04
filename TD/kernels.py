import numpy as np
from math import gamma
from typing import Self
from abc import abstractmethod
from sklearn.neighbors import NearestNeighbors

class Kernel:
    """A class for kernel density estimation, which also stores the cloud of points
    Attributes:
        d: int                 -- dimension of the ambient space
        data: list[np.ndarray] -- list of coordinates of the points (each of dimension self.d)
    """
    def __init__(self: Self, d: int, data: list[np.ndarray]):
        self.data = data
        self.d = d

    @abstractmethod
    def density(self: Self, x: np.ndarray) -> float:
        pass

    def norm(self: Self, x: np.ndarray, y: np.ndarray) -> float:
        return np.linalg.norm(x - y)**2 #Renvoie la distance au carré entre 2 points

class Radial(Kernel):
    def __init__(self: Self, d: int, data: list[np.ndarray], bandwidth: float):
        super().__init__(d, data)
        self.bandwidth = bandwidth

    @abstractmethod
    def volume(self: Self) -> float:
        pass

    @abstractmethod
    def profile(self: Self, t: float) -> float:
        pass

    def density(self: Self, x: np.ndarray) -> float:
        result = 0
        for point in self.data: #Itérator sur les points
            result += self.profile((np.linalg.norm(x - point) / self.bandwidth)**2)
        result /= (len(self.data) * self.volume() * (self.bandwidth**self.d))   #Normalisation
        return result
    
class Flat(Radial):
    def __init__(self: Self, d: int, data: list[np.ndarray], bandwidth: float):
        super().__init__(d, data, bandwidth)

    def volume(self: Self) -> float:
        return np.pi**(self.d/2) / gamma((self.d)/2 + 1) 

    def profile(self: Self, t: float) -> float:
        return 1 if t <= 1 else 0
    
class Gaussian(Radial):
    def __init__(self: Self, d: int, data: list[np.ndarray], bandwidth: float):
        super().__init__(d, data, bandwidth)

    def volume(self: Self) -> float:
        return (2 * np.pi)**(self.d/2)

    def profile(self: Self, t: float) -> float:
        return np.exp(-t / 2)
    
    def guess_bandwidth(self: Self)-> None:
        guess = 0
        mean = np.mean(self.data, axis=0)
        '''
        mean = np.zeros(self.d)
        for point in self.data:
            mean += point
        mean /= len(self.data)
        '''
        for point in self.data:
            guess += self.norm(point, mean)
        guess = np.sqrt(guess / (len(self.data)-1))
        self.bandwidth = guess * ((self.d + 2)*len(self.data)/4)**(-1/(self.d + 4))

    def guess_bandwidth_challenge(self: Self)-> None:
        pass

class Knn(Kernel):
    """A class for kernel density estimation with k-Nearest Neighbors
       derived from Kernel
    Attributes not already in Kernel:
        k: int      -- parameter for k-NN
        V: float    -- "volume" constant appearing in density
        neigh:    sklearn.neighbors.NearestNeighbors   -- data structure and methods for efficient k-NN computations
    """
    def __init__(self: Self, d: int, data: list[np.ndarray], k: int, V: float):
        super().__init__(d,data)
        self.k, self.V = k, V
        self.neigh = NearestNeighbors(n_neighbors=self.k)
        self.fit_knn()
   		#....

    def fit_knn(self):
        """Computes the inner data structure acccording to the data points."""
        self.neigh.fit(np.array(self.data))

    def knn(self, x: np.ndarray, vk:int):
        """The vk nearest-neighbors (vk can be different from self.k)."""
        return [np.array(self.data[i]) for i in self.neigh.kneighbors([x], n_neighbors=vk)[1][0] ]

    def k_dist_knn(self, x: np.ndarray, vk: int) -> float:
        """The distance to vk-th nearest-neighbor."""
        return self.neigh.kneighbors([x], n_neighbors=vk)[0][0][vk-1]

    def density(self, x):
        return self.k / (2 * len(self.data) * self.V * self.k_dist_knn(x, self.k))
    
    def meanshift(self: Self, k: int) -> None:
        for i in range(len(self.data)):
            self.data[i] = np.mean(self.knn(self.data[i], k), axis=0)


mydata = np.loadtxt("TD/../csv/galaxies_3D-short.xyz", delimiter=" ", dtype=float)
myknn=Knn(3,mydata,10,1.0)
for i in range(4):
    myknn.meanshift(10)
np.savetxt("csv/galaxies_3D-short-afterMeanshift.xyz", np.array(myknn.data), delimiter=" ")