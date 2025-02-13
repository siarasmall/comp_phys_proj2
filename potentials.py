import numpy as np
from scipy.sparse import diags
class Potential:
    def __init__(self, mags, total_num) -> None:
        """
        Inits a Potential object.

        Params:
            - Mags (np array of ints): Magnitudes of x values
            - Total_num (int): total number of x values
        """
        self.mags = mags
        self.matrix = diags([np.ones(total_num-1),-2*np.ones(total_num),np.ones(total_num-1)], [-1,0,1]).toarray()
        self.total_num = total_num

    def getMatrix(self):
        return self.matrix
class PotentialWithBarriers(Potential):
    def __init__(self, data, total_num) -> None:
        """
        Inits a Potential object with barriers.

        Params:
            - data: List of tuples (start, stop, mag) representing the start index, stop index, 
            and magnitude of each "barrier"/block
        """
        V = np.zeros([total_num, total_num])
        for start, stop, mag in data:
            for i in range (start, stop):
                V[i][i] = mag
        super.__init__(V, total_num)

class Barrier(PotentialWithBarriers):
    def __init__(self, start, stop, y0, a, total_num) -> None:
        """
        Inits a Potential barrier.

        Params:
            - Start (int): start index of barrier
            - Stop (int): stop index of barrier
            - y0 (int): initial height (pre-barrier)
            - a (int): height of barrier (NOT total height with barrier)
            - Total_num (int): total number of x values
        """
        data = [(0, start, y0), (start, stop, y0 + a), (stop, total_num, y0)]
        super().__init__(data, total_num)

class InfiniteSquareWell(PotentialWithBarriers):
    def __init__(self, start, stop, y0, total_num) -> None:
        """
        Inits an infinite square well potential.

        Params:
            - Start (int): start index of well
            - Stop (int): stop index of well
            - y0 (int): initial height (pre-well)
            - Total_num (int): total number of x values
        """
        data = [(0, start, y0), (start, stop, -np.inf), (stop, total_num, y0)]
        super().__init__(data, total_num)

class FiniteSquareWell(PotentialWithBarriers):
    def __init__(self, start, stop, y0, a, total_num) -> None:
        """
        Inits a finite square well potential.

        Params:
            - Start (int): start index of well
            - Stop (int): stop index of well
            - y0 (int): initial height (pre-well)
            - a (int): depth of well (NOT total height with well)
            - Total_num (int): total number of x values
        """
        data = [(0, start, y0), (start, stop, y0 - a), (stop, total_num, y0)]
        super().__init__(data, total_num)

class FreeParticle(Potential):
    pass

class TriangleWell(Potential):
    pass

class SimpleHarmonicOscillator(Potential):
    pass


