import numpy as np
from scipy.sparse import diags
class Potential:
    def __init__(self, mags, total_num) -> None:
        """
        Inits a Potential object.

        Params:
            - Mags (np array of ints): Magnitudes of x values
        """
        self.mags = mags
        self.matrix = diags([np.ones(total_num-1),-2*np.ones(total_num),np.ones(total_num-1)], [-1,0,1]).toarray()
        self.total_num = total_num

    def getMatrix(self):
        return self.matrix
class PotentialWithBarriers(Potential):
    def __init__(self, data, total_num) -> None:
        # data = list of tuples: (start, stop, mag)
        V = np.zeros([total_num, total_num])
        for start, stop, mag in data:
            for i in range (start, stop):
                V[i][i] = mag
        super.__init__(V, total_num)

class Barrier(PotentialWithBarriers):
    def __init__(self, start, stop, y0, a, total_num) -> None:
        data = [(0, start, y0), (start, stop, y0 + a), (stop, total_num, y0)]
        super().__init__(data, total_num)

class InfiniteSquareWell(PotentialWithBarriers):
    def __init__(self, start, stop, y0, total_num) -> None:
        data = [(0, start, y0), (start, stop, -np.inf), (stop, total_num, y0)]
        super().__init__(data, total_num)

class FiniteSquareWell(PotentialWithBarriers):
    def __init__(self, start, stop, y0, a, total_num) -> None:
        data = [(0, start, y0), (start, stop, y0 - a), (stop, total_num, y0)]
        super().__init__(data, total_num)

class FreeParticle(Potential):
    pass

class TriangleWell(Potential):
    pass

class SimpleHarmonicOscillator(Potential):
    pass


