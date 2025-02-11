import numpy as np

class Potential:
    pass
class FreeParticle(Potential):
    pass

class InfiniteSquareWell(Potential):
    pass

class FiniteSquareWell(Potential):
    pass

class TriangleWell(Potential):
    pass

class SimpleHarmonicOscillator(Potential):
    pass

class PotentialBarrier(Potential):
    def __init__(self, start, stop, mag, dim) -> None:
        V = np.zeros([dim, dim])
        for i in range(start, stop):
            V[i][i] = mag
