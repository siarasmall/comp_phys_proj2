import numpy as np

class FreeParticle:
    pass

class InfiniteSquareWell:
    pass

class FiniteSquareWell:
    pass

class TriangleWell:
    pass

class SimpleHarmonicOscillator:
    pass

class PotentialBarrier:
    def __init__(self, start, stop, mag, dim) -> None:
        V = np.zeros([dim, dim])
        for i in range(start, stop):
            V[i][i] = mag
