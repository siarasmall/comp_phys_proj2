import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class WaveFunctionSimulation:
  def __init__(self, start=0, end=100, points=500, dt=100, steps=300, periodic=True, potential=None):
    self.START=start
    self.END=end
    self.NUM_POINTS=points
    self.STEPS=steps
    self.dt=dt
    self.x, self.dx = np.linspace(start, end, points, retstep=True)
    self.periodic=periodic
    self.wavefunctions = []
    # self.initial_psi = np.zeros([self.NUM_POINTS])

    if potential==None:
      self.potential = np.zeros([self.NUM_POINTS])
    else:
      self.potential = potential

    self.step_matrix = self.init_matrix()

  def init_matrix(self):
    M = diags([np.ones(self.NUM_POINTS-1),-2*np.ones(self.NUM_POINTS),np.ones(self.NUM_POINTS-1)], [-1,0,1]).toarray()
    if self.periodic:
      M[0][-1]=1
      M[-1][0]=1

    H = -0.5 * (self.dx*self.dx) * M + self.potential
    I = np.identity(self.NUM_POINTS)
    h1 = I - (1j * self.dt * 0.5 * H)
    h2 = I + (1j * self.dt * 0.5 * H)
    h2_inv = np.linalg.inv(h2)
    return np.matmul(h2_inv, h1)

  def init_gaussian(self, x, mu, sigma, k):
    A = (2*np.pi)**-0.25 * sigma**-0.5 * np.exp(-(x-mu)**2/((2*sigma)**2))
    return A * np.exp(1j*k*x)

  def step(self, psi):
    return np.matmul(self.step_matrix, psi)

  def simulate(self, mu, sigma, k):
    self.wavefunctions = [self.init_gaussian(self.x, mu, sigma, k)]
    for i in range(self.STEPS):
      self.wavefunctions.append(self.step(self.wavefunctions[-1]))

    self.wavefunctions = np.array(self.wavefunctions)
    return self.wavefunctions

  def get_animation_vars(self):
    return self.x, self.START, self.END, self.STEPS