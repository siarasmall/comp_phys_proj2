!pip install ipywidgets scipy matplotlib numpy

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

# Simulation parameters
Nx = 300  # Change the number of spatial points to make it more detailed
L = 10  # Spatial domain from -L to L to make middle of plot equal 0
dx = 2 * L / Nx  # Spatial step size
dt = 0.005  # Time step size
x = np.linspace(-L, L, Nx)  # The background grid
Nt = 400  # Change the number of time points to make it longer

# I added two potential cases here: no barrier and barrier
def potential_free():
    return np.zeros(Nx)

def potential_barrier(height=5, width=2):
    V = np.zeros(Nx)
    center = 0
    mask = (x > center - width/2) & (x < center + width/2)
    V[mask] = height
    return V

# Sliders to change settings of certain parameters while the sim is running so it's more fun and dynamic
potential_dropdown = widgets.Dropdown(
    options=["Free Space", "Barrier"],
    value="Free Space",
    description="Potential:"
)

k0_slider = widgets.FloatSlider(value=5, min=1, max=10, step=0.5, description="Wave Momentum k₀:")
barrier_height_slider = widgets.FloatSlider(value=5, min=0, max=10, step=0.5, description="Barrier Height:")
barrier_width_slider = widgets.FloatSlider(value=2, min=0.5, max=5, step=0.5, description="Barrier Width:")

output = widgets.Output()

def run_simulation(potential_type, k0, barrier_height, barrier_width):
    with output:
        clear_output(wait=True)

        # Assign potential dynamically
        if potential_type == "Barrier":
            V = potential_barrier(barrier_height, barrier_width)
        else:
            V = potential_free()

        # Optimized Hamiltonian using sparse matrix representation
        diagonals = [
            np.full(Nx-1, 1 / (2 * dx**2)),  # Upper diagonal (Ψi+1)
            np.full(Nx, -1 / (dx**2)) + V,   # Main diagonal (KE + PE)
            np.full(Nx-1, 1 / (2 * dx**2))   # Lower diagonal (Ψi−1)
        ]
        H = sp.diags(diagonals, offsets=[-1, 0, 1], format="csc")  # Ensures correct format

        # Time evolution matrices using Crank-Nicolson
        I = sp.identity(Nx, dtype=complex, format="csc")  # Identity matrix in CSC format
        h1 = I - (1j * dt / 2) * H  # Left-hand matrix
        h2 = I + (1j * dt / 2) * H  # Right-hand matrix
        h1_solver = spla.splu(h1)  # Compute sparse LU decomposition only ONCE per change

        # Setting up the wave
        x0, sigma = -5, 1
        psi = np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)) * np.exp(1j * k0 * x)
        psi /= np.linalg.norm(psi)

        # Dynamic computation inside animation function
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlim(-L, L)
        ax.set_ylim(0, 1.2 * np.max(np.abs(psi)**2))
        ax.set_xlabel("Position x", color='black')
        ax.set_ylabel("Probability Density |Ψ|²", color='black')
        ax.set_title(f"Quantum Wave Evolution: {potential_type}", color='black')

        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        # Make wave look better with a colormap gradient
        line_main, = ax.plot([], [], color='crimson', lw=2, label="|Ψ|²")

        # Shade the potential barrier for better visibility
        if potential_type == "Barrier":
            ax.fill_between(x, 0, V / np.max(V) * np.max(np.abs(psi)**2) * 0.8, color='blue', alpha=0.3, label="Potential V(x)")
        ax.legend(facecolor='white', edgecolor='black', labelcolor='black')

        # Function to dynamically update the wave function
        def update(frame):
            nonlocal psi
            psi = h1_solver.solve(h2 @ psi)  # Solve for Ψ(t+1)
            line_main.set_data(x, np.abs(psi)**2)  # Update plot with probability density
            return line_main,

        ani = animation.FuncAnimation(fig, update, frames=Nt, blit=True, interval=30)
        display(HTML(ani.to_jshtml()))

# Dynamically update simulation when parameters change
interactive_sim = widgets.interactive(run_simulation,
                                      potential_type=potential_dropdown,
                                      k0=k0_slider,
                                      barrier_height=barrier_height_slider,
                                      barrier_width=barrier_width_slider)

display(interactive_sim, output)
