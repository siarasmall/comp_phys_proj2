!pip install ipywidgets

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

# Simulation parameters
Nx = 300  # Change the number of spacial points to make it more detailed
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

# Sliders to change settings of certain paramters while the sim is running so it's more fun and dynamic
potential_dropdown = widgets.Dropdown(
    options=["Free Space", "Barrier"],
    value="Free Space",
    description="Potential:"
)

# Sliders work now
k0_slider = widgets.FloatSlider(value=5, min=1, max=10, step=0.5, description="Wave Momentum k₀:")
barrier_height_slider = widgets.FloatSlider(value=5, min=0, max=10, step=0.5, description="Barrier Height:")
barrier_width_slider = widgets.FloatSlider(value=2, min=0.5, max=5, step=0.5, description="Barrier Width:")

output = widgets.Output()

def run_simulation(potential_type, k0, barrier_height, barrier_width):
    with output:
        clear_output(wait=True)

        # After sim is run, the function is supposed to assign the potential value based on what the user sets but doesn't work
        # Don't know why <-- someone look at it

        if potential_type == "Barrier":
            V = potential_barrier(barrier_height, barrier_width)
        else:
            V = potential_free()

        # Here, I tried to convert your solution for the Hamiltonian matrix using diagonal matrices into a for loop
        # 1D Hamiltonian: H = - (h²/2m) x (d²/dx²) + V(x)
        # Approximate d²/dx² with finite diff


        H = np.zeros((Nx, Nx), dtype=complex)
        for i in range(1, Nx - 1):
            H[i, i - 1] = 1 / (2 * dx**2)  # Kinetic energy Ψi−1
            H[i, i + 1] = 1 / (2 * dx**2)  # Kinetic energy Ψi+1
            H[i, i] = -1 / (dx**2) + V[i]  # Diagonal KE + PE


        # Calculating time evolution matrices using the Crank-Nicolson method
        # Intial EQ:             i(dΨ/dt) = HΨ                                        "!! IMPORTANT NOTE !!"
        # Discretized EQ:        (I - iHΔt/2) Ψ(t+1) = (I+iHΔt/2) Ψ(t)             SOMEONE CHECK IF THIS IS RIGHT

        I = np.identity(Nx, dtype=complex) # Diagonal Matrix to solve for Ψ(t+1)
        h1 = I - (1j * dt / 2) * H # Inverts matrix (Really big matrix)
        h2 = I + (1j * dt / 2) * H # Plugs in wavefunction
        lu, piv = la.lu_factor(h1) # Using Lower/Upper Decomposition method to factor h1 into two simpler matrices which are easier to compute with.
                                                                                            # This should make the runtime faster.
        # This part of the code calculates the wave as time changes.



        # Setting up the wave
        x0, sigma = -5, 1
            # x0 = Initial position
            # Sigma = width of wave
        psi = np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)) * np.exp(1j * k0 * x)
            # psi = the wavefunction
        psi /= np.linalg.norm(psi)
            # This norm function normalizes the wavefunction

        # Making the wave evolve over time by plugging in the values we calculated above
        wave_evolution = []
        for t in range(Nt): # This loop keeps solving for the wavefunction at time t+1
            psi = la.lu_solve((lu, piv), h2 @ psi) # Plug initial values in and use the for loop above to solve for Ψ(t+1)
            wave_evolution.append(np.abs(psi) ** 2) # Append the evolution array with the new probability calculated at the end of the loop

        wave_evolution = np.array(wave_evolution) # After the end of the loop, set the array as the array with stored values in the loop


      #---------------------------------------------------------------------------------------------------------------------------------------

       # POSSIBLE IDEA GO BACK HERE LATER: maybe the reason that changing the settings doesn't work is because the code only calculates the probability
       # once and saves it in the wave_evolution array while the code is being compiled. Once the data points are saved there, the array can't be overwritten
       # whie the simulation is being run.


       # POSSIBLE SOLUTION: Instead of calculating the probability plot points while the code is being compiled, find a way to have an empty array at the
       # start of the sim. Then, set parameters and calculate the plot points while the sim is running. Then you can watch the simulation using the calculated
       # points. This would mean that we have to find a way to dynamically fill the wave_evolution while the sim is running. I don't know how to do that.

      #---------------------------------------------------------------------------------------------------------------------------------------


        # Seting up the plot diagram background
        # Can change the colors if you want; decide what looks better
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlim(-L, L)
        ax.set_ylim(0, np.max(wave_evolution) * 1.2)
        ax.set_xlabel("Position x", color='black')
        ax.set_ylabel("Probability Density |Ψ|²", color='black')
        ax.set_title(f"Quantum Wave Evolution: {potential_type}", color='black')

        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        # Visualizing the wave function and potential
        line_main, = ax.plot([], [], 'r', lw=2, label="|Ψ|²")  # Made wave red

        # If the user changes the potential from free space to barrier, it should show a barrier
        if potential_type == "Barrier":
            ax.plot(x, V / np.max(V) * np.max(wave_evolution) * 0.8, '--b', label="Potential V(x)")  # Made barrier color blue
        ax.legend(facecolor='white', edgecolor='black', labelcolor='black')

        # Animation functions
        def init():
            line_main.set_data([], [])
            return line_main,

        # Updates the next frame in the animation for the wave to follow the data points saved in the evolution array
        def update(frame):
            line_main.set_data(x, wave_evolution[frame])
            return line_main,

        ani = animation.FuncAnimation(fig, update, frames=Nt, init_func=init, blit=True, interval=30)
        display(HTML(ani.to_jshtml()))

# Trying to update sim after changing settings but not working
# Barrier dropdown update just worked once but now it's not working again
interactive_sim = widgets.interactive(run_simulation,
                                      potential_type=potential_dropdown,
                                      k0=k0_slider,
                                      barrier_height=barrier_height_slider,
                                      barrier_width=barrier_width_slider)

display(interactive_sim, output)
