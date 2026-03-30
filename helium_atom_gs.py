import numpy as np
import matplotlib.pyplot as plt
import click

## CONSTANTS
E_initial_low = -10
E_initial_high = 0
epsilon = 1e-6
hbar = 1
m = 1
e = 1

def normalize_without_infinity(u, dx):
    minimum_index = np.argmin(np.abs(u[1:]))
    # removes the part of the wave function that diverges to infinity, which would mess up the normalization
    for i in range(minimum_index, len(u)):
        u[i] = 0
    
    return normalize_WF(u, dx)

def normalize_WF(psi, dx):
    prob_density = psi ** 2
    total_probability = np.trapezoid(prob_density, dx=dx)
    return psi / np.sqrt(total_probability)

def initial_u_guess(x_grid, dx):
    u = x_grid * np.exp(-2 * x_grid) 
    return normalize_WF(u, dx)


def calculate_V_cl(u, x_grid, dx):
    V = np.zeros(len(x_grid))  # NumPy array, not a list
    Q = 0
    for i in range(1, len(x_grid)):
        Q += u[i]**2 * dx
        V[i] = V[i-1] - e * Q * (dx/x_grid[i])/x_grid[i] # collecting the contribution of the charge inside the sphere of radius x_grid[i]
    
    V += e**2/x_grid[-1] - V[-1] 
    return V

def calculate_u(u, u_1, V_cl, x_grid, dx, E_accent):
    u[0] = 0.0
    u[1] = u_1
    for i in range(1, len(u)-1):
            u[i+1] = 2*u[i] - u[i-1] + (2 * m * dx**2 / hbar**2)* (-2 * e**2 / (x_grid[i]) + V_cl[i] - E_accent) * u[i]
    return u

def wave_function_cycle(u_1, x_grid, dx, V_cl, E_guess):
    E = E_guess
    dE = 0.1

    u = np.zeros(len(x_grid))
    
    # 1. First, find where the wave function at the boundary flips sign
    u = calculate_u(u, u_1, V_cl, x_grid, dx, E)
    t_0 = u[-1]
    
    while abs(dE) > 1e-7:
        E += dE
        u = calculate_u(u, u_1, V_cl, x_grid, dx, E)
        t_1 = u[-1]
        
        # If the sign flipped, we passed the eigenvalue
        # Reverse direction and sharpen the search.
        if t_0 * t_1 < 0:
            E -= dE        # Go back to the previous 'good' energy
            dE /= -2.0     # Reverse and halve the step
        
        t_0 = t_1

    return normalize_without_infinity(u, dx), E
    
def calculate_kinetic_and_repulsion_energy(u, x_grid, dx):
    # We calculate the kinetic and repulsion energy using the formulas from the paper, which are derived from the Hartree method.
    b_integrand = u * u / x_grid
    b = -2 * e**2 * np.trapezoid(b_integrand, dx=dx)

    d2_u = np.zeros(len(x_grid))
    for i in range(1, len(u)-1):
        d2_u[i] = (u[i+1] + u[i-1] - 2*u[i]) / (dx**2)

    a_integrand = u * d2_u
    a = - hbar**2 / (2 * m) * np.trapezoid(a_integrand, dx=dx)

    return a, b
    

@click.command()
@click.option('--dx', default=0.001, help='The step size for the radial grid, standard value is from the paper.')
@click.option('--i_max', default=15000, help='The number of steps for the radial grid, standard value is from the paper.')
def main(dx, i_max):
    wave_function_calculator(dx, i_max)

def wave_function_calculator(dx, i_max):
    E_actual = 0    
    print("Starting the self-consistent Hartree cycle for the helium atom...")
    x_min = dx # too prevent 0 divison errors
    x_max = i_max * dx
    x_grid = np.arange(x_min, x_max, dx)
    u_1 = 1.0 # we start with these values and tend to replace them later

    u = initial_u_guess(x_grid, dx)
    one_s_energy_guess = -5

    self_consistent = False
    i = 0
    E_current = 0
    E_previous = 100 # dummy value
    while self_consistent == False:
        i += 1
        V_cl = calculate_V_cl(u, x_grid, dx)
        u, E_current = wave_function_cycle(u_1, x_grid, dx, V_cl, one_s_energy_guess)

        print(f"Cycle {i}: Energy = {E_current:.6f}, previous Energy {E_previous:.6f}")
        if np.abs(E_current - E_previous) < 1e-3:
            self_consistent = True
            print(f"Self-consistent solution found after {i} cycles with orbital energy {E_current:.6f}")

            A, B = calculate_kinetic_and_repulsion_energy(u, x_grid, dx)
            E_actual = E_current + A + B
            E_actual = 27.211 * E_actual
            print(f"The final result is the energy {E_actual:.6f} eV")

            plt.plot(x_grid, (u/x_grid)) # we plot u(r)/r to get the actual wave function
            plt.xlabel("r")
            plt.ylabel("R_1s(r)")
            plt.title("The plot of the WF of the ground state")
            plt.savefig("helium_atom_ground_state.png")
            plt.show()

        E_previous = E_current
        u_1 = u[1]
    return E_actual

def convergence_test():
    dx_values = [0.01, 0.005, 0.001, 0.0005]
    i_max_value = 15000 # we keep this constant
    energies_found = []

    for dx in dx_values:
        print(f"Testing with dx = {dx}")
        energy = wave_function_calculator(dx, i_max_value)
        energies_found.append(energy)

    plt.plot(energies_found, marker='o')
    plt.xlabel("Test Case")
    plt.ylabel("Found Energy")
    plt.title("Convergence Test")
    plt.savefig("convergence_test_gs_1s.png")
    plt.show()

if __name__ == "__main__":
    convergence_test()