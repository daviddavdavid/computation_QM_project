import numpy as np
import matplotlib.pyplot as plt

## CONSTANTS
a_0 = 1.0
E_initial = 0
Z = 2
epsilon_field = 1e-6
epsilon_shooting_method = 1e-9

def relative_potential(x_i, x_j, dx):
    return 1 / (np.abs(x_i - x_j) + dx)

def normalize_WF(psi, dx):
    prob_density = psi ** 2
    total_probability = np.trapezoid(prob_density, dx=dx)
    return psi / np.sqrt(total_probability)

def initial_psi_guess(x_grid, b):
    psi = np.exp(-b * x_grid**2)
    return normalize_WF(psi, x_grid[1] - x_grid[0])

def calculate_V_ij(psi_j, x_grid, dx):
    N = len(x_grid)
    V_ij = np.zeros(N)
    for i, x_i in enumerate(x_grid):
        V_one_i = []
        for j, x_j in enumerate(x_grid):
            integral_function = psi_j[j] * relative_potential(x_i, x_j, dx) * psi_j[j]
            V_one_i.append(integral_function)
        
        V_i = np.trapezoid(V_one_i, dx =dx)
        V_ij[i] = V_i
    
    return V_ij

def calculate_psi_i(V_ij, x_array, dx, E):
    psi = np.zeros(len(x_array))
    psi[0] = 1.0
    psi[1] = 1.0 # initial values, they dont actually matter because we normalize. We put the right BC already to make it quicker
    
    for i in range(1, len(psi)-1):
        psi[i+1] = (
            2 * psi[i]
            + 2 * (V_ij[i] - Z/(x_array[i] + dx) - E) * psi[i] * (dx**2)
            - (1 - dx/(x_array[i] + dx)) * psi[i-1]
        ) / (1 + dx/(x_array[i] + dx))
        
    psi = normalize_WF(psi, dx)
    return psi

def wave_function_cycle(psi_j, x_grid, dx):
    V_ij = calculate_V_ij(psi_j, x_grid, dx)
    converged = False
    reached_zero = False
    psi_i = np.copy(psi_j) # just to have the same shape, we will overwrite it in the loop anyway
    print("hi")
    cycles = 0
    delta_e = 1
    E_current = E_initial
    E_found = E_current
    previous_psi_last = psi_i[-1]

    while converged == False:
        E_current = E_current - delta_e
        cycles += 1
        if (np.abs(delta_e) < epsilon_shooting_method):
            converged = True
            E_found = E_current
            break

        psi_i = calculate_psi_i(V_ij, x_grid, dx, E_current)

        if psi_i[-1] * previous_psi_last < 0: # if the last value of the wave function changes sign, we know that we have passed the correct energy
            delta_e = delta_e - delta_e / 2
            print(delta_e)
            reached_zero = True
        previous_psi_last = psi_i[-1]
        print(E_current)
    if reached_zero == False:
        print("Warning: boundary condition not reached in wave function cycle")
    return psi_i, V_ij, E_found
    

def main():
    print("Starting the self-consistent Hartree calculation for the helium atom in 1D...")
    x_min = 0
    x_max = 15 * a_0
    dx = 0.002
    N = int((x_max - x_min) / dx) + 1
    x_grid = np.linspace(x_min, x_max, N)
    dx = x_grid[1] - x_grid[0]

    self_consistent = 0
    previous_psi_j = initial_psi_guess(x_grid, b=0.5) # we put an initial guess before we start
    E_true = 0
    cycle = 0

    while self_consistent <= 3:
        print("hi")
        psi_i_found, V_ij, E_found = wave_function_cycle(previous_psi_j, x_grid, dx)
       
        psi_average = (previous_psi_j + psi_i_found) / 2
        psi_i_again, V_ij, E_again = wave_function_cycle(psi_average, x_grid, dx)
        previous_psi_j = psi_i_found
        cycle += 1
        print(f"Cycle {cycle}: Found energy = {E_again}, Energy difference = {np.abs(E_found - E_again)}")
        # plt.plot(x_grid, psi_i_again)
        # plt.xlabel("x value")
        # plt.ylabel("psi(x)")
        # plt.title("The ground state WF of the helium 1D atom")
        # plt.show()

        energy_difference = np.abs(E_found - E_again)
        if energy_difference <= epsilon_field:
            self_consistent += 1
        else:
            self_consistent = 0

        if self_consistent == 3:
            electron_repulsion =[]

            for i, psi_i_value in enumerate(psi_i_again):
                electron_repulsion_i = psi_i_value ** 2 * V_ij[i]
                electron_repulsion.append(electron_repulsion_i)

            E_true = E_again
            print(f"The found energy of the helium electron ground state is {E_true}")
            E_helium = 2 * E_true - np.trapezoid(electron_repulsion, dx=dx)
            print(f"The found energy of the helium ground state is {E_helium}")
            
            plt.plot(x_grid, psi_i_again)
            plt.xlabel("x value")
            plt.ylabel("psi(x)")
            plt.title("The ground state WF of the helium 1D atom")
            plt.show()
            plt.savefig("Helium.png")
            print(cycle)
            break

        previous_psi_j = psi_i_again
    
if __name__ == "__main__":
    main()