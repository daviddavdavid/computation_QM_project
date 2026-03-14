import numpy as np
import matplotlib.pyplot as plt

## CONSTANTS
E_initial_low = -10
E_initial_high = 0
epsilon = 1e-6
Z = 2 # nuclear charge for helium
hbar = 1
m = 1
e = 1


def initial_u_1s_guess(x_grid, dx):
    u = x_grid * np.exp(-Z * x_grid) 
    return normalize_WF(u, dx)

def initial_u_nl_guess(x_grid, dx, n, l):
    if n == 1 and l == 0:
        return initial_u_1s_guess(x_grid, dx)
    elif n == 2 and l == 0:
        u_2s = x_grid * (2 - Z * x_grid) * np.exp(-Z * x_grid / 2)
        return normalize_WF(u_2s, dx)
    else:
        raise ValueError("Initial wave function for the given n and l is not implemented.")

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

def calculate_V_cl(u, x_grid, dx):
    V = np.zeros(len(x_grid))  # NumPy array, not a list
    Q = 0
    for i in range(1, len(x_grid)):
        Q += u[i]**2 * dx
        V[i] = V[i-1] - e * Q * (dx/x_grid[i])/x_grid[i] # collecting the contribution of the charge inside the sphere of radius x_grid[i]
    
    V += e**2/x_grid[-1] - V[-1] 
    return V

def calculate_u_1s(u_1s, u_1s_1, u_nl, u_nl_1, V_cl, V_x, x_grid, dx, E_accent, singlet_or_triplet):
    N = len(x_grid)
    u_1s[0] = 0.0
    u_1s[1] = u_1s_1

    for i in range(1, N-1): ## N-1 because we need to calculate u[i+1]
        u_1s[i+1] = 2*u_1s[i] - u_1s[i-1] + (2 * m * dx**2 / hbar**2)* (-2 * e**2 / (x_grid[i]) + V_cl[i] - E_accent) * u_1s[i]

        if singlet_or_triplet == "singlet":
            u_1s[i+1] += 2 * m * dx**2 / hbar**2 * V_x[i] * u_nl[i]
        elif singlet_or_triplet == "triplet":
            u_1s[i+1] -= 2 * m * dx**2 / hbar**2 * V_x[i] * u_nl[i]
        else:
            raise ValueError("Invalid value for singlet_or_triplet. Must be 'singlet' or 'triplet'.")
    return u_1s

def calculate_u_n0(u_1s, u_1s_1, u_nl, u_nl_1, V_cl, V_x, x_grid, dx, E_accent, singlet_or_triplet):
    N = len(x_grid)
    u_nl[0] = 0.0
    u_nl[1] = u_nl_1

    for i in range(1, N-1): ## N-1 because we need to calculate u[i+1]
        u_nl[i+1] = 2*u_nl[i] - u_nl[i-1] + (2 * m * dx**2 / hbar**2)* (-2 * e**2 / (x_grid[i]) + V_cl[i] - E_accent) * u_nl[i]
        if singlet_or_triplet == "singlet":
            u_nl[i+1] += 2 * m * dx**2 / hbar**2 * V_x[i] * u_1s[i]
        elif singlet_or_triplet == "triplet":
            u_nl[i+1] -= 2 * m * dx**2 / hbar**2 * V_x[i] * u_1s[i]
        else:
            raise ValueError("Invalid value for singlet_or_triplet. Must be 'singlet' or 'triplet'.")
    return u_nl

# this neat trick lets me calculate the wave function for both the 1s and 2s states using the same function
# the arguments are getting a bit out of hand though
def calculate_u(u_1s, u_1s_1, u_nl, u_nl_1, V_cl, V_x, x_grid, dx, E_accent, singlet_or_triplet, n, l):
    if n == 1 and l == 0:
        return calculate_u_1s(u_1s, u_1s_1, u_nl, u_nl_1, V_cl, V_x, x_grid, dx, E_accent, singlet_or_triplet)
    elif n == 2 and l == 0:
        return calculate_u_n0(u_1s, u_1s_1, u_nl, u_nl_1, V_cl, V_x, x_grid, dx, E_accent, singlet_or_triplet)
    else:
        raise ValueError("Wave function calculation for the given n and l is not implemented.")

def wave_function_cycle(u_1s, u_1s_1, u_nl, u_nl_1, V_cl, V_x, x_grid, dx, E_guess, singlet_or_triplet, n, l):
    E = E_guess
    dE = 0.1

    u = np.zeros(len(x_grid))
    
    # 1. First, find where the wave function at the boundary flips sign
    u = calculate_u(u_1s, u_1s_1, u_nl, u_nl_1, V_cl, V_x, x_grid, dx, E, singlet_or_triplet, n, l)
    t_0 = u[-1]
    
    while abs(dE) > 1e-7:
        E += dE
        u = calculate_u(u_1s, u_1s_1, u_nl, u_nl_1, V_cl, V_x, x_grid, dx, E, singlet_or_triplet, n, l)
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

# Here is the implementation for V_x. I did this for every l instead of just for the s states because 
# I wanted to be able to calculate the energy of the excited states as well, which would require V_x for l > 0.
def calculate_V_x(u_1s, u_nl, l, x_grid, dx):
    N = len(x_grid)
    Q_l_integrand = u_1s * u_nl / (2 * l + 1) * (x_grid**l)
    Q_l = np.trapezoid(Q_l_integrand, dx=dx)

    w_a = np.zeros(N)
    w_b = np.zeros(N)

    w_a[0] = 0
    w_b[0] = 0

    w_a[1] = 0.1 * dx
    w_b[1] = 0.2 * dx

    for i in range(1, N-1):
        source = (u_1s[i] * u_nl[i] / x_grid[i]) * dx**2
        centrifugal = l * (l + 1) / (x_grid[i]**2) * dx**2
        w_a[i+1] =  source + (centrifugal + 2) * w_a[i] - w_a[i-1]
        w_b[i+1] = source + (centrifugal + 2) * w_b[i] - w_b[i-1]
    
    r_1l = x_grid[-1]**l
    r_2l = x_grid[-2]**l
    
    a = Q_l * (r_2l * w_b[-2] - r_1l * w_b[-1]) /  (r_1l * r_2l * (w_a[-1] * w_b[-2] - w_b[-1] * w_a[-2]))
    b = Q_l * (r_1l * w_a[-1] - r_2l * w_a[-2]) /  (r_1l * r_2l * (w_a[-1] * w_b[-2] - w_b[-1] * w_a[-2]))

    V_x = (a * w_a + b * w_b) / x_grid

    return V_x
    

def main():
    n_second_WF = 2
    l_second_WF = 0
    singlet_or_triplet = "singlet" # change this to triplet to calculate the triplet state

    print("Starting the self-consistent Hartree cycle for the helium atom...")

    # grid parameters
    dx = 0.001
    i_max = 15000
    x_min = dx # too prevent 0 divison errors
    x_max = i_max * dx
    x_grid = np.arange(x_min, x_max, dx)

    # wave function and energy initialization
    u_1s_1 = 1.0 # we start with these values and tend to replace them later
    u_nl_1 = 1.0
    u_1s = initial_u_1s_guess(x_grid, dx)
    u_nl = initial_u_nl_guess(x_grid, dx, n_second_WF, l_second_WF)
    one_s_energy_guess = -5
    two_s_energy_guess = -10

    # self-consistent loop
    self_consistent = False
    i = 0
    E_current_1s = 0 # dummy value
    E_previous_1s = 100 # dummy value
    
    E_current_nl = 0 # dummy value
    E_previous_nl = 100 # dummy value
    while self_consistent == False:
        i += 1
        V_cl = calculate_V_cl(u_1s, x_grid, dx)
        V_x = calculate_V_x(u_1s, u_nl, l_second_WF, x_grid, dx)

        u_1s, E_current_1s = wave_function_cycle(u_1s, u_1s_1, u_nl, u_nl_1, V_cl, V_x, x_grid, dx, one_s_energy_guess, singlet_or_triplet, n_second_WF, l_second_WF)

        V_cl = calculate_V_cl(u_1s, x_grid, dx)
        V_x = calculate_V_x(u_1s, u_nl, l_second_WF, x_grid, dx)

        u_nl, E_current_nl = wave_function_cycle(u_1s, u_1s_1, u_nl, u_nl_1, V_cl, V_x, x_grid, dx, two_s_energy_guess, singlet_or_triplet, n_second_WF, l_second_WF)

        print(f"Cycle {i}: Energy_1s = {E_current_1s:.6f}, previous Energy 1s {E_previous_1s:.6f}")
        print(f"Cycle {i}: Energy_nl = {E_current_nl:.6f}, previous Energy nl {E_previous_nl:.6f}")

        # plt.plot(x_grid, u_1s) # we plot u(r)/r to get the actual wave function
        # plt.show()

        # plt.plot(x_grid, u_nl) # we plot u(r)/r to get the actual wave function
        # plt.show()
        if np.abs(E_current_1s - E_previous_1s) < 1e-3 and np.abs(E_current_nl - E_previous_nl) < 1e-3:
            self_consistent = True
            print(f"Self-consistent solution found after {i} cycles with orbital energy 1s {E_current_1s:.6f}")
            print(f"Self-consistent solution found after {i} cycles with orbital energy 1nl {E_current_nl:.6f}")

            A, B = calculate_kinetic_and_repulsion_energy(u_1s, x_grid, dx)
            E_actual_1s= E_current_1s + A + B
            E_actual_1s = 27.211 * E_actual_1s # convert to eV
            E_actual_nl = E_current_nl + A + B
            E_actual_nl = 27.211 * E_actual_nl # convert to e
            print(f"The final result is the energy of the 1s {singlet_or_triplet} state : {E_actual_1s:.6f} eV")
            print(f"The final result is the energy of the 1nl {singlet_or_triplet} state: {E_actual_nl:.6f} eV")

            plt.plot(x_grid, (u_1s /x_grid)) # we plot u(r)/r to get the actual wave function
            plt.xlabel("r")
            plt.ylabel("psi")
            plt.title("The plot of the 1s WF of the ground state")
            plt.show()

            plt.plot(x_grid, (u_nl /x_grid)) # we plot u(r)/r to get the actual wave function
            plt.xlabel("r")
            plt.ylabel("psi")
            plt.title(f"The plot of the {n_second_WF}s WF of the {singlet_or_triplet} state")
            plt.show()

        E_previous_1s = E_current_1s
        E_previous_nl = E_current_nl
        u_1s_1 = u_1s[1]
        u_nl_1 = u_nl[1]

if __name__ == "__main__":
    main()