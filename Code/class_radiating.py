import numpy as np
import matplotlib.pyplot as plt

# Constants
epsilon_0 = 8.854e-12  # Permittivity of free space
e_charge = 1.602e-19  # Elementary charge
m_electron = 9.109e-31  # Mass of electron
c = 2.998e8  # Speed of light

# Coulomb's constant
k_e = 1 / (4 * np.pi * epsilon_0)

# Function to calculate electrostatic force
def electrostatic_force(r):
    r_mag = np.linalg.norm(r)
    return k_e * e_charge**2 / r_mag**2 * r / r_mag

# Function to update position and velocity using Euler's method
def update_motion(r, v, dt):
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)
    acceleration = - electrostatic_force(r) / m_electron
    recoil_force = radiated_power(v) / c * v / v_mag  # Recoil force due to radiation
    r_new = r + v * dt
    v_new = v + (acceleration + recoil_force) * dt
    return r_new, v_new

# Function to calculate power radiated by an accelerated charge (Larmor's formula)
def radiated_power(v):
    a = np.linalg.norm(electrostatic_force(r)) / m_electron  # Magnitude of acceleration
    return (2 / 3) * e_charge**2 * a**2 / (4 * np.pi * epsilon_0 * c**3)

# Simulation parameters
num_steps = 100000
dt = 1e-18  # Time step (seconds)
initial_distance = 5e-11  # Initial distance from nucleus (meters)
initial_velocity = 2e6  # Initial velocity (m/s)

# Initial conditions
r = np.array([initial_distance, 0, 0])  # Initial position vector
v = np.array([0, initial_velocity, 0])  # Initial velocity vector

# Arrays to store position, velocity, and radiated power over time
positions = np.zeros((num_steps, 3))
velocities = np.zeros((num_steps, 3))
radiated_powers = np.zeros(num_steps)

# Simulate motion
for i in range(num_steps):
    positions[i] = r
    velocities[i] = v
    radiated_powers[i] = radiated_power(v)
    r, v = update_motion(r, v, dt)

# Plot trajectory
plt.figure(figsize=(10, 6))
plt.plot(positions[:, 0], positions[:, 1], label='Electron trajectory')
plt.plot(0, 0, 'ro', label='Nucleus (Proton)')
plt.xlabel('X position (m)')
plt.ylabel('Y position (m)')
plt.title('Simulation of Electron Motion in Hydrogen Atom with Radiation')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

# Plot radiated power
plt.figure()
plt.plot(np.arange(num_steps) * dt, radiated_powers)
plt.xlabel('Time (s)')
plt.ylabel('Radiated Power (W)')
plt.title('Radiated Power vs Time')
plt.grid(True)
plt.show()
