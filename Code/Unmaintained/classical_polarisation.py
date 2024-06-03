import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants in SI units
k = 8.9875517923e9
m_e = 9.10938356e-31
e = 1.602176634e-19

# Initial conditions assuming proton doesn't move
initial_position = np.array([2.0, 2.0, 0])
initial_velocity = np.array([1e4, -1e4,0 ])
external_field = np.array([0, 0, 1e3])      #taken constant for now
dt = 1e-14
num_steps = 100000


def calculate_acceleration(position):
    r = np.linalg.norm(position)
    acceleration = -k * e * position / (m_e * r**3)
    acceleration += external_field / m_e
    return acceleration


positions = np.zeros((num_steps, 3))
polarization = np.zeros(num_steps)
position = initial_position
velocity = initial_velocity
for i in range(num_steps):
    acceleration = calculate_acceleration(position)
    velocity += acceleration * dt
    position += velocity * dt
    positions[i] = position
    polarization[i] =  (np.linalg.norm(position))**3 #just distance^3 for now, no constants up front
    # or actually doing first principles proportionality extraction


fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot(positions[:,0], positions[:,1], positions[:,2])
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('3D Trajectory of Electron in the Presence of Proton with External Field')


ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(np.arange(num_steps), polarization)
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Polarization (radians)')
ax2.set_title('Polarization of Electron with External Field')

plt.tight_layout()
plt.show()
