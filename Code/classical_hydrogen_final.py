import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def gravitational_force(m1, m2, r):
    G = 8.987551788e9  # Gravitational constant (--> 1/4 pi e_0)
    return G * m1 * m2 / r**2

def rk4_step(func, t, y, dt):
    k1 = func(t, y)
    k2 = func(t + 0.5*dt, y + 0.5*dt*np.array(k1))
    k3 = func(t + 0.5*dt, y + 0.5*dt*np.array(k2))
    k4 = func(t + dt, y + dt*np.array(k3))
    return y + (dt/6.0) * (np.array(k1) + 2*np.array(k2) + 2*np.array(k3) + np.array(k4))

def two_body_equations(t, state):
    x1, y1, z1, vx1, vy1, vz1, x2, y2, z2, vx2, vy2, vz2 = state

    # Calculate distance between the two bodies
    r = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

    # Calculate gravitational force magnitude
    force_magnitude = gravitational_force(m1, m2, r)

    # Calculate gravitational force components
    force_x = force_magnitude * (x2 - x1) / r
    force_y = force_magnitude * (y2 - y1) / r
    force_z = force_magnitude * (z2 - z1) / r
    ext_force_x = 1e-10 # q E for E = 6.2 N/C
    ext_force_y = 0
    ext_force_z = 0
    # Accelerations
    M1 = 1.67262192e-27 # proton mass
    M2 = 9.1093837e-31 #electron mass
    ax1 = (force_x / M1) + ext_force_x/M1
    ay1 = (force_y / M1) + ext_force_y/M1
    az1 = (force_z / M1) + ext_force_z/M1
    ax2 = (force_x / M2) + ext_force_x/M2 #-(force_x / M2) - ext_force_x/M2
    ay2 = (force_y / M2) + ext_force_y/M2 #-(force_y / M2) - ext_force_y/M2
    az2 = (force_z / M2) + ext_force_z/M2 #-(force_z / M2) - ext_force_z/M2

    return [vx1, vy1, vz1, ax1, ay1, az1, vx2, vy2, vz2, ax2, ay2, az2]

def simulate_two_body(m1, m2, r_initial, v_initial, num_steps, dt):
    state = [0, 0, 0, 0, 0, 0, r_initial, 0, 0, 0, v_initial, 0]  # Initial state

    positions = np.zeros((num_steps, 2, 3))  # Array to store positions of both bodies
    distances = np.zeros(num_steps)  # Array to store distances between the bodies
    angles = np.zeros(num_steps)
    
    for step in range(num_steps):
        t = step * dt
        state = rk4_step(two_body_equations, t, state, dt)
        positions[step] = np.array([[state[0], state[1], state[2]], [state[6], state[7], state[8]]])
        distances[step] = np.linalg.norm(positions[step, 1] - positions[step, 0])
        angles[step] =  np.dot(positions[step, 1], positions[step, 0]) / (np.linalg.norm(positions[step, 1])*np.linalg.norm(positions[step, 0])) # cosine
    return positions, distances, angles 

# Parameters
m1 = 1.60217663 * 1e-19 # Mass of body 1 (e.g., Earth) in kg --> proton charge
m2 =-1.60217663 * 1e-19  # Mass of body 2 (e.g., Moon) in kg  --> electron charge
r_initial = 5.3e-11 # Initial distance between the bodies in meters --> bohr radius
v_initial = 2e6 #2e6 # Initial velocity of body 2 in m/s
num_steps = 10000  # Number of time steps
dt = 10e-20  # Time step size in seconds

# Simulate two-body system
positions, distances, angles = simulate_two_body(m1, m2, r_initial, v_initial, num_steps, dt)

# Plot positions in 3D
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.plot(positions[:, 0, 0], positions[:, 0, 1], positions[:, 0, 2], label='Body 1')
ax.plot(positions[:, 1, 0], positions[:, 1, 1], positions[:, 1, 2], label='Body 2')
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Z Position (m)')
ax.set_title('Two Body Problem Simulation (3D)')
ax.legend()

# Plot distance versus time
fig3 = plt.figure()
ax = fig3.add_subplot(122)
plt.plot(np.arange(num_steps) * dt, distances)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Distance (m)')
ax.set_title('Distance Between Bodies vs Time')

# Plot angles 
fig2 = plt.figure()
ax = fig2.add_subplot(122)
ax.plot(np.arange(num_steps) * dt, angles, color = "red")
ax.set_xlabel('Time (s)')
ax.set_ylabel('cos(Angles)')
ax.set_title(' cos(Angle Between Bodies) vs Time')


#plt.plot(np.arange(num_steps) * dt, angles)
#plt.xlabel('Time (s)')
#plt.ylabel('Angles')
#plt.title(' cos (Angle Between Bodies) vs Time')

plt.tight_layout()
plt.show()
