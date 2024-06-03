import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def gravitational_force(m1, m2, r):
    G = 8.987551788e9  # Gravitational constant (--> 1/4 pi e_0)
    return G * m1 * m2 / r ** 2


def rk4_step(func, t, y, dt):
    k1 = func(t, y)
    k2 = func(t + 0.5 * dt, y + 0.5 * dt * np.array(k1))
    k3 = func(t + 0.5 * dt, y + 0.5 * dt * np.array(k2))
    k4 = func(t + dt, y + dt * np.array(k3))
    return y + (dt / 6.0) * (np.array(k1) + 2 * np.array(k2) + 2 * np.array(k3) + np.array(k4))


def two_body_equations(t, state):
    x1, y1, z1, vx1, vy1, vz1, x2, y2, z2, vx2, vy2, vz2 = state

    # Calculate distance between the two bodies
    r = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    # Calculate gravitational force magnitude
    force_magnitude = gravitational_force(m1, m2, r)

    # Calculate gravitational force components
    force_x = force_magnitude * (x2 - x1) / r
    force_y = force_magnitude * (y2 - y1) / r
    force_z = force_magnitude * (z2 - z1) / r
    ext_force_x = 1e-10  # q E for E = 6.2 N/C
    ext_force_y = 0
    ext_force_z = 0
    # Accelerations
    M1 = 1.67262192e-27  # proton mass
    M2 = 9.1093837e-31  # electron mass
    ax1 = (force_x / M1) + ext_force_x / M1
    ay1 = (force_y / M1) + ext_force_y / M1
    az1 = (force_z / M1) + ext_force_z / M1
    ax2 = (force_x / M2) + ext_force_x / M2  # -(force_x / M2) - ext_force_x/M2
    ay2 = (force_y / M2) + ext_force_y / M2  # -(force_y / M2) - ext_force_y/M2
    az2 = (force_z / M2) + ext_force_z / M2  # -(force_z / M2) - ext_force_z/M2

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
        angles[step] = np.dot(positions[step, 1], positions[step, 0]) / (
                    np.linalg.norm(positions[step, 1]) * np.linalg.norm(positions[step, 0]))  # cosine
    return positions, distances, angles

def two_body_equations_i(t, state):
    xi1, yi1, zi1, vxi1, vyi1, vzi1, xi2, yi2, zi2, vxi2, vyi2, vzi2 = state

    # Calculate distance between the two bodies
    r_i = np.sqrt((xi2 - xi1)**2 + (yi2 - yi1)**2 + (zi2 - zi1)**2)

    # Calculate gravitational force magnitude
    force_magnitude = gravitational_force(m1, m2, r_i)

    # Calculate gravitational force components
    force_x = force_magnitude * (xi2 - xi1) / r_i
    force_y = force_magnitude * (yi2 - yi1) / r_i
    force_z = force_magnitude * (zi2 - zi1) / r_i
    ext_force_x = 0 #1e-18 # q E for E = 6.2 N/C
    ext_force_y = 0
    ext_force_z = 0
    # Accelerations
    M1 = 1.67262192e-27 # proton mass
    M2 = 9.1093837e-31 #electron mass
    axi1 = (force_x / M1) + ext_force_x/M1
    ayi1 = (force_y / M1) + ext_force_y/M1
    azi1 = (force_z / M1) + ext_force_z/M1
    axi2 = (force_x / M2) + ext_force_x/M2 #-(force_x / M2) - ext_force_x/M2
    ayi2 = (force_y / M2) + ext_force_y/M2 #-(force_y / M2) - ext_force_y/M2
    azi2 = (force_z / M2) + ext_force_z/M2 #-(force_z / M2) - ext_force_z/M2

    return [vxi1, vyi1, vzi1, axi1, ayi1, azi1, vxi2, vyi2, vzi2, axi2, ayi2, azi2]

def simulate_two_body_i(m1, m2, r_initial, v_initial, num_steps, dt):
    state = [0, 0, 0, 0, 0, 0, r_initial, 0, 0, 0, v_initial, 0]  # Initial state

    positions = np.zeros((num_steps, 2, 3))  # Array to store positions of both bodies
    distances = np.zeros(num_steps)  # Array to store distances between the bodies

    for step in range(num_steps):
        t = step * dt
        state = rk4_step(two_body_equations_i, t, state, dt)
        positions[step] = np.array([[state[0], state[1], state[2]], [state[6], state[7], state[8]]])
        distances[step] = np.linalg.norm(positions[step, 1] - positions[step, 0])

    return positions, distances



# Parameters
m1 = 1.60217663 * 1e-19  # Mass of body 1 (e.g., Earth) in kg --> proton charge
m2 = -1.60217663 * 1e-19  # Mass of body 2 (e.g., Moon) in kg  --> electron charge
r_initial = 5.3e-11  # Initial distance between the bodies in meters --> bohr radius
v_initial = 2e6  # 2e6 # Initial velocity of body 2 in m/s
num_steps = 100000  # Number of time steps
dt = 10e-20  # Time step size in seconds

# Simulate two-body system
positions, distances, angles = simulate_two_body(m1, m2, r_initial, v_initial, num_steps, dt)
positions_i, distances_i = simulate_two_body_i(m1, m2, r_initial, v_initial, num_steps, dt)


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
ax.plot(np.arange(num_steps) * dt, angles, color="red")
ax.set_xlabel('Time (s)')
ax.set_ylabel('cos(Angles)')
ax.set_title(' cos(Angle Between Bodies) vs Time')

# plt.plot(np.arange(num_steps) * dt, angles)
# plt.xlabel('Time (s)')
# plt.ylabel('Angles')
# plt.title(' cos (Angle Between Bodies) vs Time')

plt.tight_layout()
plt.show()

# Plot positions in 3D
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.plot(positions_i[:, 0, 0], positions_i[:, 0, 1], positions_i[:, 0, 2], label='Body 1')
ax.plot(positions_i[:, 1, 0], positions_i[:, 1, 1], positions_i[:, 1, 2], label='Body 2')
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Z Position (m)')
ax.set_title('Two Body Problem Simulation With No Field (3D)')
ax.legend()

# Plot distance versus time
plt.subplot(122)
plt.plot(np.arange(num_steps) * dt, distances_i)
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.title('Distance Between Bodies vs Time')

plt.tight_layout()
plt.show()

#print(positions) #ignore
#print(positions_i[0][0]) #ignore


diff_positions =[]
for i in range(len(positions)):
    diff_positions.append([[positions[i][0][0]-positions_i[i][0][0],positions[i][0][1]-positions_i[i][0][1],positions[i][0][2]-positions_i[i][0][2]],[positions[i][1][0]-positions_i[i][1][0],positions[i][1][1]-positions_i[i][1][1],positions[i][1][2]-positions_i[i][1][2]]])

def sum_corresponding_elements(list_of_lists):
  """
  This function takes a list of lists, all of size 3, and returns a list containing the sum of corresponding elements across all sublists.
  Args:
      list_of_lists: A list containing sublists of size 3.
  Returns:
      A list containing the sum of corresponding elements across all sublists.
  """
  # Initialize three variables to store the sums
  sum_first = 0
  sum_second = 0
  sum_third = 0

  # Iterate through each sublist
  for sublist in list_of_lists:

    # Add corresponding elements to their respective sums
    sum_first += sublist[0]
    sum_second += sublist[1]
    sum_third += sublist[2]

  # Return the list with the sums
  return [sum_first, sum_second, sum_third]

def time_average(distances_i):
    initpos = distances_i[0]
    print(distances_i[0])
    Ts = []
    for j in range(len(distances_i)):
        if abs(distances_i[j] - initpos) < 1e-16:
            Ts.append(j)

    return Ts

Ts = time_average(distances_i)[1:]

#the output of code above was: [0, 1214, 1215, 1216, 2430, 2431, 2432, 3645, 3646, 3647, 4861, 4862, 4863, 6077, 6078, 6079, 7293, 7294, 7295, 8508, 8509, 8510, 9724, 9725, 9726, 10940, 10941, 10942, 12156, 12157, 12158, 13371, 13372, 13373, 13374, 14587, 14588, 14589, 15803, 15804, 15805, 17019, 17020, 17021, 18235, 18236, 18237, 19450, 19451, 19452, 20666, 20667, 20668, 21882, 21883, 21884, 23098, 23099, 23100, 24313, 24314, 24315, 25529, 25530, 25531, 26745, 26746, 26747, 27961, 27962, 27963, 29176, 29177, 29178, 29179, 30392, 30393, 30394, 31608, 31609, 31610, 32824, 32825, 32826, 34040, 34041, 34042, 35255, 35256, 35257, 36471, 36472, 36473, 37687, 37688, 37689, 38903, 38904, 38905, 40118, 40119, 40120, 41334, 41335, 41336, 42550, 42551, 42552, 43766, 43767, 43768, 44981, 44982, 44983, 44984, 46197, 46198, 46199, 47413, 47414, 47415, 48629, 48630, 48631, 49845, 49846, 49847, 51060, 51061, 51062, 52276, 52277, 52278, 53492, 53493, 53494, 54708, 54709, 54710, 55923, 55924, 55925, 57139, 57140, 57141, 58355, 58356, 58357, 59571, 59572, 59573, 60786, 60787, 60788, 60789, 62002, 62003, 62004, 63218, 63219, 63220, 64434, 64435, 64436, 65650, 65651, 65652, 66865, 66866, 66867, 68081, 68082, 68083, 69297, 69298, 69299, 70513, 70514, 70515, 71728, 71729, 71730, 72944, 72945, 72946, 74160, 74161, 74162, 75376, 75377, 75378, 76591, 76592, 76593, 76594, 77807, 77808, 77809, 79023, 79024, 79025, 80239, 80240, 80241, 81455, 81456, 81457, 82670, 82671, 82672, 83886, 83887, 83888, 85102, 85103, 85104, 86318, 86319, 86320, 87533, 87534, 87535, 88749, 88750, 88751, 89965, 89966, 89967, 91181, 91182, 91183, 92397, 92398, 92399, 93612, 93613, 93614, 94828, 94829, 94830, 96044, 96045, 96046, 97260, 97261, 97262, 98475, 98476, 98477, 99691, 99692, 99693]

tt = Ts[0]
ll = []
Tsrep = []
for i in Ts:
    if abs(i - tt) <100:
        ll.append(i)
    else:
        Tsrep.append(int(sum(ll)/len(ll)))
        ll = [i]
        tt = i
#print(Tsrep)
Ts = Tsrep

Avs = []
Avs_i = []
blah = [0,0,0]
blah_i = [0,0,0]
for j in range(len(positions)):
    if j in Ts:
        jj = Ts.index(j)
        delt = (Ts[jj] - Ts[jj-1])
        blah[0] = (blah[0] + positions[j][0][0])/delt
        blah[1] = (blah[1] + positions[j][0][1])/delt
        blah[2] = (blah[2] + positions[j][0][2])/delt
        Avs.append(blah)
        blah = [0,0,0]

        blah_i[0] = (blah_i[0] + positions_i[j][0][0]) / delt
        blah_i[1] = (blah_i[1] + positions_i[j][0][1]) / delt
        blah_i[2] = (blah_i[2] + positions_i[j][0][2]) / delt
        Avs_i.append(blah_i)
        blah_i = [0, 0, 0]
    else:
        blah[0] = (blah[0] + positions_i[j][0][0])
        blah[1] = (blah[1] + positions_i[j][0][1])
        blah[2] = (blah[2] + positions_i[j][0][2])
        blah_i[0] = (blah_i[0] + positions_i[j][0][0])
        blah_i[1] = (blah_i[1] + positions_i[j][0][1])
        blah_i[2] = (blah_i[2] + positions_i[j][0][2])

Avs = Avs[1:]
Avs_i = Avs_i[1:]
#print(Avs,Avs_i)
Dels = []

for i in range(len(Avs)):
    Dels.append([Avs[i][0]-Avs_i[i][0],Avs[i][1]-Avs_i[i][1], Avs[i][2]-Avs_i[i][2]])

print(Dels)

def plot_trajectory_and_norm(sublists, dt):
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    # Extract x, y, z coordinates from sublists
    x_coords = [sublist[0] for sublist in sublists]
    y_coords = [sublist[1] for sublist in sublists]
    z_coords = [sublist[2] for sublist in sublists]

    # Plot trajectory
    ax1.plot(x_coords, y_coords, z_coords)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Separation Trajectory')

    # Calculate norm for each sublist
    norms = [np.linalg.norm(sublist) for sublist in sublists]

    # Generate time array
    time = np.arange(len(sublists)) * dt

    # Plot norm as a function of time
    ax2.plot(time, norms, color='blue')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Norm')
    ax2.set_title('Sep Distance vs Time')

    plt.tight_layout()
    plt.show()

plot_trajectory_and_norm(Dels,dt)
