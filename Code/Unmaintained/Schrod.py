import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

class CrankNicolsonSolver:
    def __init__(self, D, V, f, L, T, J, N, u0):
        self.D = D  # Diffusion coefficient
        self.V = V  # Velocity
        self.f = f  # Source term function
        self.L = L  # Spatial domain length
        self.T = T  # Total time
        self.J = J  # Number of spatial steps
        self.N = N  # Number of time steps
        self.dx = L / J  # Spatial step size
        self.dt = T / N  # Time step size
        self.sigma = D * self.dt / (2 * self.dx ** 2)

        # self.nu = V * self.dt / (4 * self.dx)

        A = D / (4 * self.dx ** 2)

        arr = -1 / (np.arange(J - 1) + 1) * A

        self.nu = V * self.dt / (4 * self.dx) + arr * self.dt

        self.x = np.linspace(0, L, J, endpoint=False)  # Spatial grid
        self.t = np.linspace(0, T, N, endpoint=False)  # Temporal grid
        self.A = self.create_matrix_A()
        self.B = self.create_matrix_B()
        self.U = [np.array([u0(xi) for xi in self.x])]  # Initial condition

    def create_matrix_A(self):
        main_diag = (1 + 2 * self.sigma) * np.ones(self.J)
        off_diag = -(self.sigma + self.nu)
        upper_diag = -(self.sigma - self.nu)

        A = np.diagflat(main_diag) + np.diagflat(off_diag, -1) + np.diagflat(upper_diag, 1)
        #print(self.x)
        #print(len(A[0]))
        #for dirichlet condition with zero at ends
        #for i in range(J):
            #A[0, i] = 0
            #A[i,0] = 0
            #A[J-1,i] = 0
            #A[i,J-1] = 0

        #A[0, 1] = - 2 * self.sigma

        #A[self.J - 2, self.J - 3] = - 2 * self.sigma

        return A

    def create_matrix_B(self):
        main_diag = (1 - 2 * self.sigma) * np.ones(self.J)
        off_diag = (self.sigma + self.nu)
        upper_diag = (self.sigma - self.nu)

        B = np.diagflat(main_diag) + np.diagflat(off_diag, -1) + np.diagflat(upper_diag, 1)
        # for dirichlet condition with zero at ends
        for i in range(J):
            B[0, i] = 0
            B[i, 0] = 0
            B[J - 1, i] = 0
            B[i, J - 1] = 0

        #B[0, 1] = 2 * self.sigma

        #B[self.J - 2, self.J - 3] = 2 * self.sigma

        return B

    def simulate(self):

        U = self.U

        F = [[] for _ in range(N)]
        F[0] = self.dt * np.array([self.f(U[0][i],self.x[i]) for i in range(len(U[0]))])  # ! take into account x_0 and x_j-1 where f( ) should be 0
        F[0][0] = 0
        F[0][-1] = 0

        B_Un = self.B.dot(U[0])  # Assuming B is a full matrix; adjust if B is also banded
        b = B_Un + F[0]

        U.append([])
        U[1] = linalg.solve(self.A, b)

        for n in range(1, self.N - 1):  # Iterate through time steps
            F[n] = self.dt * np.array([self.f(U[n][i],self.x[i]) for i in range(len(U[n]))])
            F[n][0] = 0
            F[n][-1] = 0

            B_Un = self.B.dot(U[n])
            b = B_Un + 3 / 2 * F[n] - 1 / 2 * F[n - 1]

            U.append([])

            # print(b)

            U[n + 1] = linalg.solve(self.A, b)

        self.U = U









hbarbym = 1
oneoverhbar = 1
coloumbbyplanck = 1
# Domain and problem parameters
L = 1  # Length of the domain
T = 1e-3  # Total time
J = 1000  # Number of spatial steps
N = 10000 # Number of time steps
# for J = 1000, stability until 0.5. 0.5 Unstable at J= 3000. J variation stability seems to be increasing with length
D = (0.5j)*hbarbym
V = 0.0

# Source term function (set to 0 for simplicity)
def f(u,x):
    E = 2e-2
    if x == 0 :
        return 0
    else:
        return (E*x*u - coloumbbyplanck*u/x)*(-1j)


# Initialize and solve
def u_0(x):
    a = 1.11211e-2# bohr radius
    return x*(a**(-3/2))*np.exp(-x/a)/np.sqrt(np.pi) #U1,0,0 state

solver = CrankNicolsonSolver(D, V, f, L, T, J, N, u_0)
solver.simulate()
final_solution = solver.U










def probd(u):
    return [((np.abs(u[k])/k)**2)*(J/L) for k in range(1,len(u))]

def rexpec(u):
    return np.sum([((np.abs(u[k]))**2)/k for k in range(1,len(u))])

def psinorm(u):
    return np.sum(probd(u))










#plot psinorm vs t
plt.figure(figsize=(10, 5))
plt.plot([t*T/N for t in range(len(final_solution))], [psinorm(final_solution[t]) for t in range(len(final_solution))], linestyle='--')
plt.xlabel('t')
plt.ylabel('Psinorm(t)')
plt.title('Psinorm(t)')
plt.show()
print("Done")

#plot probd vs x
plt.figure(figsize=(10, 5))
plt.plot([x*L/J for x in range(1,len(final_solution[len(final_solution)-1]))], probd(final_solution[len(final_solution)-1]), linestyle='--')
plt.xlabel('x')
plt.ylabel('ProbDens')
plt.title('Probability Density at Final Time')
plt.show()
print("Done")

#plot expecr vs t
plt.figure(figsize=(10, 5))
plt.plot([t*T/N for t in range(len(final_solution))], [rexpec(final_solution[t]) for t in range(len(final_solution))], linestyle='--')
plt.xlabel('t')
plt.ylabel('Rexpec')
plt.title('Rexpec(t)')
plt.show()
print("Done")










k = 999
# print(len(solver.U))
# Plot the numerical and analytical solutions for comparison
plt.figure(figsize=(10, 5))
plt.plot(solver.x[1:], [(final_solution[k][i].real)/(i*(L/J)) for i in range(1,len(final_solution[k]))], linestyle='--')
plt.plot(solver.x[1:], [(final_solution[k][i].imag)/(i*(L/J)) for i in range(1,len(final_solution[k]))], linestyle='--')
plt.plot(solver.x[1:],[np.linalg.norm(final_solution[k][i])/(i*(L/J)) for i in range(1,len(final_solution[k]))], linestyle='--')
#plt.plot(solver.x, u_analytical, label='Analytical Solution', linestyle='-.')
plt.xlabel('x')
plt.ylabel('psi(x,T)')
plt.title('Wavefunction versus space at T = '+str(T))
plt.show()
print("Done")





k = len(final_solution)-1
# print(len(solver.U))
# Plot the numerical and analytical solutions for comparison
plt.figure(figsize=(10, 5))
#plt.plot(solver.x[1:], [(final_solution[k][i].real) for i in range(1,len(final_solution[k]))], linestyle='--')
plt.plot(solver.x[1:], [(final_solution[k][i].real)/(i*(L/J)) for i in range(1,len(final_solution[k]))], linestyle='--')
plt.plot(solver.x[1:], [(final_solution[k][i].imag)/(i*(L/J)) for i in range(1,len(final_solution[k]))], linestyle='--')
plt.plot(solver.x[1:],[np.linalg.norm(final_solution[k][i])/(i*(L/J)) for i in range(1,len(final_solution[k]))], linestyle='--')
#plt.plot(solver.x, u_analytical, label='Analytical Solution', linestyle='-.')
plt.xlabel('x')
plt.ylabel('psi(x,T)')
plt.title('Wavefunction versus space at T = '+str(T))
plt.show()
print("Done")










import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def generate_animation(x, y):
    # Create a figure and axis
    fig, ax = plt.subplots()
    line, = ax.plot(x, y[0])

    # Function to update the plot for each frame
    def update(frame):
        line.set_ydata(y[frame])
        return line,

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(y), blit=True, interval=1000/60)

    plt.show()

x = solver.x
y = [[np.linalg.norm(final_solution[k][0])/((L/J))]+[np.linalg.norm(final_solution[k][i])/(i*(L/J)) for i in range(1,len(final_solution[k]))] for k in range(len(final_solution))]
generate_animation(solver.x, y)
