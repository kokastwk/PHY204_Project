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


# Gaussian initial condition parameters
U0 = 2
sigma = 5
x0 = 50

class Gaussian:
    """Gaussian function"""
    def __init__(self, width, amplitude, x0):
        """Create Gaussian with the given width (standard deviation) and amplitude, centered at x = x0"""
        self.width = width
        self.amplitude = amplitude
        self.x0 = x0

    def __call__(self, x):
        """Evaluate the Gaussian function at x"""
        return self.amplitude * np.exp(-((x - self.x0) ** 2) / ( (self.width ** 2)))

gauss = Gaussian(sigma, U0, x0)
# Domain and problem parameters
L = 10000  # Length of the domain
T = 200  # Total time
J = 10000  # Number of spatial steps
N = 1000  # Number of time steps

# Generating 100 points from -3 to 3
x = np.linspace(0, 100, 600)
# Gaussian function


'''# Plotting
plt.figure(figsize=(8, 5))
plt.plot(x, gauss(x), label='Standard Gaussian')
plt.title('Standard Gaussian Distribution')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

D = 1j
V = 0.0


# Source term function (set to 0 for simplicity)
def f(u,x):
    E = 10e-12
    coulomb = 2.3070775389e-28
    if x == 0 :
        return 0
    else:
        return (E*x*u - coulomb*u/x)*(-1j)


# Initialize and solve
def u_0(x):
    a = 1# bohr radius
    return x*(a**(-3/2))*np.exp(-x/a)/np.sqrt(np.pi) #U1,0,0 state

solver = CrankNicolsonSolver(D, V, f, L, T, J, N, u_0)

# print(len(solver.U))

solver.simulate()

final_solution = solver.U
k = 999

# print(len(solver.U))
# Plot the numerical and analytical solutions for comparison
plt.figure(figsize=(10, 5))
plt.plot(solver.x, [final_solution[k][i].real for i in range(len(final_solution[k]))], linestyle='--')
plt.plot(solver.x, [final_solution[k][i].imag for i in range(len(final_solution[k]))], linestyle='--')
plt.plot(solver.x, [np.linalg.norm(final_solution[k][i]) for i in range(len(final_solution[k]))], linestyle='--')
#plt.plot(solver.x, u_analytical, label='Analytical Solution', linestyle='-.')
plt.xlabel('x')
plt.ylabel('u(x,T)')
plt.title('Comparison of Numerical and Analytical Solutions')
plt.show()