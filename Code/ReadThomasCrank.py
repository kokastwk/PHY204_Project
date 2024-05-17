import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


class CrankNicolsonSolver:
    def __init__(self, D, f, L, T, J, N, u0):
        self.D = D  # Diffusion coefficient
        self.f = f  # Source term function
        self.L = L  # Spatial domain length
        self.T = T  # Total time
        self.J = J  # Number of spatial steps
        self.N = N  # Number of time steps
        self.dx = L / J  # Spatial step size
        self.dt = T / N  # Time step size
        self.sigma = D * self.dt / (2 * (self.dx ** 2))

        self.x = np.linspace(0, L, J, endpoint=False)  # Spatial grid
        self.t = np.linspace(0, T, N, endpoint=False)  # Temporal grid
        self.A = self.create_matrix_A()
        self.B = self.create_matrix_B()
        self.U = [np.array([u0(i) for i in range(len(self.x))])]  # Initial condition

    def create_matrix_A(self):
        main_diag = (1 + 2 * self.sigma) * np.ones(self.J,dtype=np.complex128)
        off_diag = -(self.sigma)* np.ones(self.J-1,dtype=np.complex128)
        upper_diag = -(self.sigma)* np.ones(self.J-1,dtype=np.complex128)

        A = np.diag(main_diag) + np.diag(off_diag,-1) + np.diag(upper_diag,1)

        return A

    def create_matrix_B(self):
        main_diag = (1 - 2 * self.sigma) * np.ones(self.J,dtype=np.complex128)
        off_diag = (self.sigma)* np.ones(self.J-1,dtype=np.complex128)
        upper_diag = (self.sigma)* np.ones(self.J-1,dtype=np.complex128)
        main_diag[0],main_diag[-1],off_diag[0],off_diag[-1],upper_diag[0],upper_diag[-1] = 0,0,0,0,0,0

        B = np.diag(main_diag) + np.diag(off_diag,-1) + np.diag(upper_diag,1)
        # for dirichlet condition with zero at ends

        #B[0, 1] = 2 * self.sigma

        #B[self.J - 2, self.J - 3] = 2 * self.sigma

        return B

    def simulate(self):

        U = self.U
        U[0][0] = 0
        U[0][-1] =0
        F = [[] for _ in range(N)]
        F[0] = self.dt * np.array([self.f(U[0][i],self.x[i]) for i in range(len(U[0]))],dtype=np.complex128)  # ! take into account x_0 and x_j-1 where f( ) should be 0
        F[0][0] = 0
        F[0][-1] = 0

        B_Un = self.B.dot(U[0])  # Assuming B is a full matrix; adjust if B is also banded
        b = B_Un + F[0]

        U.append([])
        A_csr = csr_matrix(self.A,dtype=np.complex128)
        U[1] = spsolve(A_csr, b)
        U[1][0] = 0
        U[1][-1] =0

        for n in range(1, self.N):  # Iterate through time steps
            F[n] = self.dt * np.array([self.f(U[n][i],self.x[i]) for i in range(len(U[n]))],dtype=np.complex128)
            F[n][0] = 0
            F[n][-1] = 0

            B_Un = self.B.dot(U[n])
            b = B_Un + 3 / 2 * F[n] - 1 / 2 * F[n - 1]

            U.append([])

            # print(b)

            U[n + 1] = spsolve(A_csr, b)
            U[n + 1][0] = 0
            U[n + 1][-1] = 0

        self.U = U


#L = 1 is about a thousand bohr radii and T = 1 is about a thousand evolutions of the H ground state
hbarbym = 1.0
oneoverhbar = 1.0
coloumbbyplanck = 1.0
# Domain and problem parameters
L = 0.5 # Length of the domain
T = 1e-2  # Total time
J = 500  # Number of spatial steps
N = 10000 # Number of time steps
# for J = 1000, stability until 0.5. 0.5 Unstable at J= 3000. J variation stability seems to be increasing with length
D = (0.5j)*hbarbym

# Source term function (set to 0 for simplicity)
def f(u,x):
    E = 1e3
    if x == 0 :
        return 0
    else:
        return ((coloumbbyplanck/x) - E*x)*(u*1.0j)


import numpy as np


def read_file_to_complex_list(filename):
    data_list = []
    try:
        with open(filename, 'r') as file:
            for line in file:
                # Strip the newline character
                line = line.strip()
                try:
                    # Convert the line to a numpy complex128 value
                    item = np.complex128(line)
                except ValueError:
                    print(f"Skipping invalid complex number: {line}")
                    continue
                data_list.append(item)
        print(f"Data successfully read from {filename}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return data_list




ul = read_file_to_complex_list('Statedata2')
# Initialize and solve
def u_0(i):
    global ul
    global L
    global J
    return ul[i]*(i*L/J)

solver = CrankNicolsonSolver(D, f, L, T, J, N, u_0)
solver.simulate()
final_solution = solver.U

def scprobd(u):
    return [(((np.abs(u[k])/(k)))**2) for k in range(len(u))]

def rexpec(u):
    return np.sum([(((np.abs(u[k]))**2)/(k)) for k in range(len(u))])  # r |psi|^2 r^2 dx = (u[k]^2)/(k*L/J)  (L/J)

def psinorm(u):
    return np.sum(scprobd(u))


k = 0
plt.plot()

plt.plot(solver.x, [(final_solution[k][i].real) for i in range(len(final_solution[k]))], linestyle='--')
plt.plot(solver.x, [(final_solution[k][i].imag) for i in range(len(final_solution[k]))], linestyle='--')
plt.plot(solver.x,[np.linalg.norm(final_solution[k][i]) for i in range(len(final_solution[k]))], linestyle='--')
plt.xlabel('x')
plt.ylabel('u(x,T)')
plt.title('u versus space at T = '+str(k*T/N))
plt.show()
print("Done")

k = 999
plt.plot()

plt.plot(solver.x, [(final_solution[k][i].real) for i in range(len(final_solution[k]))], linestyle='--')
plt.plot(solver.x, [(final_solution[k][i].imag) for i in range(len(final_solution[k]))], linestyle='--')
plt.plot(solver.x,[np.linalg.norm(final_solution[k][i]) for i in range(len(final_solution[k]))], linestyle='--')

plt.xlabel('x')
plt.ylabel('u(x,T)')
plt.title('u versus space at T = '+str(k*T/N))
plt.show()
print("Done")

k = -1
plt.plot()

plt.plot(solver.x, [(final_solution[k][i].real) for i in range(len(final_solution[k]))], linestyle='--')
plt.plot(solver.x, [(final_solution[k][i].imag) for i in range(len(final_solution[k]))], linestyle='--')
plt.plot(solver.x,[np.linalg.norm(final_solution[k][i]) for i in range(len(final_solution[k]))], linestyle='--')

plt.xlabel('x')
plt.ylabel('u(x,T)')
plt.title('u versus space at T = '+str(T))
plt.show()
print("Done")

nor = 0
psifinal = []
for k in range(1,len(final_solution[-1])):
    r = k*L/J
    psifinal.append(final_solution[-1][k]/r)
    nor += final_solution[-1][k]/r

psifinal = [i/nor for i in psifinal]

plt.plot()

plt.plot(solver.x[1:], [(psifinal[i].real) for i in range(len(psifinal))], linestyle='--')
plt.plot(solver.x[1:], [(psifinal[i].imag) for i in range(len(psifinal))], linestyle='--')
plt.plot(solver.x[1:],[np.linalg.norm(psifinal[i]) for i in range(len(psifinal))], linestyle='--')
plt.xlabel('x')
plt.ylabel('psi(x,T)')
plt.title('psi versus space at T = '+str(T))
plt.show()
print("Done")




plt.figure(figsize=(10, 5))
plt.plot([x*L/J for x in range(len(final_solution[0]))], scprobd(final_solution[0]), linestyle='--')
plt.xlabel('x')
plt.ylabel('ProbDens')
plt.title('Probability Density at Initial Time')
plt.show()
print("Done")

#plot probd vs x
plt.figure(figsize=(10, 5))
plt.plot([x*L/J for x in range(len(final_solution[-1]))], scprobd(final_solution[-1]), linestyle='--')
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
