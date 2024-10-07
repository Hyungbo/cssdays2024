#
# Animation of layered coordination of heterogeneous agents
#
# Hyungbo Shim and Hyeonyeong Jang 
# Sept 23, 2024
#
# version 1.0
#
## Usage:
# space = toggle coupling on / off
# q = quit
##
# Behavior changes every time because of random generations
#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.spatial.distance import pdist, squareform

# Parameters
N = 11     # Numbers of agents, must be odd number
AL = 10    # Arena Length: AL x AL m
MD = 2     # Minimum distance from other agents
CR = 5     # Connecting range: within this range, two agents are connected

Ani_Skip = 30            # Animation skip count for speed 
PT = 0.02  # pause time for animation
SP = 200   # Sun lighting period
DF = False # display progress time
MSF = False    # Movie save flag
MFN = 'test'   # avi file name

SF = False # Synchronization flag; alternating whenever 'space' pressed
QF = False # If true, quit simulation

# Function to calculate distances and Laplacian matrix
def pdist2(X, R):
    # Compute the pairwise Euclidean distance matrix
    D = squareform(pdist(X.T))
    L = np.zeros((X.shape[1], X.shape[1]))
    
    # Create Laplacian matrix by connecting agents within range R
    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            if D[i, j] < R:
                L[i, j] = L[j, i] = -1
    L[np.diag_indices(X.shape[1])] = -L.sum(axis=1)
    
    return D, L

# Function to runge-kutta integration (RK4)
def RK4(dt, f, x, u2, u3, sun):
    k1 = f(x, u2, u3, sun)
    k2 = f(x + 0.5 * dt * k1, u2, u3, sun)
    k3 = f(x + 0.5 * dt * k2, u2, u3, sun)
    k4 = f(x + dt * k3, u2, u3, sun)
    
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

# Define Agent class
class Agent:
    def __init__(self, position, mu, nu):
        self.position = position
        self.mu = mu
        self.nu = nu
        self.state = np.array([-1 + 2 * np.random.rand(), -1 + 2 * np.random.rand(), np.random.rand()])
    
    def dynamics(self, x, u2, u3, sun):
        return np.array([
            -x[0] + x[1],
            (1 - self.mu * (x[0] ** 2 - 1)) * (-x[0] + x[1]) - self.nu * x[0] + u2,
            np.sign(sun - x[2]) + u3
        ])

# Random position generation
flag = True
while flag:
    positions = AL * np.random.rand(2, N)
    D, L = pdist2(positions, CR)
    if np.all(D[np.triu_indices(N, k=1)] > MD):
        flag = False

# Define agents
agents = []
for i in range(N):
    pp = 0.1
    if np.random.rand() < 0.5:
        mu = 0.01
        nu = 1.2 * (1 + (-pp + 2 * pp * np.random.rand()))
    else:
        mu = 2
        nu = 1 * (1 + (-pp + 2 * pp * np.random.rand()))
    
    agent = Agent(positions[:, i], mu, nu)
    agents.append(agent)

# Inverse of theta function
def itheta(x, mu):
    return (4 - 3 * np.tanh(np.tanh(mu - 1.025) * 5 * (x - 0.5)))

# Hide menu bar
plt.rcParams['toolbar'] = 'None'

# Animation setup
fig, ax = plt.subplots()
ax.set_xlim(-1, AL + 1)
ax.set_ylim(-1, AL + 1)

def key_press(event):
    global SF, QF
    if event.key == ' ':
        SF = not SF
    if event.key == 'q':
        QF = True

fig.canvas.mpl_connect('key_press_event', key_press)

# Simulation loop
time = 0
dt = 0.01
ani_count = 0

while not QF:
    lw = min(time % SP, SP - time % SP)
    lp = 2 * lw / SP * AL

    # Animation and Drawing
    if ani_count > Ani_Skip:
        ax.clear()
        ax.set_xlim(-1, AL + 1)
        ax.set_ylim(-1, AL + 1)
        
        # Draw the ground and sun light
        ground = Rectangle((-1, -1), AL + 2, AL + 2, color='gray', ec='none')
        sunlight = Rectangle((-1, -1), lp + 1, AL + 2, color='yellow', ec='none')
        ax.add_patch(ground)
        ax.add_patch(sunlight)

        for i, agent in enumerate(agents):
            w = 0.2
            h = abs(agent.state[1] / 5 * (3 * w))

            # Draw node and breath
            node = Rectangle((agent.position[0] - w, agent.position[1] - 3 * w), 2 * w, 6 * w, ec='black', fc='white')
            breath = Rectangle((agent.position[0] - w, agent.position[1] - h), 2 * w, 2 * h, fc='blue', ec='none')
            ax.add_patch(node)
            ax.add_patch(breath)
        
        plt.pause(PT)
        ani_count = 0

    # Synchronization and agent state update
    for i in range(N):
        if SF:
            u2 = u3 = 0
            for j in range(N):
                if L[i, j] < 0:
                    u2 += agents[j].state[1] - agents[i].state[1]
                    u3 += agents[j].state[2] - agents[i].state[2]
            u2 = 1 * u2 * itheta(agents[i].state[2], agents[i].mu)
            u3 = 5 * u3
        else:
            u2 = u3 = 0

        agents[i].state = RK4(dt, agents[i].dynamics, agents[i].state, u2, u3, agents[i].position[0] < lp)
    
    time += dt
    ani_count += 1

plt.show()
