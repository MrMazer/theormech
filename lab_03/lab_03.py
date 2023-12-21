import math
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def odesys(y, t, m1, m2, r, l, a, c , g):
    dy = np.zeros(4)
    dy[0] = y[2]
    dy[1] = y[3]

    a11 = m1*r*r + 2 * m2 * a * a
    a12 = 2 * m2 * a * l * np.cos(y[0] + y[1])
    a21 = m2*l*a*np.cos(y[0] + y[1])
    a22 = m2*l*l

    b1 = 2*m2*a*l*y[3]*y[3]*np.sin(y[0] + y[1]) + 2*m2*g*a*np.sin(y[0]) - 2*c*(y[0] + y[1])
    b2 = m2*l*a*y[2]*y[2]*np.sin(y[0] + y[1]) - m2*g*l*np.sin(y[1]) - c*(y[0] + y[1])

    dy[2] = (b1*a22 - b2*a12)/(a11*a22 - a12*a21)
    dy[3] = (b2*a11 - b1*a21)/(a11*a22 - a12*a21)

    return dy


# data of task
m1 = 1
m2 = 1
r = 1
l = 3
a =0.5
c = 1
g = 9.8

steps = 1001
t_fin  = 10

t = np.linspace(0, t_fin, steps)

phi0 = np.pi
psi0 = 0

dphi0 = 0
dpsi0 = 0

y0 = [phi0, psi0, dphi0, dpsi0]

Y = odeint(odesys, y0, t, ( m1, m2, r, l, a, c , g))
phi = Y[:, 0]
psi = Y[:, 1]
dphi = Y[:,2]
dpsi = Y[:, 3]

ddphi = [odesys(y, t, m1, m2, r, l, a, c, g)[2] for y, t in zip(Y, t)]
ddpsi = [odesys(y, t, m1, m2, r, l, a, c, g)[3] for y, t in zip(Y, t)]

Nox = (m1 + m2) * g - m2 * (a*(ddphi*np.sin(phi) + dphi * dphi * np.cos(phi)) - l*(ddpsi*np.sin(psi) + dpsi*dpsi * np.cos(psi)))

#code from old lab
def Circle1(X, Y, radius):
 CX = [X + radius * math.cos(i / 100) for i in range(0, 628)]
 CY = [Y + radius * math.sin(i / 100) for i in range(0, 628)]
 return CX, CY

l = 15
alpha_angle = 30

alpha = 5

r = 8
c = 1

g = 9.8

t = np.linspace(0, 10, 1001)
#задаем точки системы и строим линии
X_start = 0
Y_start = 0

X_O = X_start 
Y_O = Y_start 
X_A = X_O - alpha * np.sin(phi) 
Y_A = Y_O + alpha * np.cos(phi) 

X_B = X_A - l * np.sin(psi) 
Y_B = Y_A - l * np.cos(psi) 

fig = plt.figure(figsize=[10, 7])
ax = fig.add_subplot(1, 1, 1)
ax.axis('equal')
ax.set(xlim=[-25, 25], ylim=[-25, 25])

Point_O = ax.plot(X_O, Y_O, marker='o', color='black')[0]
Point_A = ax.plot(X_A[0], Y_A[0], marker='o', color='black')[0]
Point_B = ax.plot(X_B[0], Y_B[0], marker='o', markersize=20, color='black')[0]

circle1 = ax.plot(*Circle1(X_O, Y_O, r), 'red') # main circle


triangle, = ax.plot([-0.5, 0, 0.5],
 [-1, 0, -1], color='black')
line_tr = ax.plot([- 0.5, 0.5], [-1, -1],
 color='black')[0]



Line_AB = ax.plot([X_A[0], X_B[0]], [Y_A[0], Y_B[0]], color='black', linewidth=3)[0]

# spiral spring
Nv = 2.6
R1 = 0.2
R2 = 5
thetta = np.linspace(0, Nv * 6.28 - psi[0], 1001)
X_SpiralSpr = -(R1 * thetta * (R2 - R1) / thetta[-1]) * np.sin(thetta)
Y_SpiralSpr = (R1 * thetta * (R2 - R1) / thetta[-1]) * np.cos(thetta)
Drawed_Spiral_Spring = ax.plot(X_SpiralSpr + X_A[0], Y_SpiralSpr + Y_A[0], color='black')[0]



#рисуем графики

fig_for_graphs = plt.figure(figsize=[13,7])

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 1)
ax_for_graphs.plot(t, phi, color='Blue')
ax_for_graphs.set_title("phi(t)")
ax_for_graphs.set(xlim=[0,t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 3)
ax_for_graphs.plot(t, psi, color='Red')
ax_for_graphs.set_title("psi(t)")
ax_for_graphs.set(xlim=[0,t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 2)
ax_for_graphs.plot(t, Nox, color='Black')
ax_for_graphs.set_title("Nox(t)")
ax_for_graphs.set(xlim=[0,t_fin])
ax_for_graphs.grid(True)


def anima(i): #функция анимации
 Point_A.set_data(X_A[i], Y_A[i])
 Point_B.set_data(X_B[i], Y_B[i])
 Line_AB.set_data([X_A[i], X_B[i]], [Y_A[i], Y_B[i]])

 thetta = np.linspace(0, Nv * 6 - psi[i], 100)
 X_SpiralSpr = -(R1 * thetta * (R2 - R1) / thetta[-1]) * np.sin(thetta)
 Y_SpiralSpr = (R1 * thetta * (R2 - R1) / thetta[-1]) * np.cos(thetta)
 Drawed_Spiral_Spring.set_data(X_SpiralSpr + X_A[i], Y_SpiralSpr +
 Y_A[i])

 return  [Point_A, Point_B, Line_AB, Drawed_Spiral_Spring]


anim = FuncAnimation(fig, anima, frames=1000, interval=10)

plt.show()