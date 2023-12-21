import math
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

def Circle1(X, Y, radius):
 CX = [X + radius * math.cos(i / 100) for i in range(0, 628)]
 CY = [Y + radius * math.sin(i / 100) for i in range(0, 628)]
 return CX, CY
# data of task

t = np.linspace(0, 10, 1001)
phi =  3.5 * np.sin(math.pi/ 6 + 3*t)
psi = -9  * np.sin(3*t +  math.pi / 6)

l = 5
alpha_angle = 30

alpha = 5

r = 8
c = 10

g = 9.81
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











# # # plots(строим графики)
# # VXB = np.diff(X_B)
# # VYB = np.diff(Y_B)
# # WXB = np.diff(VXB)
# # WYB = np.diff(VYB)
# # ax2 = fig.add_subplot(4, 2, 2)
# # ax2.plot(VXB)
# # plt.title('Vx of ball')
# # plt.xlabel('t values')
# # plt.ylabel('Vx values')
# # ax3 = fig.add_subplot(4, 2, 4)
# # ax3.plot(VYB)
# # plt.title('Vy of ball')
# # plt.xlabel('t values')
# # plt.ylabel('Vy values')
# # ax4 = fig.add_subplot(4, 2, 6)
# # ax4.plot(WXB)
# # plt.title('Wx of ball')
# # plt.xlabel('t values')
# # plt.ylabel('Wy values')
# # ax5 = fig.add_subplot(4, 2, 8)
# # ax5.plot(WYB)
# # plt.title('Wy of ball')
# # plt.xlabel('t values')
# # plt.ylabel('Wx values')
# # plt.subplots_adjust(wspace=0.3, hspace=0.7)


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