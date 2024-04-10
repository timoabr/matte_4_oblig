import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

#Lager en liten bølgegradient til seinere
blue_grad = LinearSegmentedColormap.from_list(
    'blue_grad', 
    ['#add8e6', '#0077be', '#004488', '#0000ff', '#0000a0']
)
#Bølgefarten
c = 1

#Initialbetingelser 
def init_condition(x, y):
    return np.exp(-5 * (x**2 + y**2))

#Initialbetingelser for den deriverte, setter til 0 for å få en startposisjon uten bevegelse.
def init_condition_dt(x, y):
    return 0

#Her defieneres grid og skritt størrelse
X_limit = 4
t_limit = 10
X_steps = 50
t_steps = 200

#Oppretter tre arrays og lager skrittlengde basert på mellomrommet mellom stegene. 
x = np.linspace(-X_limit, X_limit, X_steps)
y = np.linspace(-X_limit, X_limit, X_steps)
t = np.linspace(0, t_limit, t_steps)
dx = x[1] - x[0]
dy = y[1] - y[0]
dt = t[1] - t[0]

#Dette stabilitetskriteriet heter "Courant–Friedrichs–Lewy condition" eller CFL forkortet og er funnet frem til ved å søke på internett.
if 2*c * dt / dx <= 1:
    print("Stable step size")
else:
    print("Unstable step size. Adjust X_limit, t_limit, X_steps or t_steps.")

#Bruker np.meshgrid for å lage to nye 2d arrays hvor verdiene fra "x-arrayen" representerer x kordinater i den nye
#"x matrisen" og i "y-matrisen" repeteres hvert element i y-vektoren langs en ny kolonne som matcher x-vektorens lengde.
#x = [0,1,2] og y = [0,1,2] ==> X = [0,1,2][0,1,2][0,1,2] og Y = [0,0,0][1,1,1][2,2,2]

x, y = np.meshgrid(x, y)


#Initialiserer bølge matrisen u
u = np.zeros((X_steps, X_steps, t_steps))

#Legger inn initalkrav i u, altsp verdier for x,y når t=0
u[:, :, 0] = init_condition(x, y)

#Bruker initalkrav for den deriverte til å eventuelt gi en start bølgefart: Dette gjøres ved og gi bølgen en verdi for t=1.
#Denne er satt som standard 0. 
u[:, :, 1] = u[:, :, 0] + dt * init_condition_dt(x, y)

#Bruker sentraldifferansen for å finne den numeriske versjonen av den dobbeltderiverte. 
for n in range(1, t_steps-1):
    for i in range(1, X_steps-1):
        for j in range(1, X_steps-1):
            u[i, j, n+1] = (2*u[i, j, n] - u[i, j, n-1] +
                           (c*dt/dx)**2 * (u[i+1, j, n] - 2*u[i, j, n] + u[i-1, j, n]) +
                           (c*dt/dy)**2 * (u[i, j+1, n] - 2*u[i, j, n] + u[i, j-1, n]))

#VISUALISERING:

#Oppretter en ny figur, dette blir som å starte en tegning på ny side med blanke ark. 
fig = plt.figure(figsize=(10, 7))

#Denne kodelinjen legger til en 3d akse for å muligjøre 3d plotting.
ax = fig.add_subplot(111, projection='3d')

#lager en funkjson som animerer utviklingen av bølgefunksjonen.
def animate_wave(i):
    ax.clear()
    ax.plot_surface(x, y, u[:, :, i], cmap=blue_grad, linewidth=0, antialiased=False)
    ax.set_zlim(-0.5, 0.5)
    ax.set_title(f"Time step {i}")
    plt.pause(0.01)

# Animate
for i in range(t_steps):
    animate_wave(i)

plt.show()
