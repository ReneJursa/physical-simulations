# Copyright (c) 2018,
# René Jursa
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import time
import sys
import numpy as np
from numpy import linspace, zeros, exp, asarray, pi, cos, sin
import matplotlib.pyplot as plt
from visual import *
from visual.graph import *

# This program simulates the motion of an object inside a circular cone.
# From the Lagrange equation one gets the following equations of motion:   
#      (1 + (tan alpha)^2) * dr^2/dt^2 - r * (dphi/dt)^2 + g * (tan alpha) = 0
#                                    r * dphi^2/dt^2 + 2 * dr/dt * dphi/dt = 0
# g is the gravity acceleration. alpha is the angle between the surface of
# the circular cone and the horizontal plane. phi is the angle of the object in the
# horizontal xz-plane and r is the distance of the object from the origin in the xz-plane.
# The equations of motion are solved numerically by the Euler-Cromer method.

# values of constants
g = 9.81    # gravity acceleration
m = 1.0     # mass of body 
angle = 30.0 * 2.0 * np.pi / 360.0  

# values for numerical calculation
T = 30.0    # time duration of the motion in seconds
dt = 0.001  # timestep 

N_t = int(round(T/dt))
print("N_t: {0}".format(N_t))
t = linspace(0, N_t*dt, N_t+1)

ph = zeros(N_t+1)    # phi 
om = zeros(N_t+1)    # dphi/dt
rxz = zeros(N_t+1)   # r 
vxz = zeros(N_t+1)   # dr/dt

ux = zeros(N_t+1)    # x-component of position vector u
vx = zeros(N_t+1)    # x-component of velocity vector v
uy = zeros(N_t+1)    # y-component of position vector u
vy = zeros(N_t+1)    # y-component of velocity vector v
uz = zeros(N_t+1)    # z-component of position vector u
vz = zeros(N_t+1)    # z-component of velocity vector v

ax = zeros(N_t+1)    # x-component of acceleration vector 
ay = zeros(N_t+1)    # y-component of acceleration vector 
az = zeros(N_t+1)    # z-component of acceleration vector 
a_abs = zeros(N_t+1) # acceleration

vabs = zeros(N_t+1)
ds_arr = zeros(N_t+1)
Ekin = zeros(N_t+1)  # kinetic energy
Epot = zeros(N_t+1)  # potential energy
Etot = zeros(N_t+1)  # total energy  
Emech = zeros(N_t+1) # mechanical energy

# Initial conditions
ux[0] = 8.0
uz[0] = 0.0
R_xz = np.sqrt(ux[0]**2 + uz[0]**2)
uy[0] = R_xz* np.tan(angle)
vx[0] = 0.0
vy[0] = 0.0
vz[0] = 4.0
vabs[0] = sqrt(vx[0]**2 + vy[0]**2 + vz[0]**2)

Beta = np.arccos( abs(ux[0]) / R_xz ) 
if 0.0 <= ux[0] and 0.0 <= uz[0]: 
    phi = Beta
elif ux[0] < 0.0 and 0.0 <= uz[0]: 
    phi = np.pi - Beta
elif ux[0] < 0.0 and uz[0] < 0.0:
    phi = np.pi + Beta
elif 0.0 < ux[0] and uz[0] < 0.0: 
    phi = 2.0*np.pi - Beta
else:
    print("phi nicht berechenbar")

a_cent = (vx[0]**2 + vz[0]**2)/ R_xz
a_diff = g * np.sin(angle) - a_cent * np.cos(angle)
an_xz = a_diff * np.cos(angle)
vxz[0] = dt * an_xz

ph[0] = phi
rxz[0] = np.sqrt(ux[0]**2 + uz[0]**2)
if ph[0] != 0.0:
    om[0] = ( vxz[0] * np.cos(ph[0]) - vx[0] )/(rxz[0] * np.sin(ph[0]))
else:
    om[0] = (vz[0] - vxz[0] * np.sin(ph[0]))/(rxz[0] * np.cos(ph[0]))

axz = ( rxz[0] * om[0]**2 - g * np.tan(angle) ) / (1.0 + (np.tan(angle))**2)
ax[0] = axz*np.cos(ph[0]) - 2.0* vxz[0]*om[0]*np.sin(ph[0]) - rxz[0]*om[0]**2*np.cos(ph[0])
az[0]  = axz*np.sin(ph[0]) - 2.0* vxz[0]*om[0]*np.cos(ph[0]) - rxz[0]*om[0]**2*np.sin(ph[0])
ay[0] = axz*np.tan(angle)
a_abs[0] = np.sqrt(ax[0]**2 + ay[0]**2 + az[0]**2)

n = 0
print("Anfangszeit t={0:6.5}s x={1:10.9}m, y={2:10.9}m und z={3:10.9}m".format(t[n],ux[n],uy[n],uz[n]))
print("Geschwindigkeit vx={0:10.9}m/s, vy={1:10.9}m/s vz={2:10.9}m/s und v={3:10.9}m/s".format(vx[n],vy[n],vz[n],vabs[n]))
print("Beschleunigung ax={0:10.9}m/s², ay={1:10.9}m/s² az={2:10.9}m/s² und a={3:10.9}m/s²".format(ax[n],ay[n],az[n],a_abs[n]))

Ekin[0] = 0.5*m*( (vx[0])**2 + (vy[0])**2 + (vz[0])**2 )
Epot[0] = m * g * uy[0]                     # reference value Epot,R = 0 at y = 0
Emech[0] = Ekin[0] + Epot[0]
Etot[0] = Emech[0] 
Et_initial = Emech[0] 

print("Anfangswerte für t={0:6.3} Ekin={1:10.9}J  Epot={2:10.9}J  Emech={3:10.9}J  EG={4:10.9}J".
      format(t[0],Ekin[0],Epot[0],Emech[0],Et_initial)) 

s = 0.0         # for summation of differential path elements
mean_abs_diff_Et = 0.0
n_max = 0
max_vel = vabs[0]
n_max_vel = 0
max_accel = a_abs[0]
n_max_accel = 0

starttime = time . time ()
for n in range(N_t):

    # one step forward using Euler-Cromer method
    vxz[n+1] = vxz[n] + dt*( rxz[n] * om[n]**2 - g * np.tan(angle) )/(1.0 + (np.tan(angle))**2)
    rxz[n+1] = rxz[n] + dt*vxz[n+1]
    om[n+1] = om[n] - dt*2.0*vxz[n+1]*om[n] / rxz[n+1]
    ph[n+1] = ph[n] + dt*om[n+1]
    
    vx[n+1] = vxz[n+1]*np.cos(ph[n+1]) - rxz[n+1]*om[n+1]*np.sin(ph[n+1])
    ux[n+1] = rxz[n+1]*np.cos(ph[n+1])

    vz[n+1] = vxz[n+1]*np.sin(ph[n+1]) + rxz[n+1]*om[n+1]*np.cos(ph[n+1])
    uz[n+1] = rxz[n+1]*np.sin(ph[n+1])

    vy[n+1] = vxz[n+1]*np.tan(angle)
    uy[n+1] = rxz[n+1]*np.tan(angle)

    axz = ( rxz[n+1] * om[n+1]**2 - g * np.tan(angle) ) / (1.0 + (np.tan(angle))**2)
    ax[n+1] = axz*np.cos(ph[n+1]) - 2.0* vxz[n+1]*om[n+1]*np.sin(ph[n+1]) - rxz[n+1]*om[n+1]**2*np.cos(ph[n+1])
    az[n+1] = axz*np.sin(ph[n+1]) + 2.0* vxz[n+1]*om[n+1]*np.cos(ph[n+1]) - rxz[n+1]*om[n+1]**2*np.sin(ph[n+1])
    ay[n+1] = axz*np.tan(angle)

    vabs[n+1] = sqrt(vx[n+1]**2 + vy[n+1]**2 + vz[n+1]**2)
    a_abs[n+1] = np.sqrt(ax[n+1]**2 + ay[n+1]**2 + az[n+1]**2)

    dx = ux[n+1] - ux[n]
    dy = uy[n+1] - uy[n]
    dz = uz[n+1] - uz[n]
    ds = sqrt(dx**2 + dy**2 + dz**2)  
    ds_arr[n+1] = ds 
    s += ds
   
    Ekin[n+1] = 0.5*m*( (vx[n+1])**2 + (vy[n+1])**2 + (vz[n+1])**2 )
    Epot[n+1] = m * g * uy[n+1]
    Emech[n+1] = Ekin[n+1] + Epot[n+1]
    Etot[n+1] = Emech[n+1] 
    mean_abs_diff_Et += abs(Etot[n+1] - Et_initial)

    if vabs[n+1] > max_vel:
       max_vel = vabs[n+1]
       n_max_vel = n+1

    if a_abs[n+1] > max_accel:
       max_accel = a_abs[n+1]
       n_max_accel = n+1

    n_max = n + 1
   
    endtime = time . time ()
difftime = endtime - starttime
print("Rechenzeit für numerische Berechnung der Bewegung: {0:10.8}s".format(difftime))

# Visualization
scene = display(title='motion of a body in a circular cone',
                x=0, y=0, width=800, height=1200,
                center=(0.0,0.0,0.0),range=(12.0,4.0,6.0))
scene.background = color.gray(0.3)
scene.ambient = color.gray(0.8)
scene.forward=vector( 0.0,-0.8, -1.0 )

L = 0.7
H = 0.7
W = 0.7
hy = 6.0*np.tan(angle)

p = paths.circle( pos=(0,6.0*np.tan(angle)/2.0-H/(2.0*cos(angle)),0), radius=5.0, up=(0,1,0) )

tri = Polygon( [(-2,0), (4,hy), (4,0)] )

extrusion(pos=p, shape=tri, color=color.yellow,up=(0,1,0))

uxp = ux[0]
uyp = uy[0]
uzp = uz[0]

phi = ph[0]
n_xz = np.sin(angle)
unx_sn = -n_xz * np.cos(phi)
uny_sn = np.cos(angle)
unz_sn = -n_xz * np.sin(phi)
n_mag = np.sqrt(unx_sn**2 + uny_sn**2 + unz_sn**2)
unx_sn /= n_mag
uny_sn /= n_mag
unz_sn /= n_mag

tx = vx[0]
ty = vy[0]
tz = vz[0]
t_mag = np.sqrt(tx**2 + ty**2 + tz**2)
utx_sn = tx/t_mag
uty_sn = ty/t_mag
utz_sn = tz/t_mag
block = box(pos=(uxp,uyp,uzp),axis=(L*utx_sn,H*uty_sn,W*utz_sn),size=(L,H,W),up=vector(unx_sn,uny_sn,unz_sn),
            color=color.black,opacity=0.95,
            make_trail=True, trail_type="points",interval=200, retain=10000)
block.velocity = vector(vx[0],vy[0],vz[0])

fgcolor=color.black; bgcolor=color.white

# energy diagramm
ymax = 60.0
labeloffset = 0.05*Etot[0] 
Egraph = gdisplay(x=800, y=000, width=600, height=450, 
             title='t-E-diagram', xtitle='t/s', ytitle='E/J', 
             xmax=T, xmin=0.0, ymax=ymax, ymin=-5, 
             foreground=fgcolor, background=bgcolor)
label(display=Egraph.display, pos=(5,ymax-labeloffset), text=("yellow: Epot"))
label(display=Egraph.display, pos=(12,ymax-labeloffset), text=("blue: Ekin"))
label(display=Egraph.display, pos=(20,ymax-labeloffset), text=("green: Emech"))
gEpot = gcurve(color=color.yellow)
gEkin = gcurve(color = color.blue)
gEmech = gcurve(color=color.green)
gEkin.plot(pos=(t[0],Ekin[0]))
gEpot.plot(pos=(t[0],Epot[0]))
gEmech.plot(pos=(t[0],Emech[0]))

# acceleration diagram
Atgraph = gdisplay(x=800, y=450, width=600, height=500, 
             title='t-a-diagram', xtitle='t/s', ytitle='a/m/s²', 
             xmax=T, xmin=0, ymax=max_accel+0.1*max_accel, ymin=-3, 
             foreground=fgcolor, background=bgcolor)
gAt = gcurve(color = color.blue)
gAt.plot(pos=(t[0],a_abs[0]))

starttime = time . time ()
for n in range(n_max):
    rate(1000)

    phi = ph[n]
    n_xz = np.sin(angle)
    unx_sn = -n_xz * np.cos(phi)
    uny_sn = np.cos(angle)
    unz_sn = -n_xz * np.sin(phi)
    n_mag = np.sqrt(unx_sn**2 + uny_sn**2 + unz_sn**2)
    unx_sn /= n_mag
    uny_sn /= n_mag
    unz_sn /= n_mag
    
    tx = vx[n+1]
    ty = vy[n+1]
    tz = vz[n+1]
    t_mag = np.sqrt(tx**2 + ty**2 + tz**2)
    utx_sn = tx/t_mag
    uty_sn = ty/t_mag
    utz_sn = tz/t_mag
    
    uxp = ux[n]
    uyp = uy[n] 
    uzp = uz[n]
    
    block.axis.x = L*utx_sn
    block.axis.y = H*uty_sn
    block.axis.z = W*utz_sn
    block.up.x = unx_sn
    block.up.y = uny_sn
    block.up.z = unz_sn
    block.size.x = L
    block.size.y = H
    block.size.z = W
    block.x = uxp
    block.velocity.x = vx[n]
    block.y = uyp
    block.velocity.y = vy[n]
    block.z = uzp
    block.velocity.z = vz[n]

    tn = t[n]
    gEkin.plot(pos=(tn,Ekin[n]))
    gEpot.plot(pos=(tn,Epot[n]))
    gEmech.plot(pos=(tn,Emech[n]))
    gAt.plot(pos=(tn,a_abs[n]))
    
endtime = time . time ()
difftime = endtime - starttime
print("Rechenzeit für Visualisierung der Bewegung: {0:10.8}s".format(difftime))
n = n_max
print("max n: {0} Endzeit t={1:6.5}s x={2:10.9}m, y={3:10.9}m und z={4:10.9}m".format(n,t[n],ux[n],uy[n],uz[n]))
print("Geschwindigkeit vx={0:10.9}m/s, vy={1:10.9}m/s vz={2:10.9}m/s und v={3:10.9}m/s".format(vx[n],vy[n],vz[n],vabs[n]))
print("Beschleunigung ax={0:10.9}m/s², ay={1:10.9}m/s² az={2:10.9}m/s² und a={3:10.9}m/s²".format(ax[n],ay[n],az[n],a_abs[n]))
print("Ekin={0:10.9}J  Epot={1:10.9}J  Emech={2:10.9}J Eges={3:10.9}J".format(Ekin[n],Epot[n],Emech[n],Etot[n])) 
Et_final = Emech[n] 
Et_diff = Et_initial - Et_final
print("Differenz (EG Anfang) - (EG Ende): {0:10.9}J".format(Et_diff))
mean_abs_diff_Et /= n_max
print("mittlere absolute Abweichung der Gesamtenergie: {0:10.9}J".format(mean_abs_diff_Et))
print("mittlere relative Abweichung der Gesamtenergie: {0:10.9}".format(mean_abs_diff_Et/Et_initial))
print("gesamter zurückgelegter Weg: {0:9.7}m".format(s))
