# Copyright (c) 2019,
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
from vpython import *

def dotProd(ax,ay,az,bx,by,bz):
    dp = ax*bx + ay*by + az*bz
    return dp

def crossProd(ax,ay,az,bx,by,bz):
    cx = ay*bz - az*by
    cy = az*bx - ax*bz
    cz = ax*by - ay*bx
    return cx, cy, cz
    
def fx(s, R, omega, b):
    d = R**2 * omega**2 + b**2
    sqd = sqrt(d)
    x = R*cos((omega*s)/sqd)
    return x

def fx_1A(s, R, omega, b):
    d = R**2 * omega**2 + b**2
    sqd = sqrt(d)
    x = -(R*omega/sqd)*sin((omega*s)/sqd)
    return x

def fx_2A(s, R, omega, b):
    d = R**2 * omega**2 + b**2
    sqd = sqrt(d)
    x = -(R*omega**2/d)*cos((omega*s)/sqd)
    return x

def fy(s, R, omega, b, h):
    d = R**2 * omega**2 + b**2
    sqd = sqrt(d)
    y = h - (b*s)/sqd 
    return y

def fy_1A(s, R, omega, b):
    d = R**2 * omega**2 + b**2
    sqd = sqrt(d)
    y = -b/sqd 
    return y

def fy_2A(s, R, omega, b):
    y = 0.0 
    return y

def fz(s, R, omega, b):
    d = R**2 * omega**2 + b**2
    sqd = sqrt(d)
    z = R*sin((omega*s)/sqd)
    return z

def fz_1A(s, R, omega, b):
    d = R**2 * omega**2 + b**2
    sqd = sqrt(d)
    z = (R*omega/sqd)*cos((omega*s)/sqd)
    return z

def fz_2A(s, R, omega, b):
    d = R**2 * omega**2 + b**2
    sqd = sqrt(d)
    z = -(R*omega**2/d)*sin((omega*s)/sqd)
    return z

def Nx(s, R, omega, b):         # calculation of the x-component of the normal vector
    fx1 = fx_1A(s,R,omega,b)
    fx2 = fx_2A(s,R,omega,b)
    fy1 = fy_1A(s,R,omega,b)
    fy2 = fy_2A(s,R,omega,b)
    fz1 = fz_1A(s,R,omega,b)
    fz2 = fz_2A(s,R,omega,b)

    nx = (fy1**2 + fz1**2) * fx2 - fx1*fy1*fy2 - fx1*fz1*fz2
    return nx

def Ny(s, R, omega, b):         # calculation of the y-component of the normal vector
    fx1 = fx_1A(s,R,omega,b)
    fx2 = fx_2A(s,R,omega,b)
    fy1 = fy_1A(s,R,omega,b)
    fy2 = fy_2A(s,R,omega,b)
    fz1 = fz_1A(s,R,omega,b)
    fz2 = fz_2A(s,R,omega,b)

    ny = (fx1**2 + fz1**2) * fy2 - fx1*fy1*fx2 - fy1*fz1*fz2
    return ny

def Nz(s, R, omega, b):         # calculation of the z-component of the normal vector
    fx1 = fx_1A(s,R,omega,b)
    fx2 = fx_2A(s,R,omega,b)
    fy1 = fy_1A(s,R,omega,b)
    fy2 = fy_2A(s,R,omega,b)
    fz1 = fz_1A(s,R,omega,b)
    fz2 = fz_2A(s,R,omega,b)

    nz = (fx1**2 + fy1**2) * fz2 - fx1*fz1*fx2 - fy1*fz1*fy2
    return nz

def unityTangentVec(s, R, omega, b):        # calculation of the unity tangent vector
    tx = fx_1A(s, R, omega, b)
    ty = fy_1A(s, R, omega, b)
    tz = fz_1A(s, R, omega, b)
    t_mag = np.sqrt(tx**2 + ty**2 + tz**2)
    tx /= t_mag
    ty /= t_mag
    tz /= t_mag
    return tx, ty, tz

def unityNormalVec(s, R, omega, b):         # calculation of the unity normal vector 
    fx1 = fx_1A(s,R,omega,b)
    fx2 = fx_2A(s,R,omega,b)
    fy1 = fy_1A(s,R,omega,b)
    fy2 = fy_2A(s,R,omega,b)
    fz1 = fz_1A(s,R,omega,b)
    fz2 = fz_2A(s,R,omega,b)
    nx = (fy1**2 + fz1**2) * fx2 - fx1*fy1*fy2 - fx1*fz1*fz2
    ny = (fx1**2 + fz1**2) * fy2 - fx1*fy1*fx2 - fy1*fz1*fz2
    nz = (fx1**2 + fy1**2) * fz2 - fx1*fz1*fx2 - fy1*fz1*fy2
    mag = np.sqrt(nx**2 + ny**2 + nz**2)
    nx /= mag
    ny /= mag
    nz /= mag
    return nx, ny, nz

def unityTrackNormalVec(s, R, omega, b, nx, ny, nz, vx, vz):
    kapxz = kappa_xz(s, R, omega, b)
    beta_xz = np.arctan( nz / nx )
    Nxz_mag = np.sqrt(nx**2 + nz**2)
    v_mag_xz_sq = vx**2 + vz**2
    alpha_xz = np.arctan(v_mag_xz_sq * kapxz / g)
    if kapxz > 0.0: 
        unpx = Nxz_mag * np.cos(beta_xz)
        unpy = ny + Nxz_mag * np.tan(2.0*np.pi/4.0 - alpha_xz)
        unpz = Nxz_mag * np.sin(beta_xz)
    else:
        unpx = nx
        unpy = ny
        unpz = nz
    unp_mag = np.sqrt(unpx**2 + unpy**2 + unpz**2)
    unpx /= unp_mag
    unpy /= unp_mag
    unpz /= unp_mag
    
    if np.sign(unpx) != np.sign(nx):
            unpx *= -1.0
    #if np.sign(unpy) != np.sign(ny):
    #        unpy *= -1.0
    if np.sign(unpz) != np.sign(nz):
            unpz *= -1.0
    return unpx, unpy, unpz

def unityBiNormalVec(s, R, omega, b):      # calculation of the unity binormal vector
    tx = fx_1A(s, R, omega, b)
    ty = fy_1A(s, R, omega, b)
    tz = fz_1A(s, R, omega, b)
    t_mag = np.sqrt(tx**2 + ty**2 + tz**2)
    tx /= t_mag
    ty /= t_mag
    tz /= t_mag
    nx, ny, nz = unityNormalVec(s, R, omega, b)
    bx = ty*nz - tz*ny
    by = tz*nx - tx*nz
    bz = tx*ny - ty*nx
    b_mag = np.sqrt(bx**2 + by**2 + bz**2)
    bx /= b_mag
    by /= b_mag
    bz /= b_mag
    return bx, by, bz

def unityTrackBiNormalVec(s, R, omega, b, nx, ny, nz, vx, vz):      
    tx, ty, tz = unityTangentVec(s, R, omega, b)
    unx, uny, unz = unityTrackNormalVec(s, R, omega, b, nx, ny, nz, vx, vz)
    bx, by, bz = crossProd(tx,ty,tz,unx,uny,unz)
    b_mag = np.sqrt(bx**2 + by**2 + bz**2)
    bx /= b_mag
    by /= b_mag
    bz /= b_mag
    return bx, by, bz

def kappa(s, R, omega, b):      # calculation of the curvature 
    fx1 = fx_1A(s,R,omega,b)
    fy1 = fy_1A(s,R,omega,b)
    fz1 = fz_1A(s,R,omega,b)
    nx, ny, nz = unityNormalVec(s, R, omega, b)
    k = sqrt(nx**2 + ny**2 + nz**2) / ( (fx1**2 + fy1**2 + fz1**2)**1.5 )
    return k

def kappa_xz(s, R, omega, b):      # calculation of the curvature in the xz-plane
    fx1 = fx_1A(s,R,omega,b)
    fz1 = fz_1A(s,R,omega,b)
    nx = Nx(s, R, omega, b)
    nz = Nz(s, R, omega, b)
    k = sqrt(nx**2 + nz**2) / ( (fx1**2 + fz1**2)**1.5 )
    return k

def trap(f, n, a, b, ang):         # numerical integration with trapezoidal rule
    h = (b - a) / float(n)
    intgr = 0.5 * h * (f(a,ang) + f(b,ang))
    for i in range(1, int(n)):
        intgr = intgr + h * f(a + i * h, ang)
    return intgr

def df(x,ang):                  # for calculation of the arc length
    fax = fx_1A(x, R, omega, b)
    fay = fy_1A(x, R, omega, b)
    faz = fz_1A(x, R, omega, b)
    return sqrt(fax**2 + fay**2 + faz**2)

# values of constants
g = 9.81    # gravity acceleration
m = 1.0     # mass of body 
mu = 0.01    # friction coefficient

# geometry parameter
R = 5.0
omega = 1.0
b = 0.5
hy = 35.0

# Initial conditions
SC_0 = 0.0
VC_0 = 0.0

# values for numerical calculation
T = 20.0    # time duration of the motion in seconds
dt = 0.01   # timestep 

N_t = int(round(T/dt))
print("N_t: {0}".format(N_t))
t = linspace(0, N_t*dt, N_t+1)

sc = zeros(N_t+1)    # position on the curve at timestep n
vc = zeros(N_t+1)    # velocity on the curve at timestep n
ux = zeros(N_t+1)    # x-component of position vector u
vx = zeros(N_t+1)    # x-component of velocity vector v
uy = zeros(N_t+1)    # y-component of position vector u
vy = zeros(N_t+1)    # y-component of velocity vector v
uz = zeros(N_t+1)    # z-component of position vector u
vz = zeros(N_t+1)    # z-component of velocity vector v

ac = zeros(N_t+1)    # acceleration in tangential direction at timestep n
ax = zeros(N_t+1)    # x-component of acceleration vector 
ay = zeros(N_t+1)    # y-component of acceleration vector 
az = zeros(N_t+1)    # z-component of acceleration vector 
a_abs = zeros(N_t+1) # acceleration

vabs = zeros(N_t+1)
ds_arr = zeros(N_t+1)
FRabs = zeros(N_t+1) # friction force
wr = zeros(N_t+1)    # differential work 
Ekin = zeros(N_t+1)  # kinetic energy
Epot = zeros(N_t+1)  # potential energy
wr_tot = zeros(N_t+1)# total work (-(thermal energy) )
Etot = zeros(N_t+1)  # total energy  
Emech = zeros(N_t+1) # mechanical energy
Ein = zeros(N_t+1)   # thermal energy

fx_sn = fx(sc[0], R, omega, b)
fy_sn = fy(sc[0], R, omega, b, hy)
fz_sn = fz(sc[0], R, omega, b)

fx1A_sn = fx_1A(sc[0],R,omega,b)
fx2A_sn = fx_2A(sc[0],R,omega,b)
fy1A_sn = fy_1A(sc[0],R,omega,b)
fy2A_sn = fy_2A(sc[0],R,omega,b)
fz1A_sn = fz_1A(sc[0],R,omega,b)
fz2A_sn = fz_2A(sc[0],R,omega,b)

sc[0] = SC_0
vc[0] = VC_0

ux[0] = fx_sn
vx[0] = fx1A_sn * vc[0]
uy[0] = fy_sn
vy[0] = fy1A_sn * vc[0]
uz[0] = fz_sn
vz[0] = fz1A_sn * vc[0]
vabs[0] = sqrt(vx[0]**2 + vy[0]**2 + vz[0]**2)

utx_sn,uty_sn,utz_sn = unityTangentVec(sc[0], R, omega, b)

print("Komponenten Tangentenvektor: {0:10.9} {1:10.9} {2:10.9}".format(utx_sn,uty_sn,utz_sn))
slope_y = uty_sn / (np.sqrt(utx_sn**2 + utz_sn**2))
print("Steigung der Bahn zur Horizontalen: {0:10.9}".format(slope_y))
kap = kappa(sc[0], R, omega, b)
print("Krümmung: {0:10.9} 1/m".format(kap))
kapxz = kappa_xz(sc[0], R, omega, b)
print("Krümmung xz-Ebene: {0:10.9} 1/m".format(kapxz))

unx_sn,uny_sn,unz_sn = unityNormalVec(sc[0], R, omega, b)

uTnx_sn,uTny_sn,uTnz_sn = unityTrackNormalVec(sc[0], R, omega, b, unx_sn, uny_sn, unz_sn, vx[0], vz[0])

dpUN_UTN = dotProd(unx_sn, uny_sn, unz_sn, uTnx_sn, uTny_sn, uTnz_sn)

if vc[0] < 0.0:
    fr_fac = -1.0
else:
    fr_fac = 1.0

ac[0] = ( (fx1A_sn*fx2A_sn + fy1A_sn*fy2A_sn + fz1A_sn*fz2A_sn)*(vc[0])**2
          + fy1A_sn*g
          + fr_fac*mu*(
                        (vc[0])**2 * kappa(sc[0], R, omega, b) * dpUN_UTN
                       + uTny_sn * g * 1.0/(sqrt(1.0+slope_y**2))
                       )
         ) * ( - 1.0/( fx1A_sn**2 + fy1A_sn**2 + fz1A_sn**2))

ax[0] =  fx2A_sn * vc[0]**2 + fx1A_sn * ac[0] 
ay[0] =  fy2A_sn * vc[0]**2 + fy1A_sn * ac[0] 
az[0] =  fz2A_sn * vc[0]**2 + fz1A_sn * ac[0]
a_abs[0] = np.sqrt(ax[0]**2 + ay[0]**2 + az[0]**2)

fr = mu*m*(
            (vc[0])**2 * kappa(sc[0], R, omega, b) * dpUN_UTN
           + uTny_sn * g * 1.0/(sqrt(1.0+slope_y**2))
           )*( - 1.0/( fx1A_sn**2 + fy1A_sn**2 + fz1A_sn**2))
FRabs[0] = abs(fr)

n = 0
print("Anfangszeit t={0:6.5}s x={1:10.9}m, y={2:10.9}m und z={3:10.9}m".format(t[n],ux[n],uy[n],uz[n]))
print("Geschwindigkeit vx={0:10.9}m/s, vy={1:10.9}m/s vz={2:10.9}m/s und v={3:10.9}m/s".format(vx[n],vy[n],vz[n],vabs[n]))
print("Beschleunigung ax={0:10.9}m/s², ay={1:10.9}m/s² az={2:10.9}m/s² und a={3:10.9}m/s²".format(ax[n],ay[n],az[n],a_abs[n]))
print("Tangentialbeschleunigung at={0:10.9} m/s²".format( ac[n]))
print("FRabs[0]: {0:10.9} N".format( FRabs[n]))

wr[0] = 0.0
wr_tot[0] = wr[0]
WR = 0.0
Ekin[0] = 0.5*m*( (vx[0])**2 + (vy[0])**2 + (vz[0])**2 )
Epot[0] = m * g * uy[0]                     # reference value Epot,R = 0 at y = 0
Emech[0] = Ekin[0] + Epot[0]
Etot[0] = Emech[0] + abs(WR)

Et_initial = Emech[0] + abs(wr_tot[0])

print("Anfangswerte für t={0:6.3} Ekin={1:10.9}J  Epot={2:10.9}J  Emech={3:10.9}J  WR={4:10.9}J  EG={5:10.9}J".
      format(t[0],Ekin[0],Epot[0],Emech[0],wr_tot[0],Et_initial)) 

s = 0.0         # for summation of differential path elements

mean_abs_diff_Eges = 0.0
s_min = sc[0]
s_max = sc[0]
n_max = 0
max_friction_force = FRabs[0]
n_max_fr = 0
x_max_fr = sc[0]
v_max_fr = 0.0

max_accel = a_abs[0]
n_max_accel = 0

starttime = time . time ()
for n in range(N_t):
    
    fx1A_sn = fx_1A(sc[n],R,omega,b)
    fx2A_sn = fx_2A(sc[n],R,omega,b)
    fy1A_sn = fy_1A(sc[n],R,omega,b)
    fy2A_sn = fy_2A(sc[n],R,omega,b)
    fz1A_sn = fz_1A(sc[n],R,omega,b)
    fz2A_sn = fz_2A(sc[n],R,omega,b)

    unx_sn,uny_sn,unz_sn = unityNormalVec(sc[n], R, omega, b)

    utx_sn,uty_sn,utz_sn = unityTangentVec(sc[n], R, omega, b)

    uTnx_sn,uTny_sn,uTnz_sn = unityTrackNormalVec(sc[n], R, omega, b, unx_sn, uny_sn, unz_sn, vx[n], vz[n]) 

    slope_y = uty_sn / (np.sqrt(utx_sn**2 + utz_sn**2))

    dpUN_UTN = dotProd(unx_sn, uny_sn, unz_sn, uTnx_sn, uTny_sn, uTnz_sn)

    if vc[n] < 0.0:
        fr_fac = -1.0
    else:
        fr_fac = 1.0

    ac[n] = ( (fx1A_sn*fx2A_sn + fy1A_sn*fy2A_sn + fz1A_sn*fz2A_sn)*(vc[n])**2
                + fy1A_sn*g
                + fr_fac*mu*(
                                (vc[n])**2 * kappa(sc[n], R, omega, b) * dpUN_UTN
                              + uTny_sn * g * 1.0/(sqrt(1.0+slope_y**2))
                             )
             ) * ( - 1.0/( fx1A_sn**2 + fy1A_sn**2 + fz1A_sn**2))

    # one step forward using Euler-Cromer method
    vc[n+1] = vc[n] + dt * ac[n]
    
    sc[n+1] = sc[n] + dt * vc[n+1]

    fx1A_sn = fx_1A(sc[n+1],R,omega,b)
    fx2A_sn = fx_2A(sc[n+1],R,omega,b)
    fy1A_sn = fy_1A(sc[n+1],R,omega,b)
    fy2A_sn = fy_2A(sc[n+1],R,omega,b)
    fz1A_sn = fz_1A(sc[n+1],R,omega,b)
    fz2A_sn = fz_2A(sc[n+1],R,omega,b)

    ux[n+1] = fx(sc[n+1], R, omega, b)
    vx[n+1] = fx1A_sn * vc[n+1]
    uy[n+1] = fy(sc[n+1], R, omega, b, hy)
    vy[n+1] = fy1A_sn * vc[n+1]
    uz[n+1] = fz(sc[n+1], R, omega, b)
    vz[n+1] = fz1A_sn * vc[n+1]

    unx_sn,uny_sn,unz_sn = unityNormalVec(sc[n+1], R, omega, b)

    utx_sn,uty_sn,utz_sn = unityTangentVec(sc[n+1], R, omega, b)

    uTnx_sn,uTny_sn,uTnz_sn = unityTrackNormalVec(sc[n+1], R, omega, b, unx_sn, uny_sn, unz_sn, vx[n+1], vz[n+1])

    slope_y = uty_sn / (np.sqrt(utx_sn**2 + utz_sn**2))

    dpUN_UTN = dotProd(unx_sn, uny_sn, unz_sn, uTnx_sn, uTny_sn, uTnz_sn)

    if vc[n+1] < 0.0:
        fr_fac = -1.0
    else:
        fr_fac = 1.0
    
    ac[n+1] = ( (fx1A_sn*fx2A_sn + fy1A_sn*fy2A_sn + fz1A_sn*fz2A_sn)*(vc[n+1])**2
                + fy1A_sn*g
                + fr_fac*mu*(
                                (vc[n+1])**2 * kappa(sc[n+1], R, omega, b) * dpUN_UTN
                              + uTny_sn * g * 1.0/(sqrt(1.0+slope_y**2))
                             )
               ) * ( - 1.0/( fx1A_sn**2 + fy1A_sn**2 + fz1A_sn**2))

    ax[n+1] =  fx2A_sn * vc[n+1]**2 + fx1A_sn * ac[n+1] 
    ay[n+1] =  fy2A_sn * vc[n+1]**2 + fy1A_sn * ac[n+1] 
    az[n+1] =  fz2A_sn * vc[n+1]**2 + fz1A_sn * ac[n+1]
    
    #value of the friction force
    fr = mu*m*(
                   (vc[n+1])**2 * kappa(sc[n+1], R, omega, b) * dpUN_UTN
                 + uTny_sn * g * 1.0/(sqrt(1.0+slope_y**2))
               ) * ( - 1.0/( fx1A_sn**2 + fy1A_sn**2 + fz1A_sn**2))

    FRabs[n+1] = abs(fr)

    vabs[n+1] = sqrt(vx[n+1]**2 + vy[n+1]**2 + vz[n+1]**2)
    a_abs[n+1] = np.sqrt(ax[n+1]**2 + ay[n+1]**2 + az[n+1]**2)

    if sc[n+1] > s_max:
        s_max = sc[n+1] 
    if sc[n+1] < s_min:
        s_min = sc[n+1]     
    dx = ux[n+1] - ux[n]
    dy = uy[n+1] - uy[n]
    dz = uz[n+1] - uz[n]
    ds = sqrt(dx**2 + dy**2 + dz**2)  
    ds_arr[n+1] = ds 
    s += ds
    
    WR -= FRabs[n+1] * ds
    wr[n+1] = -FRabs[n+1] * ds
    wr_tot[n+1] = WR
    Ein[n+1] = -wr_tot[n+1]
    Ekin[n+1] = 0.5*m*( (vx[n+1])**2 + (vy[n+1])**2 + (vz[n+1])**2 )
    Epot[n+1] = m * g * uy[n+1]
    Emech[n+1] = Ekin[n+1] + Epot[n+1]
    Etot[n+1] = Emech[n+1] + abs(wr_tot[n+1])
    mean_abs_diff_Eges += abs(Etot[n+1] - Et_initial)
    if FRabs[n+1] > abs(max_friction_force):
       max_friction_force = FRabs[n+1]
       v_max_fr = vabs[n+1]
       x_max_fr = sc[n+1]
       n_max_fr = n+1

    if a_abs[n+1] > max_accel:
       max_accel = a_abs[n+1]
       n_max_accel = n+1
    
    n_max = n + 1
   
endtime = time . time ()
difftime = endtime - starttime
print("Rechenzeit für numerische Berechnung der Bewegung: {0:10.8}s".format(difftime))
tt = zeros(n_max) 
utx = zeros(n_max) 
uty = zeros(n_max)
utz = zeros(n_max)
unx = zeros(n_max) 
uny = zeros(n_max)
unz = zeros(n_max)
uTnx = zeros(n_max) 
uTny = zeros(n_max)
uTnz = zeros(n_max)
uTn_mag_ar = zeros(n_max)
uTbx = zeros(n_max) 
uTby = zeros(n_max)
uTbz = zeros(n_max)
ang_track = zeros(n_max)

# Visualization
scene = canvas(title='Bewegung eines Koerpers auf einer Helix-Bahn',
                x=0, y=0, width=800, height=600)
scene.center=vec(0.0,35.0,1.0)
scene.range=6.0
scene.background = color.gray(0.3)
scene.ambient = color.gray(0.8)

curvx = zeros(n_max)
curvy = zeros(n_max)
curvz = zeros(n_max)
curvx_i = zeros(n_max)
curvy_i = zeros(n_max)
curvz_i = zeros(n_max)
curvx_o = zeros(n_max)
curvy_o = zeros(n_max)
curvz_o = zeros(n_max)

for n in range(n_max):
    curvx[n] = fx(sc[n],R,omega,b)
    curvy[n] = fy(sc[n],R,omega,b,hy)
    curvz[n] = fz(sc[n],R,omega,b)
    
halfOfdistRail = 0.25
dr = halfOfdistRail


for n in range(n_max):
    utx[n],uty[n],utz[n] = unityTangentVec(sc[n], R, omega, b)
    unx[n],uny[n],unz[n] = unityNormalVec(sc[n], R, omega, b)
    uTnx[n], uTny[n], uTnz[n] = unityTrackNormalVec(sc[n], R, omega, b, unx[n], uny[n], unz[n], vx[n], vz[n])
    uTbx[n], uTby[n], uTbz[n] = unityTrackBiNormalVec(sc[n], R, omega, b, unx[n], uny[n], unz[n], vx[n], vz[n])

curvx_i = curvx + dr * uTbx
curvy_i = curvy + dr * uTby
curvz_i = curvz + dr * uTbz
curvx_o = curvx - dr * uTbx
curvy_o = curvy - dr * uTby
curvz_o = curvz - dr * uTbz

radius = 0.03
cur1 = curve( color = color.red, radius = radius)
cur2 = curve( color = color.yellow, radius = radius)
cur3 = curve( color = color.yellow, radius = radius)
for i in range(len(curvx)):
    cur1.append(vector(curvx[i],curvy[i],curvz[i]))
    cur2.append(vector(curvx_i[i],curvy_i[i],curvz_i[i]))
    cur3.append(vector(curvx_o[i],curvy_o[i],curvz_o[i]))
#curve( x=curvx_i, y=curvy_i, z=curvz_i, color = color.yellow, radius = radius)
#curve( x=curvx_o, y=curvy_o, z=curvz_o, color = color.yellow, radius = radius)

fx_sn = fx(sc[0], R, omega, b)
fy_sn = fy(sc[0], R, omega, b, hy)
fz_sn = fz(sc[0], R, omega, b)

fx1A_sn = fx_1A(sc[0],R,omega,b)
fx2A_sn = fx_2A(sc[0],R,omega,b)
fy1A_sn = fy_1A(sc[0],R,omega,b)
fy2A_sn = fy_2A(sc[0],R,omega,b)
fz1A_sn = fz_1A(sc[0],R,omega,b)
fz2A_sn = fz_2A(sc[0],R,omega,b)
    
L = 0.5
H = 0.5
W = 0.5
height = 0.025
uxp = ux[0] + (H/2 + height)*uTnx[0]
uyp = uy[0] + (H/2 + height)*uTny[0]
uzp = uz[0] + (H/2 + height)*uTnz[0]

block = box(pos=vec(uxp,uyp,uzp),axis=vec(L*utx[0],H*uty[0],W*utz[0]),size=vec(L,H,W),up=vec(uTnx[0],uTny[0],uTnz[0]),
            color=color.black,opacity=0.9)
            #make_trail=True, trail_type="points",interval=30, retain=30)

block.velocity = vector(vx[0],vy[0],vz[0])

fgcolor=color.black; bgcolor=color.white

# energy diagram
energydiagram = True
if energydiagram:
    ymax = Etot[0] + 0.15*Etot[0]
    labeloffset = 0.05*Etot[0] 
    Egraph = graph(x=800, y=000, width=650, height=450, 
                 title='t-E-Diagramm', xtitle='t/s', ytitle='E/J', 
                 xmax=T, xmin=0.0, ymax=ymax, ymin=-4, 
                 foreground=fgcolor, background=bgcolor)
    label(pos=vec(3,ymax-labeloffset,0), text=("cyan: Epot"))
    label(pos=vec(11,ymax-labeloffset,0), text=("blue: Ekin"))
    label(pos=vec(20,ymax-labeloffset,0), text=("green: Emech"))
    label(pos=vec(28,ymax-labeloffset,0), text=("red: Eth"))
    label(pos=vec(36,ymax-labeloffset,0), text=("black: Etot"))

    gEpot = gcurve(color=color.cyan)
    gEkin = gcurve(color = color.blue)
    gEmech = gcurve(color=color.green)
    gEin = gcurve(color = color.red)
    gEtot = gcurve(color = color.black)
    gEkin.plot(pos=(t[0],Ekin[0]))
    gEpot.plot(pos=(t[0],Epot[0]))
    gEmech.plot(pos=(t[0],Emech[0]))
    gEin.plot(pos=(t[0],Ein[0]))
    gEtot.plot(pos=(t[0],Etot[0]))

    Atgraph = graph(x=800, y=450, width=650, height=400, 
                 title='t-a-Diagramm', xtitle='t/s', ytitle='a/(m/s²)', 
                 xmax=T, xmin=0, ymax=max_accel+0.1*max_accel, ymin=0, 
                 foreground=fgcolor, background=bgcolor)
    gAt = gcurve(color = color.blue)
    gAt.plot(pos=(t[0],a_abs[0]))

#scene.mouse.getclick() #wait for click

icam_chg = 0

starttime = time . time ()
for n in range(n_max):
    rate(100)

    if abs(scene.center.y - uy[n]) > 5.0:
        if icam_chg > 0:
            scene.center=vec(0.0,uy[n] - 3.5,1.0)
        else:
            scene.center=vec(0.0,uy[n] + 0.5,1.0)
            
        icam_chg += 1
    
    sc_n = sc[n]
    fx1A_snp1 = fx_1A(sc_n,R,omega,b)
    fy1A_snp1 = fy_1A(sc_n,R,omega,b)
    fz1A_snp1 = fz_1A(sc_n,R,omega,b)

    uxp = ux[n] + (H/2 + height)*uTnx[n]
    uyp = uy[n] + (H/2 + height)*uTny[n] 
    uzp = uz[n] + (H/2 + height)*uTnz[n]

    block.axis = vec(L*fx1A_snp1, H*fy1A_snp1, W*fz1A_snp1)

    block.up = vec(uTnx[n], uTny[n], uTnz[n]) 

    block.size = vec(L, H, W)

    block.pos = vector(uxp, uyp, uzp)  

    block.velocity = vector(vx[n], vy[n], vz[n])  

    tn = t[n]
    if energydiagram:
        gEkin.plot(pos=(tn,Ekin[n]))
        gEpot.plot(pos=(tn,Epot[n]))
        gEmech.plot(pos=(tn,Emech[n]))
        gEin.plot(pos=(tn,Ein[n]))
        gEtot.plot(pos=(tn,Etot[n]))

        gAt.plot(pos=(tn,a_abs[n]))
    
endtime = time . time ()
difftime = endtime - starttime
print("Rechenzeit für Visualisierung der Bewegung: {0:10.8}s".format(difftime))

n = n_max
print("max n: {0} Endzeit t={1:6.5}s x={2:10.9}m, y={3:10.9}m und z={4:10.9}m".format(n,t[n],ux[n],uy[n],uz[n]))
print("Geschwindigkeit vx={0:10.9}m/s, vy={1:10.9}m/s vz={2:10.9}m/s und v={3:10.9}m/s".format(vx[n],vy[n],vz[n],vabs[n]))
print("Beschleunigung ax={0:10.9}m/s², ay={1:10.9}m/s² az={2:10.9}m/s² und a={3:10.9}m/s²".format(ax[n],ay[n],az[n],a_abs[n]))
print("maximale Beschleunigung a_max={0:10.9}m/s² zur Zeit {1:6.5}s an der Position {2:6.5}m"
      .format(max_accel, t[n_max_accel],sc[n_max_accel]))
print("Tangentialbeschleunigung at={0:10.9} m/s²".format( ac[n]))
print("Ekin={0:10.9}J  Epot={1:10.9}J  Emech={2:10.9}J WR={3:10.9}J Ein={4:10.9}J Eges={5:10.9}J".
        format(Ekin[n],Epot[n],Emech[n],wr_tot[n],Ein[n],Etot[n])) 
Et_final = Emech[n] + abs(wr_tot[n])
Et_diff = Et_initial - Et_final
print("Differenz (EG Anfang) - (EG Ende): {0:10.9}J".format(Et_diff))
mean_abs_diff_Eges /= n_max
print("mittlere absolute Abweichung der Gesamtenergie: {0:10.9}J".format(mean_abs_diff_Eges))
print("mittlere relative Abweichung der Gesamtenergie: {0:10.9}".format(mean_abs_diff_Eges/Et_initial))
asc = s_min
bsc = s_max
intgr_trap = trap(df,1000, asc, bsc,0)

print("Länge der Bahnkurve: {0:9.7}m zwischen min s-Wert smin={1:9.7}m und max s-Wert smax={2:9.7}m"
      .format(abs(intgr_trap),asc,bsc))
print("gesamter zurückgelegter Weg: {0:9.7}m".format(s))

kap = kappa(sc[n], R, omega, b)
print("Krümmung: {0:10.9}".format(kap))
kapxz = kappa_xz(sc[n], R, omega, b)
print("Krümmung xz-Ebene: {0:10.9}".format(kapxz))

mean_FR = 0.0
nr_mean_FR = 0
for n in range(n_max):
    mean_FR += abs(FRabs[n])
    nr_mean_FR += 1
mean_FR /= nr_mean_FR
print("zeitlicher Mittelwert der Reibungskraft FR: {0:10.9}N".format(mean_FR))
print("max Betrag von FR: {0:10.9}N zur Zeit {1:9.7}s bei s={2:9.7}m mit der Geschwindigkeit v={3:9.7}m/s"
      .format(max_friction_force,t[n_max_fr],x_max_fr,v_max_fr)) 

