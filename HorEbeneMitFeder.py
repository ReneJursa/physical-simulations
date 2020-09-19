import time
import numpy as np
from numpy import linspace, zeros, exp, asarray, pi, cos, sin
import matplotlib.pyplot as plt
from visual import *
from visual.graph import *

# geometry parameter of construction
alpha = 10.0
x_offset = -1.0
xB = 4.0
xD = 4.80
xZ = 20.0

def f(x):
    condition1 = x <= xD
    condition2 = x > xD
    y1 = 1.0
    y2 = 0.0
    y = np.where(condition1, y1, 0.0)
    y = np.where(condition2, y2, y)
    return y

def f_1A(x):
    y = 0.0
    return y

def f_2A(x):
    y = 0.0
    return y

def gf(x):
    condition1 = x <= xD
    condition2 = x > xD
    y1 = 1.0
    y2 = 0.0
    y = np.where(condition1, y1, 0.0)
    y = np.where(condition2, y2, y)
    return y

def gf_1A(x):
    y = 0.0
    return y

def gf_2A(x):
    y = 0.0
    return y

def df(x,ang):
    fa = f_1A(x)
    return sqrt(1.0 + fa**2)

def trap(f, n, a, b, ang):              # numerical integration with trapez rule
    h = (b - a) / float(n)
    intgr = 0.5 * h * (f(a,ang) + f(b,ang))
    for i in range(1, int(n)):
        intgr = intgr + h * f(a + i * h, ang)
    return intgr

scene = display(title='Bewegung mit zentral elastischem Stoss',
                x=0, y=0, width=1400, height=350,
                center=(6.0,0.8,0),range=(8.0,0.8,0.5)
                )
scene.background = color.gray(0.3)
scene.ambient = color.gray(0.5)
scene.forward=vector( 0.0,-0.2, -1.0 )

# values of constants
g = 9.81    # gravity acceleration
m = 1.5     # mass of body
#m2 = 0.8   # mass of body2
m2 = 0.6    # mass of body2
mu = 0.1    # friction coefficient
D = 50.0    # spring constant
comp = 0.8  # initial compression of the spring 
T = 3.0     # time duration of the motion in seconds
dt = 0.001   # timestep for numerical calculation


# calculation of the friction force/mass
def friction(f_1A, f_2A, uxn, uyn, vxn, vyn):
    f_1Aux = f_1A(uxn)
    f_2Aux = f_2A(uxn)
    K = 1.0 + f_1Aux**2
    sqrtK = np.sqrt(K)
    rho_inv = (abs(f_2Aux)) / K**(3/2)    # inverse of local curve radius rho
    fz = ((vxn)**2 + (vyn)**2) * rho_inv  # centripetal acceleration
    gn = g / sqrtK                        # normal component of g
    sign_f_2A = np.sign(f_2Aux)
    if f_2Aux == 0:
        sign_f_2A = 1.0
    fz_x = fz * sign_f_2A * ( - f_1Aux ) / sqrtK  # x-comp. of centripetal acceleration
    fz_y = fz * sign_f_2A / sqrtK                 # y-comp. of centripetal acceleration
    if f_2Aux < 0.0:
        fz_x *= -1.0
        fz_y *= -1.0
    gn_x = gn * sign_f_2A * ( - f_1Aux ) / sqrtK
    gn_y = gn * sign_f_2A / sqrtK
    #print("sign_f_2A: {0} gn: {1} gnx: {2} gny: {3}".format(sign_f_2A,gn,gn_x,gn_y))
    nx = fz_x + gn_x   # normal acceleration due g and radial acceleration
    ny = fz_y + gn_y

    Rx =  mu * sqrt(nx**2 + ny**2) / sqrtK   # x-comp. of friction in tangential direction
    Ry =  Rx * f_1Aux                        # y-comp. of friction force/mass in tangential direction

    if vxn >= 0.0 and vyn >= 0.0:            # friction force/mass in opposite direction of velocity
        Rx *= - 1.0
        Ry *= - 1.0
    if vxn >= 0.0 and vyn < 0.0:
        Rx *= - 1.0
        Ry *= - 1.0

    return Rx, Ry

def collision(m1, m2, v1, v2):
    u1 = ( (m1 - m2)*v1 + 2.0*m2*v2 ) / (m1 + m2)
    u2 = ( 2.0*m1*v1 + (m2 - m1)*v2 ) / (m1 + m2)
    return u1, u2

def oneStep(Body, fly, m, n, f, f_1A, f_2A, uxn, uyn, vxn, vyn, WR, rmse_y, nr_rmse_y):

    if (fly == True) and ((abs(f(uxn) - uyn) < 1.0e-7) or uyn < f(uxn)):
    
        print("Flug beendet bei ux[{0}]={1:10.9}m f(ux[{0}])={2:10.9}m uy[{0}]={3:10.9}m".format(n,uxn,f(uxn),uyn))
        fly = False
        
        Epot_i = m * g * uyn
        uyn = f(uxn)
        Epot_f = m * g * uyn
        diff_Epot = Epot_f - Epot_i
        print("Änderung der pot. Energie durch Auftreffen auf die Bahn: {0:10.9}J".format(diff_Epot))
        Ekin_i = 0.5*m*( vxn**2 + vyn**2 ) # Ekin before touching the rail
        f_1Aux = f_1A(uxn)
        K = 1.0 + f_1Aux**2
        sqrtK = np.sqrt(K)
        abs_vt = (vxn + vyn*f_1Aux) / sqrtK # velocity in tangential direction
        vxn = abs_vt / sqrtK                  # x-component of velocity in tangential direction
        vyn = vxn * f_1Aux                    # y-component of velocity in tangential direction
        Ekin_f = 0.5*m*( vxn**2 + vyn**2 )
        diff_Ekin = Ekin_f - Ekin_i
        print("Änderung der kin. Energie durch Auftreffen auf die Bahn: {0:10.9}J".format(diff_Ekin)) 
        WR += diff_Ekin
        WR += diff_Epot

    # one step forward using Heun method    
    ux_star = uxn + dt*vxn

    if fly == True:
        ax = 0.0
        ay = -g
    else:
        f_1Aux = f_1A(uxn)
        f_2Aux = f_2A(uxn)
        K = 1.0 + f_1Aux**2
        Rx, Ry = friction(f_1A, f_2A, uxn, uyn, vxn, vyn)
        ay = (f_2Aux*(vxn)**2 + g) / K - g + Ry               # y-component of acceleration
        ax = (f_2Aux*(vxn)**2 + g) / K * ( - f_1Aux) + Rx     # x-component of acceleration
        if Body == 1 and uxn <= comp:
            ax += -D/m*(uxn - comp)
        
    vx_star = vxn + dt*ax

    uy_star = uyn + dt*vyn
               
    vy_star = vyn + dt*ay

    uxnp1 = uxn + 0.5*dt*(vxn + vx_star)
    uynp1 = uyn + 0.5*dt*(vyn + vy_star)
    
    if fly == True:        # acceleration in case of flying
        ax = 0.0
        ax_star = 0.0
        ay = -g
        ay_star = -g
    else:
        f_1Aux_star = f_1A(ux_star)
        f_2Aux_star = f_2A(ux_star)
        K_star = 1.0 + f_1Aux_star**2
        Rx_star, Ry_star = friction(f_1A, f_2A, ux_star, uy_star, vx_star, vy_star)
        ax_star = (f_2Aux_star*(vx_star)**2 + g) / K_star * ( - f_1Aux_star) + Rx_star
        if Body == 1 and ux_star <= comp:
            ax_star += -D/m*(ux_star - 0.8)
        ay_star = (f_2Aux_star*(vx_star)**2 + g) / K_star - g + Ry_star

   
    if fly == False and ay < -g and ay_star < -g:  # take off is possible, 
        ay = -g                                    # if absolute value of y-component of acceleration is greater g 
        ay_star = -g                               # and the direction of ay is downwards 
        ax = 0.0
        ax_star = 0.0
        
    vxnp1 = vxn + dt*0.5*(ax + ax_star)
    vynp1 = vyn + dt*0.5*(ay + ay_star)
                        
    accnp1 = sqrt(ax**2 + ay**2)
    
    if fly == False and ( (xD < uxn) and (f(uxn) + 1.0e-6) < uyn):  # flying if y-component of position is greater y-value of the rail
        print("starte Flug bei ux[{0}]={1:10.9}m f(ux[{0}])={2:10.9}m uy[{0}]={3:10.9}m"
              .format(n,uxn,f(uxn),uyn))
        fly = True

    if fly == False:  
        Rx, Ry = friction(f_1A, f_2A, uxnp1, uynp1, vxnp1, vynp1)
        FRxnp1 = m * Rx   # x-component of friction force 
        FRynp1 = m * Ry   # y-component of friction force
        rmse_y += (f(uxnp1) - uynp1)**2
        nr_rmse_y += 1 
    else:             # flying is without friction
        FRxnp1 = 0.0
        FRynp1 = 0.0

    return fly, uxnp1, uynp1, vxnp1, vynp1, accnp1, FRxnp1, FRynp1, WR, rmse_y, nr_rmse_y  
    
N_t = int(round(T/dt))
print("Anzahl Zeitschritte N_t: {0}".format(N_t))
t = linspace(0, N_t*dt, N_t+1)

ux = zeros(N_t+1)    # x-component of position vector u
vx = zeros(N_t+1)    # x-component of velocity vector v
uy = zeros(N_t+1)    # y-component of position vector u
vy = zeros(N_t+1)    # y-component of velocity vector v
acc = zeros(N_t+1)
vabs = zeros(N_t+1)
ds_arr = zeros(N_t+1)
FRx = zeros(N_t+1)   # x-component of friction force
FRy = zeros(N_t+1)   # y-component of friction force
FRabs = zeros(N_t+1) 
wr = zeros(N_t+1)    # differential work 
Ekin = zeros(N_t+1)  # kinetic energy
Epot = zeros(N_t+1)  # potential energy
wr_tot = zeros(N_t+1)# total work (-(inner energy) )
Etot = zeros(N_t+1)  # total energy  
Emech = zeros(N_t+1) # mechanical energy
Ein = zeros(N_t+1)   # thermal energy

wr2 = zeros(N_t+1)    # differential work 
Ekin2 = zeros(N_t+1)  # kinetic energy
Epot2 = zeros(N_t+1)  # potential energy
wr_tot2 = zeros(N_t+1)# total work (-(inner energy) )
Etot2 = zeros(N_t+1)  # total energy  
Emech2 = zeros(N_t+1) # mechanical energy
Ein2 = zeros(N_t+1)   # thermal energy

Esp = zeros(N_t+1)   # spring energy
Epot_ges = zeros(N_t+1)
Ekin_ges = zeros(N_t+1)
Emech_ges = zeros(N_t+1)
Ein_ges = zeros(N_t+1)
Etot_ges = zeros(N_t+1)

ux_gf = zeros(N_t+1)    # x-component of position vector u
vx_gf = zeros(N_t+1)    # x-component of velocity vector v
uy_gf = zeros(N_t+1)    # y-component of position vector u
vy_gf = zeros(N_t+1)    # y-component of velocity vector v
acc_gf = zeros(N_t+1)
vabs_gf = zeros(N_t+1)


rateValue = 1.0/dt
show_slow_motion = True
schowEgraphGes = False
schowAtgraph =  False
schowVtgraph =  False
cnt = 0

while cnt < 2:
    # Initial conditions
    U_0 = 0.0                           # initial x-coordinate
    V_0 = 0.0                           # initial magnitude of velocity
    ux[0] = U_0
    #vx[0] = V_0                        # if V_0 is initial x-component of velocity
    vx[0] = V_0 / (sqrt(1.0 + (f_1A(U_0))**2))
    uy[0] = f(U_0)
    #vy[0] = V_0 * f_1A(U_0)            # if V_0 is initial x-component of velocity
    vy[0] = V_0 * f_1A(U_0) / (sqrt(1.0 + (f_1A(U_0))**2))

    vabs[0] = sqrt(vx[0]**2 + vy[0]**2)

    ux_gf[0] = xB
    V2_0 = 0.0                          # initial magnitude of velocity
    vx_gf[0] = V2_0 / (sqrt(1.0 + (gf_1A(ux_gf[0]))**2))
    uy_gf[0] = f(ux_gf[0])
    vy_gf[0] = V2_0 * gf_1A(ux_gf[0]) / (sqrt(1.0 + (gf_1A(ux_gf[0]))**2))
    vabs_gf[0] = sqrt(vx_gf[0]**2 + vy_gf[0]**2)

    Rx, Ry = friction(f_1A, f_2A, ux[0], uy[0], vx[0], vy[0])
    f_1Aux = f_1A(ux[0])
    f_2Aux = f_2A(ux[0])
    K = 1.0 + f_1Aux**2
    ax = (f_2Aux*(vx[0])**2 + g) / K * ( - f_1Aux) + Rx
    ay = (f_2Aux*(vx[0])**2 + g) / K - g + Ry

    acc[0] = ax

    Rx2, Ry2 = friction(gf_1A, gf_2A, ux_gf[0], uy_gf[0], vx_gf[0], vy_gf[0])
    gf_1Aux = gf_1A(ux_gf[0])
    gf_2Aux = gf_2A(ux_gf[0])
    K2 = 1.0 + gf_1Aux**2
    ax = (gf_2Aux*(vx_gf[0])**2 + g) / K2 * ( - gf_1Aux) + Rx2
    ay = (gf_2Aux*(vx_gf[0])**2 + g) / K2 - g + Ry2
    acc_gf[0] = sqrt(ax**2 + ay**2)

    thickness = 0.05
    height = 0.1
    width = 0.8

    rct = shapes.rectangle(width=width, height=height, thickness=thickness, roundness=0.1, invert=True)
    hole = shapes.rectangle(width=width/3, height=height, thickness=thickness, roundness=0.1, invert=True)

    shp = rct

    x = arange( x_offset, xD, 0.01 )  # x interval of the curve
    y = f(x)

    extr = extrusion(x=x,y=y, shape=shp, color = color.red)
    zyl_radius=0.06
    xgrd = arange( xD, xZ, 0.01 )
    ground = extrusion(x=xgrd,y=0.0, shape=shp)

    L = 0.4
    H = 0.4
    W = 0.5
    uxp = ux[0] + comp - (H/2 + height/2)*sin(arctan(f_1Aux))
    uyp = f(ux[0]) + (H/2 + height/2)*cos(arctan(f_1Aux))

    block = sphere(pos=(uxp,uyp,0),radius=H/2, color=color.blue,opacity=0.8)

    block.velocity = vector(vx[0],vy[0],0)
    radiusSpring = H/2
     
    xSpring = x_offset
    ySpring = f(xSpring) + (radiusSpring + height/2)*cos(arctan(f_1Aux))
    lengthSpring = abs(x_offset) - H/2 + comp
    spring = helix(pos=(xSpring,ySpring,0.0),axis=(L,H*f_1A(x_offset),0),radius=radiusSpring, length=lengthSpring, coils=15,thickness=0.02,color=color.yellow)


    radius_Sph2 = 0.2
    xSph2 = xB - (radius_Sph2 + height/2)*sin(arctan(f_1Aux))
    ySph2 = f(xSph2) + (radius_Sph2 + height/2)*cos(arctan(f_1Aux)) 
    sph2 = sphere(pos=(xSph2,ySph2,0), radius=radius_Sph2, color=color.green,opacity=0.8)
    sph2.velocity = vector(0.0,0.0,0.0)

    FRx[0] = m * Rx
    FRy[0] = m * Ry
    FRabs[0] = sqrt(FRx[0]**2 + FRy[0]**2)
    wr[0] = 0.0
    WR = 0.0
    Ekin[0] = 0.5*m*( (vx[0])**2 + (vy[0])**2 )
    Epot[0] = m * g * uy[0]                     # reference value Epot,R = 0 at y = 0
    Em = Ekin[0] + Epot[0]
    Emech[0] = Ekin[0] + Epot[0]
    Etot[0] = Emech[0] + abs(WR)
    Et = Em + abs(WR)
    wr_tot[0] = wr[0]
    Et_initial = Em + abs(WR)

    print("Anfangswerte (blaue Kugel): t={0:6.3} v={1:10.9}m/s Ekin={2:10.9}J  Epot={3:10.9}J  Emech={4:10.9}J  WR={5:10.9}J  EG={6:10.9}J".
          format(t[0],vabs[0],Ekin[0],Epot[0],Em,WR,Et_initial)) 


    wr2[0] = 0.0
    WR2 = 0.0
    Ekin2[0] = 0.5*m2*( (vx_gf[0])**2 + (vy_gf[0])**2 )
    Epot2[0] = m2 * g * uy_gf[0]                     # reference value Epot,R = 0 at y = 0
    Em2 = Ekin2[0] + Epot2[0]
    Emech2[0] = Ekin2[0] + Epot2[0]
    Etot2[0] = Emech2[0] + abs(WR2)
    Et2 = Em2 + abs(WR2)
    wr_tot2[0] = wr2[0]
    Et2_initial = Em2 + abs(WR2)

    print("Anfangswerte (grüne Kugel): t={0:6.3} Ekin={1:10.9}J  Epot={2:10.9}J  Emech={3:10.9}J  WR={4:10.9}J  EG={5:10.9}J".
          format(t[0],Ekin2[0],Epot2[0],Em2,WR2,Et2_initial)) 

    Esp[0] = 0.5*D*(comp - ux[0])**2

    print("Anfangswert der Spannenergie: t={0:6.3} Esp={1:10.9}J".format(t[0],Esp[0])) 

    Epot_ges[0] = Epot[0] + Epot2[0]
    Ekin_ges[0] = Ekin[0] + Ekin2[0]
    Emech_ges[0] = Epot_ges[0]+ Ekin_ges[0] + Esp[0]
    Ein_ges[0] = abs(WR) + abs(WR2)
    Etot_ges[0] = Emech_ges[0] + Ein_ges[0]
    Et_ges_initial = Et_initial + Et2_initial + Esp[0]

    print("Anfangswerte: t={0:6.3} Ekin_ges={1:10.9}J  Epot_ges={2:10.9}J  Emech_ges={3:10.9}J  Ein_ges={4:10.9}J  E_ges={5:10.9}J".
          format(t[0],Ekin_ges[0],Epot_ges[0],Emech_ges[0],Ein_ges[0],Etot_ges[0])) 
    s = 0.0         # for summation of differential path elements
    s_fly = 0.0
    s_rail = 0.0
    s2 = 0.0  
    s_fly2 = 0.0
    s_rail2 = 0.0
    rmse_y = 0.0
    nr_rmse_y = 0
    mean_abs_diff_Eges = 0.0
    ux_min = ux[0]
    ux_max = ux[0]
    n_max = 0
    max_friction_force = FRabs[0]
    n_max_fr = 0 
    x_max_fr = ux[0]
    v_max_fr = 0.0

    WR_gf = 0.0
    rmse_y_gf = 0.0
    nr_rmse_y_gf = 0

    # color scheme for graphs
    #
    BW=1
    if BW==1:
        fgcolor=color.black; bgcolor=color.white
    else:
        fgcolor=color.white; bgcolor=color.black
    b1color=color.red
    b2color=(0,.75,0)

    #
    # energy graphs
    #
    ymax = 35.0
    labeloffset = 0.05*ymax 
    Egraph = gdisplay(x=0, y=350, width=700, height=400, 
                 title='Energiediagramm blaue Kugel', xtitle='t/s', ytitle='E/J', 
                 xmax=T, xmin=0.0, ymax=ymax, ymin=0.0, 
                 foreground=fgcolor, background=bgcolor)

    label(display=Egraph.display, pos=(0.3,ymax-labeloffset), text=("cyan: Epot"))
    label(display=Egraph.display, pos=(0.8,ymax-labeloffset), text=("blue: Ekin"))
    label(display=Egraph.display, pos=(1.3,ymax-labeloffset), text=("green: Emech"))
    label(display=Egraph.display, pos=(1.9,ymax-labeloffset), text=("red: Eth"))
    label(display=Egraph.display, pos=(2.5,ymax-labeloffset), text=("black: Etot"))
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

    Egraph2 = gdisplay(x=700, y=350, width=700, height=400, 
                 title='Energiediagramm gruene Kugel', xtitle='t/s', ytitle='E/J', 
                 xmax=T, xmin=0.0, ymax=ymax, ymin=0.0, 
                 foreground=fgcolor, background=bgcolor)
    label(display=Egraph2.display, pos=(0.3,ymax-labeloffset), text=("cyan: Epot"))
    label(display=Egraph2.display, pos=(0.8,ymax-labeloffset), text=("blue: Ekin"))
    label(display=Egraph2.display, pos=(1.3,ymax-labeloffset), text=("green: Emech"))
    label(display=Egraph2.display, pos=(1.9,ymax-labeloffset), text=("red: Eth"))
    label(display=Egraph2.display, pos=(2.5,ymax-labeloffset), text=("black: Etot"))
    gEpot2 = gcurve(color=color.cyan)
    gEkin2 = gcurve(color = color.blue)
    gEmech2 = gcurve(color=color.green)
    gEin2 = gcurve(color = color.red)
    gEtot2 = gcurve(color = color.black)
    gEkin2.plot(pos=(t[0],Ekin2[0]))
    gEpot2.plot(pos=(t[0],Epot2[0]))
    gEmech2.plot(pos=(t[0],Emech2[0]))
    gEin2.plot(pos=(t[0],Ein2[0]))
    gEtot2.plot(pos=(t[0],Etot2[0]))


    if schowEgraphGes ==  True:
        EgraphGes = gdisplay(x=0, y=000, width=650, height=400, 
                     title='Energiediagramm', xtitle='t/s', ytitle='E/J', 
                     xmax=T, xmin=0.0, ymax=ymax, ymin=0.0, 
                     foreground=fgcolor, background=bgcolor)
        label(display=EgraphGes.display, pos=(0.3,ymax-0.5), text=("Esp: cyan"))
        label(display=EgraphGes.display, pos=(0.7,ymax-0.5), text=("Epot: gelb"))
        label(display=EgraphGes.display, pos=(1.1,ymax-0.5), text=("Ekin: blau"))
        label(display=EgraphGes.display, pos=(1.5,ymax-0.5), text=("Emech: grün"))
        label(display=EgraphGes.display, pos=(1.9,ymax-0.5), text=("E_in: rot"))
        label(display=EgraphGes.display, pos=(2.4,ymax-0.5), text=("Eges: schwarz"))
        gEsp_ges = gcurve(color = color.cyan)
        gEpot_ges = gcurve(color=color.yellow)
        gEkin_ges = gcurve(color = color.blue)
        gEmech_ges = gcurve(color=color.green)
        gEin_ges = gcurve(color = color.red)
        gEtot_ges = gcurve(color = color.black)
        gEsp_ges.plot(pos=(t[0],Esp[0]))
        gEkin_ges.plot(pos=(t[0],Ekin_ges[0]))
        gEpot_ges.plot(pos=(t[0],Epot_ges[0]))
        gEmech_ges.plot(pos=(t[0],Emech_ges[0]))
        gEin_ges.plot(pos=(t[0],Ein_ges[0]))
        gEtot_ges.plot(pos=(t[0],Etot_ges[0]))

    if schowAtgraph ==  True:
        Atgraph = gdisplay(x=0, y=450, width=650, height=400, 
                     title='t-a-Diagramm', xtitle='t/s', ytitle='a/m/s²', 
                     xmax=T, xmin=0, ymax=30, ymin=0, 
                     foreground=fgcolor, background=bgcolor)
        label(display=Atgraph.display, pos=(8,18), text=("a: blau"))
        label(display=Atgraph.display, pos=(8,20), text=("a: grün"))
        gAt = gcurve(color = color.blue)
        gAt_gf = gcurve(color = color.green)

        gAt.plot(pos=(t[0],acc[0]))
        if SecondPath:
            gAt_gf.plot(pos=(t[0],acc_gf[0]))
    if schowVtgraph ==  True:
        Vtgraph = gdisplay(x=0, y=450, width=650, height=400, 
                     title='t-v-Diagramm', xtitle='t/s', ytitle='v/m/s', 
                     xmax=T, xmin=0, ymax=10, ymin=0, 
                     foreground=fgcolor, background=bgcolor)
        label(display=Vtgraph.display, pos=(8,18), text=("v: blau"))
        label(display=Vtgraph.display, pos=(8,20), text=("v: grün"))
        gVt = gcurve(color = color.blue)
 
        gVt_gf = gcurve(color = color.green)

        gVt.plot(pos=(t[0],vabs[0]))

        gVt_gf.plot(pos=(t[0],vabs_gf[0]))

    spt = 1
    time.sleep(spt)

    spring.length = lengthSpring - comp
    block.x = uxp - comp      # compress spring

    spt = 2
    time.sleep(spt)

    movBlock = True
 
    movBlock2 = False

    uy_B1_f = uy[0]
    fly = False
    fly2 = False
    
    starttime = time . time ()
    time_max_fr = starttime

    for n in range(N_t):        # loop over time

        rate(rateValue)
        
        #if scene.mouse.clicked:
        #    scene.mouse.getclick()
        
        tn = t[n]
        
        if ux[n] >= xB and movBlock2 == False:
            v1_mag = sqrt(vx[n]**2 + vy[n]**2)
            v2_mag = sqrt(vx_gf[n]**2 + vy_gf[n]**2)
            u1, u2 = collision(m, m2, v1_mag, v2_mag)
            vx[n] = u1 / (sqrt(1.0 + (f_1A(ux[n]))**2))
            vy[n] = u1 * f_1A(ux[n]) / (sqrt(1.0 + (f_1A(ux[n]))**2))
            ux_gf[n] = ux[n]
            uy_gf[n] = f(ux_gf[n])
            vx_gf[n] = u2 / (sqrt(1.0 + (f_1A(ux_gf[n]))**2))
            vy_gf[n] = u2 * f_1A(ux_gf[n]) / (sqrt(1.0 + (f_1A(ux_gf[n]))**2))
            movBlock2 = True
       
        if movBlock == True:
            Body = 1
            fly, uxnp1, uynp1, vxnp1, vynp1, accnp1, FRxnp1, FRynp1, WR, rmse_y, nr_rmse_y = oneStep(Body, fly, m, n, f, f_1A, f_2A, ux[n], uy[n], vx[n], vy[n], WR, rmse_y, nr_rmse_y)

            ux[n+1] = uxnp1
            uy[n+1] = uynp1
            vx[n+1] = vxnp1
            vy[n+1] = vynp1
            acc[n+1] = accnp1 
            FRx[n+1] = FRxnp1
            FRy[n+1] = FRynp1
            frx = FRx[n+1]
            fry = FRy[n+1]

            if ux[n+1] > ux_max:
                ux_max = ux[n+1] 
            if ux[n+1] < ux_min:
                ux_min = ux[n+1]     
            dx = ux[n+1] - ux[n]
            dy = uy[n+1] - uy[n]
            ds = sqrt(dx**2 + dy**2)  # differential path element
            ds_arr[n+1] = ds 
            s += ds
            if fly == False:
                s_rail += ds
            else:
                s_fly += ds
            fr_abs = sqrt( frx**2 + fry**2)  # absolute value of the friction force
            FRabs[n+1] = fr_abs
            WR += - fr_abs * ds
            wr[n+1] = - fr_abs * ds
            wr_tot[n+1] = WR
            Ein[n+1] = -WR
            Ekin[n+1] = 0.5*m*( (vx[n+1])**2 + (vy[n+1])**2 )
            Epot[n+1] = m * g * uy[n+1]
            Em = Ekin[n+1] + Epot[n+1]
            Et = Em + abs(WR)
            Emech[n+1] = Ekin[n+1] + Epot[n+1]
            Etot[n+1] = Emech[n+1] + abs(wr_tot[n+1])
            #Etot[n+1] = Emech[n+1] + abs(WR)
                    
            v_abs = sqrt((vx[n+1])**2 + (vy[n+1])**2)
            vabs[n+1] = v_abs

            if fr_abs > max_friction_force:
               max_friction_force = fr_abs
               v_max_fr = v_abs
               x_max_fr = ux[n+1]
               time_max_fr = time . time () - starttime
               n_max_fr = n+1 

            f_1Aux = f_1A(ux[n+1])
            grad_ang = arctan(f_1Aux)
            block.x = ux[n+1] - (H/2 + height/2)*sin(grad_ang)
            block.velocity.x = vx[n+1]
            block.y = uy[n+1] + (H/2 + height/2)*cos(grad_ang)
            block.velocity.y = vy[n+1]
            if ux[n+1] <= comp:
                spring.length = lengthSpring - comp + float(ux[n+1])
                

        if movBlock2 == True:
            Body = 2
            fly2, uxnp1, uynp1, vxnp1, vynp1, accnp1, FRxnp1, FRynp1, WR2, rmse_y_gf, nr_rmse_y_gf = oneStep(Body, fly2, m2, n, gf, gf_1A, gf_2A, ux_gf[n], uy_gf[n], vx_gf[n], vy_gf[n], WR2, rmse_y_gf, nr_rmse_y_gf)
            ux_gf[n+1] = uxnp1
            uy_gf[n+1] = uynp1
            vx_gf[n+1] = vxnp1
            vy_gf[n+1] = vynp1
            acc_gf[n+1] = accnp1
        else:
            ux_gf[n+1] = xB
            uy_gf[n+1] = f(xB)
            vx_gf[n+1] = 0.0
            vy_gf[n+1] = 0.0
            acc_gf[n+1] = 0.0
            
        v_gf_abs = sqrt((vx_gf[n+1])**2 + (vy_gf[n+1])**2)
        vabs_gf[n+1] = v_gf_abs

        
        frx = FRxnp1
        fry = FRynp1

        dx = ux_gf[n+1] - ux_gf[n]
        dy = uy_gf[n+1] - uy_gf[n]
        ds = sqrt(dx**2 + dy**2)  # differential path element

        s2 += ds
        if fly2 == False:
            s_rail2 += ds
        else:
            s_fly2 += ds
        fr_abs = sqrt( frx**2 + fry**2)  # absolute value of the friction force
 
        WR2 += - fr_abs * ds
        wr2[n+1] = - fr_abs * ds
        wr_tot2[n+1] = WR2
        Ein2[n+1] = -WR2
        Ekin2[n+1] = 0.5*m2*( (vx_gf[n+1])**2 + (vy_gf[n+1])**2 )
        Epot2[n+1] = m2 * g * uy_gf[n+1]
        Em2 = Ekin2[n+1] + Epot2[n+1]
        Et2 = Em2 + abs(WR2)
        Emech2[n+1] = Ekin2[n+1] + Epot2[n+1]
        Etot2[n+1] = Emech2[n+1] + abs(wr_tot2[n+1])

        if ux[n+1] <= comp:
            Esp[n+1] = 0.5*D*(comp - ux[n+1])**2
        else:
            Esp[n+1] = 0.0

        Epot_ges[n+1] = Epot[n+1] + Epot2[n+1]
        Ekin_ges[n+1] = Ekin[n+1] + Ekin2[n+1]
        Emech_ges[n+1] = Epot_ges[n+1]+ Ekin_ges[n+1] + Esp[n+1]
        Ein_ges[n+1] = abs(WR) + abs(WR2)
        Etot_ges[n+1] = Emech_ges[n+1] + Ein_ges[n+1]

        mean_abs_diff_Eges += abs(Etot[n+1] + Etot2[n+1] + Esp[n+1] - Et_ges_initial)

        gf_1Aux = gf_1A(ux_gf[n+1])
        grad_ang = arctan(gf_1Aux)

        if movBlock2 == True:
            sph2.x = ux_gf[n+1] - (H/2 + height/2)*sin(grad_ang)
            sph2.velocity.x = vx_gf[n+1]
            sph2.y = uy_gf[n+1] + (H/2 + height/2)*cos(grad_ang)
            sph2.velocity.y = vy_gf[n+1]
        
        if movBlock == True:
            gEkin.plot(pos=(t[n+1],Ekin[n+1]))
            gEpot.plot(pos=(t[n+1],Epot[n+1]))
            gEmech.plot(pos=(t[n+1],Emech[n+1]))
            gEin.plot(pos=(t[n+1],Ein[n+1]))
            gEtot.plot(pos=(t[n+1],Etot[n+1]))
            if schowAtgraph ==  True:
                gAt.plot(pos=(t[n+1],acc[n+1]))
            if schowVtgraph ==  True:    
                gVt.plot(pos=(t[n+1],vabs[n+1]))
       
        gEkin2.plot(pos=(t[n+1],Ekin2[n+1]))
        gEpot2.plot(pos=(t[n+1],Epot2[n+1]))
        gEmech2.plot(pos=(t[n+1],Emech2[n+1]))
        gEin2.plot(pos=(t[n+1],Ein2[n+1]))
        gEtot2.plot(pos=(t[n+1],Etot2[n+1]))
        if schowAtgraph ==  True:
            gAt_gf.plot(pos=(t[n+1],acc_gf[n+1]))
        if schowVtgraph ==  True:
            gVt_gf.plot(pos=(t[n+1],vabs_gf[n+1]))

        if schowEgraphGes ==  True:
            gEsp_ges.plot(pos=(t[n+1],Esp[n+1]))
            gEkin_ges.plot(pos=(t[n+1],Ekin_ges[n+1]))
            gEpot_ges.plot(pos=(t[n+1],Epot_ges[n+1]))
            gEmech_ges.plot(pos=(t[n+1],Emech_ges[n+1]))
            gEin_ges.plot(pos=(t[n+1],Ein_ges[n+1]))
            gEtot_ges.plot(pos=(t[n+1],Etot_ges[n+1]))

        n_max = n+1

    endtime = time . time ()
    difftime = endtime - starttime
    print("System Zeit: {0:10.8}s".format(difftime))

    spt = 3
    time.sleep(spt)
    
    if show_slow_motion == True:
        extr.visible = 0
        del(extr)
        ground.visible = 0
        del(ground)
        block.visible = 0
        del(block)
        spring.visible = 0
        del(spring)
        sph2.visible = 0
        del(sph2)

    
        #title = text(pos=(6,0.5,1),text="Slow Motion", align='center', depth=-0.3, color=color.black)
        
        spt = 1
        time.sleep(spt)
       
        #title.visible = 0
        rateValue = rateValue / 4.0
        
        show_slow_motion = False
    else:
        break

    cnt += 1
    
n = n_max - 1
print("max n: {0} Endwerte (blaue Kugel): Ekin={1:10.9}J  Epot={2:10.9}J  Emech={3:10.9}J WR={4:10.9}J Ein={5:10.9}J Eges={6:10.9}J".
        format(n,Ekin[n+1],Epot[n+1],Emech[n+1],wr_tot[n+1],Ein[n+1],Etot[n+1]))
print("max n: {0} Endwerte (grüne Kugel): Ekin={1:10.9}J  Epot={2:10.9}J  Emech={3:10.9}J WR={4:10.9}J Ein={5:10.9}J Eges={6:10.9}J".
        format(n,Ekin2[n+1],Epot2[n+1],Emech2[n+1],wr_tot2[n+1],Ein2[n+1],Etot2[n+1]))
print("Endwert der Spannenergie: t={0:6.3} Esp={1:10.9}J".format(t[n+1],Esp[n+1]))
print("Endwerte: t={0:6.3} Ekin_ges={1:10.9}J  Epot_ges={2:10.9}J  Emech_ges={3:10.9}J  Ein_ges={4:10.9}J  E_ges={5:10.9}J".
      format(t[n+1],Ekin_ges[n+1],Epot_ges[n+1],Emech_ges[n+1],Ein_ges[n+1],Etot_ges[n+1])) 
 
Et_ges_final = Esp[n+1] + Emech[n+1] + abs(wr_tot[n+1]) + Emech2[n+1] + abs(wr_tot2[n+1])

EG_diff = Et_ges_initial - Et_ges_final
print("Differenz (EG Anfang) - (EG Ende): {0:10.9}J".format(EG_diff))
mean_abs_diff_Eges /= n_max
print("mittlere absolute Abweichung der Gesamtenergie: {0:10.9}J".format(mean_abs_diff_Eges))
rmse_y /= nr_rmse_y
rmse_y = np.sqrt(rmse_y)
a = ux_min
b = ux_max
intgr_trap = trap(df,1000, a, b,0)

print("Fehler (RMSE) der numerisch berechneten y-Werte der Bahnkurve: {0:9.7}m".format(rmse_y))
print("Länge der Bahnkurve: {0:9.7}m zwischen xmin={1:9.7}m und xmax={2:9.7}m"
      .format(abs(intgr_trap),a,b))
print("gesamte zurückgelegte Strecke der blauen Kugel: {0:9.7}m".format(s))
print("zurückgelegte Strecke der blauen Kugel auf der Bahn: {0:9.7}m".format(s_rail))
print("zurückgelegte Strecke der blauen Kugel im Flug: {0:9.7}m".format(s_fly))
print("gesamte zurückgelegte Strecke der grünen Kugel: {0:9.7}m".format(s2))
print("zurückgelegte Strecke der grünen Kugel auf der Bahn: {0:9.7}m".format(s_rail2))
print("zurückgelegte Strecke der grünen Kugel im Flug: {0:9.7}m".format(s_fly2))

n = n_max - 1
Eges_ende = Esp[n+1] + Ekin[n+1] + Epot[n+1] + abs(WR) + Ekin2[n+1] + Epot2[n+1] + abs(WR2)
print("Eges am Anfang {0:10.9}J  Eges am Ende {1:10.9}J".format(Et_ges_initial,Eges_ende))
