import numpy as np
from astropy import units as un

import sys
sys.path.insert(0, './')

from GC_Orbit_class import GCorbit

r,x,y,z,vx,vy,vz,r_guide=np.loadtxt('x-y-z_orbit',unpack=True)
bh_orbit=GCorbit(inputfilename='densityfile_IMBH1.txt',bhmass_M_sun=10102)


### calculate circular orbit time ###
v_circ=bh_orbit.v_circ(r) #[pc/s]
T_circ=(2*np.pi*r_guide)/v_circ #[s]
T_circ_yr=un.s.to(un.yr,T_circ) #[yr]


### number of steps ### 
step=1e-12

### calculate orbits ###
n=0 #J_r = 0 and J_theta = 0
m=1 #J_r=0 rest random
K=2 #random orbit
xn,yn,zn,vxn_kms,vyn_kms,vzn_kms,tn_yr=bh_orbit.orbit_integration(x[n],y[n],z[n],vx[n],vy[n],vz[n],dt=step*8.*T_circ_yr[n],t_end=8.*T_circ_yr[n])
np.savetxt('home/milanov/Bachelorarbeit/output/circular_plane_orbit.dat',(xn,yn,zn,vxn_kms,vyn_kms,vzn_kms,tn_yr))

xm,ym,zm,vxm_kms,vym_kms,vzm_kms,tm_yr=bh_orbit.orbit_integration(x[m],y[m],z[m],vx[m],vy[m],vz[m],dt=step*8.*T_circ_yr[m],t_end=8.*T_circ_yr[m])
np.savetxt('home/milanov/Bachelorarbeit/output/circular_orbit.dat',(xm,ym,zm,vxm_kms,vym_kms,vzm_kms,tm_yr))

xk,yk,zk,vxk_kms,vyk_kms,vzk_kms,tk_yr=bh_orbit.orbit_integration(x[k],y[k],z[k],vx[k],vy[k],vz[k],dt=step*8.*T_circ_yr[k],t_end=8.*T_circ_yr[k])
np.savetxt('home/milanov/Bachelorarbeit/output/random_orbit.dat',(xk,yk,zk,vxk_kms,vyk_kms,vzk_kms,tk_yr))

### calculate energy for every timestep of the orbits ###

En = np.zeros(len(xn))
for i in range(len(xn)):
    En[i]=bh_orbit.energy(xn[i],yn[i],zn[i],vxn_kms[i],vyn_kms[i],vzn_kms[i])

np.savetxt('home/milanov/Bachelorarbeit/output/E_orbit_circ_plane.dat', (En))

Em = np.zeros(len(xm))
for i in range(len(xm)):
    Em[i]=bh_orbit.energy(xm[i],ym[i],zm[i],vxm_kms[i],vym_kms[i],vzm_kms[i])

np.savetxt('home/milanov/Bachelorarbeit/output/E_circular_orbit.dat', (Em))

Ek = np.zeros(len(xk))
for i in range(len(xk)):
    Ek[i]=bh_orbit.energy(xk[i],yk[i],zk[i],vxk_kms[i],vyk_kms[i],vzk_kms[i])

np.savetxt('home/milanov/Bachelorarbeit/output/E_random_orbit.dat', (Ek))

### calculate angular momentum for every timestep of the orbits ###

Ln=np.zeros(len(xn))
for i in range(len(xn)):
    Ln[i]=bh_orbit.angularmom(xn[i],yn[i],zn[i],vxn_kms[i],vyn_kms[i],vzn_kms[i])[0]

np.savetxt('home/milanov/Bachelorarbeit/output/L_ncircular_plane_orbit.dat', (Ln))

Lm=np.zeros(len(xm))
for i in range(len(xm)):
    Lm[i]=bh_orbit.angularmom(xm[i],ym[i],zm[i],vxm_kms[i],vym_kms[i],vzm_kms[i])[0]

np.savetxt('home/milanov/Bachelorarbeit/output/L_circular_orbit.dat', (Lm))

Lk=np.zeros(len(xk))
for i in range(len(xk)):
    Lk[i]=bh_orbit.angularmom(xk[i],yk[i],zk[i],vxk_kms[i],vyk_kms[i],vzk_kms[i])[0]

np.savetxt('home/milanov/Bachelorarbeit/output/L_random_orbit.dat', (Lk))

### calculate radial actions for every timestep of the orbits ###

Jrn=np.zeros(len(xn))
rn=np.sqrt(xn**2+yn**2+zn**2)
for i in range(len(xn)):
    Jrn[i],rnmin[i],rnmax[i]=bh_orbit._J_r(rn[i],xn[i],yn[i],zn[i],vxn_kms[i],vyn_kms[i],vzn_kms[i])

np.savetxt('home/milanov/Bachelorarbeit/output/Jr_circular_plane_orbit.dat', (Jrn,rnmin,rnmax))

Jrm=np.zeros(len(xm))
rm=np.sqrt(xm**2+ym**2+zm**2)
for i in range(len(xm)):
    Jrm[i],rmmin[i],rmmax[i]=bh_orbit._J_r(rm[i],xm[i],ym[i],zm[i],vxm_kms[i],vym_kms[i],vzm_kms[i])

np.savetxt('home/milanov/Bachelorarbeit/output/Jr_circular_orbit.dat', (Jrm,rmmin,rmmax))

Jrk=np.zeros(len(xk))
rk=np.sqrt(xk**2+yk**2+zk**2)
for i in range(len(xk)):
    Jrk[i],rkmin[i],rkmax[i]=bh_orbit._J_r(rk[i],xk[i],yk[i],zk[i],vxk_kms[i],vyk_kms[i],vzk_kms[i])

np.savetxt('home/milanov/Bachelorarbeit/output/Jr_random_orbit.dat', (Jrk,rkmin,rkmax))

### calculate guding star radius ###

r_g_n=np.zeros(len(xn))
for i in range(len(xn)):
    r_g_n[i]=bh_orbit.r_guide_root(rn[i],xn[i],yn[i],zn[i],vxn_kms[i],vyn_kms[i],vzn_kms[i])

np.savetxt('home/milanov/Bachelorarbeit/output/rg_circular_plane_orbit.dat', (r_g_n))

r_g_m=np.zeros(len(xm))
for i in range(len(xm)):
    r_g_m[i]=bh_orbit.r_guide_root(rm[i],xm[i],ym[i],zm[i],vxm_kms[i],vym_kms[i],vzm_kms[i])

np.savetxt('home/milanov/Bachelorarbeit/output/rg_circular_orbit.dat', (r_g_m))

r_g_k=np.zeros(len(xk))
for i in range(len(xk)):
    r_g_k[i]=bh_orbit.r_guide_root(rk[i],xk[i],yk[i],zk[i],vxk_kms[i],vyk_kms[i],vzk_kms[i])

np.savetxt('home/milanov/Bachelorarbeit/output/rg_random_orbit.dat', (r_g_k))
