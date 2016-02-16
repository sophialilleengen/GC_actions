import numpy as np
from astropy import units as un
from scipy import constants as cs

class GCphasespace:
    def __init__(self,r,bin_num=20):
        """
        NAME:
            __init__
        PURPOSE:
            initializes a GC object
        INPUT:
            
        OUTPUT:
            instance
        HISTORY:
            2015-09-15 - Written (Milanov, MPIA)
        """
        self._stars=len(r)
        self._bin_num=bin_num
        self._spb=self._stars/self._bin_num
        self._G=(un.m**3/(un.kg*un.s**2)).to(un.pc**3/(un.solMass*un.s**2), cs.G) #[pc³/M_sun*s²]
        self._c=(un.m/un.s).to(un.pc/un.s, cs.c)
        return None

        
    def mean_distance(self,r):
        R_mean=np.zeros(self._bin_num)
        for n in range(self._bin_num):            
            R_mean[n]=np.mean(r[n*self._spb:(n+1)*self._spb]) #berechnet mittlere Entfernung der Sterne in einem bin
        return R_mean

    def velocity_dispersion(self,vr,vtheta,vphi):

        sigrad=np.zeros(self._bin_num)
        sigtheta=np.zeros(self._bin_num)
        sigphi=np.zeros(self._bin_num)
        
        sigraderr=np.zeros(self._bin_num)
        sigthetaerr=np.zeros(self._bin_num)
        sigphierr=np.zeros(self._bin_num)
        
        for n in range(self._bin_num):
            sigrad[n]=np.std(vr[n*self._spb:(n+1)*self._spb])       #[km/s]     #vel disp Wert der radial velocity
            sigtheta[n]=np.std(vtheta[n*self._spb:(n+1)*self._spb]) #vel disp Wert der theta velocity
            sigphi[n]=np.std(vphi[n*self._spb:(n+1)*self._spb])     #vel disp Wert der phi velocity

            sigraderr[n]=sigrad[n]/np.sqrt(2.*self._spb)       #Fehler der radialen Standardabweichung
            sigthetaerr[n]=sigtheta[n]/np.sqrt(2.*self._spb)   #Fehler der azimuthalen Standardabweichung
            sigphierr[n]=sigphi[n]/np.sqrt(2.*self._spb)       #Fehler der polaren Standardabweichung
        
        return sigrad,sigtheta,sigphi,sigraderr,sigthetaerr,sigphierr

    def mean_velocity(self,vr,vtheta,vphi):

        meanrad=np.zeros(self._bin_num)
        meantheta=np.zeros(self._bin_num)
        meanphi=np.zeros(self._bin_num)

        for n in range(self._bin_num):        
            meanrad[n]=np.mean(vr[n*self._spb:(n+1)*self._spb])         #mean Wert der neuen radial velocity
            meantheta[n]=np.mean(vtheta[n*self._spb:(n+1)*self._spb])   #mean Wert der theta velocity
            meanphi[n]=np.mean(vphi[n*self._spb:(n+1)*self._spb])       #mean Wert der phi velocity
        return meanrad,meantheta,meanphi

    def anisotropy_param(self,vr,vtheta,vphi):     
        
        beta=np.zeros(self._bin_num)
        
        sigrad=self.velocity_dispersion(vr,vtheta,vphi)[0] 
        sigphi=self.velocity_dispersion(vr,vtheta,vphi)[2] 
        sigtheta=self.velocity_dispersion(vr,vtheta,vphi)[1] 
        for n in range(self._bin_num):       
            beta[n]=1.-(sigtheta[n]**2.+sigphi[n]**2.)/(2.*sigrad[n]**2.) #anisotropy parameter beta
        return beta

    def density(self,r,m1,m2,start,end,step):
        r0=start
        stepsize=((np.log10(end)-np.log10(r0))/step)
        R=np.zeros(step)
        r_aux=np.logspace(np.log10(r0),np.log10(end),step+1)
        r_i=r_aux[:-1:]
        r_a=r_aux[1::]
        #binwidth=r_a-r_i
        #r_error=binwidth/2

        M = np.zeros(step)
        rho=np.zeros(step)

        for n in range(step):
            inbin=(r_i[n]<r) * (r<=r_a[n]) #r_3d wird verwendet! creates boolean arrays woth true values if star is in bin distance
            M[n]=np.sum(m1[inbin])+np.sum(m2[inbin]) #mass array with both masses of binary system
            rho[n]=M[n]/((r_a[n]**2-r_i[n]**2)*np.pi) #calculates density of bin
            R[n]=np.mean(r[inbin]) #calculates mean distance of bin
            print(n,np.sum(inbin))

    
        #extrabin am anfang
        M_extra=np.sum(m1[r<r0])+np.sum(m2[r<r0])
        rho_extra=M_extra/(r0**2*np.pi)
        R_extra=np.mean(r[r<r0])
        #einfuegen in bereits angefertigte arrays an erster Stelle
        M_final=np.insert(M,0,M_extra)
        rho_final=np.insert(rho,0,rho_extra)
        R_final=np.insert(R,0,R_extra)
        
        return R_final,rho_final

    def event_horizon(self,bh_mass_msun):
        r_s=2.*self._G*bh_mass_msun/(self._c**2.) #[pc]
        return r_s
        
