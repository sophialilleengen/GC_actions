import numpy as np
from scipy import interpolate
from scipy import integrate as intg
from scipy import optimize as opt
from scipy import constants as cs
from astropy import units as un
import sys



class GCorbit:
    def __init__(self,r_pc=None,rho_Msunpc3=None,bhmass_Msun=None,inputfilename=None,n=None): 
        """
        NAME:
           __init__
        PURPOSE:
           initialize a GCorbit object - a Globular cluster object which calculates actions and integrates orbits.
        INPUT:
            r_pc = array of radius coordinates in pc (default: None)
            rho_pc..
        OUTPUT:
           instance
        HISTORY:
           2016-01-15 - Written (Milanov, MPIA)
        """

        if inputfilename is not None:
            data=np.loadtxt(inputfilename)        
            self._r_bin  = data[0,:]     #[pc]
            self._rho = data[1,:]     #[M_sun/pc³]
        elif r_pc is None or rho_Msunpc3 is None:
            sys.exit("Error in GCorbit.__init__(): Specify input file or r and rho.")
        else:
            self._r_bin=r_pc        #[pc]
            self._rho=rho_Msunpc3     #[M_sun/pc³]
        self.s=interpolate.InterpolatedUnivariateSpline(np.log(self._r_bin[:]),np.log(self._rho[:]))
        pot_y=self._potential_stars(self._r_bin,density=self.density(self._r_bin),full_integration=True) #[pc²/s²] #potential muss hier scheinbar vom oben definierten r abhängen, soll das so sein? demnach dann auch die Dichte
        self._G=un.m**3/(un.kg*un.s**2).to(un.pc**3/(un.solMass*un.s**2), cs.G) #[pc³/M_sun*s²]
        if bhmass_Msun is None:
            self._bhmass=0
        else:
            self._bhmass=bhmass_Msun    #[M_sun]
        self._n=30 #degree of Gauss-Legendre quadrature
        self._x_i=np.polynomial.legendre.leggauss(self.n)[0]        
        self._w_i=np.polynomial.legendre.leggauss(self.n)[1]


        self.pot=interpolate.InterpolatedUnivariateSpline(self._r_bin,pot_y)
        return None

    def density(self,r):
        """
        NAME:
            density
        PURPOSE:
            returns density at distance r
        INPUT:
            r = array of radius coordinates in pc (default: None)        
        #        self.s = interpolation of density profile
        OUTPUT:
            density
        HISTORY:
            2016-01-16 - Written (Milanov, MPIA)
        """
        return np.exp(self.s(np.log(r)))

    def _potential_stars(self,r,density=None,full_integration=False):   
        """
        NAME:
            _potential_stars
        PURPOSE:
            calculates potential caused by stars in Globular cluster
        INPUT:
            r = array of radius coordinates in pc (default: None)
        OUTPUT:
            potential of the stars
        HISTORY:
            2015-12-03 - Written (Milanov, MPIA)
        """
        low=np.min(r)
        high=np.max(r)
        if np.any(r<low) or np.any(r>high): #s.Z. 37 r wird als array eingegeben, also muss ich hier array befehle benutzen 
                    sys.exit("r is smaller or bigger than star boundaries")

        if full_integration==True:
            if density is None:
                sys.exit("Error in GCorbit._potential_stars(): Specify density.")    
            sum1=np.zeros(self._n)    
            sum2=np.zeros(self._n)
            if isinstance(r,np.ndarray):
                return np.array([potential(rr,density=density,low=low,high=high,x_i=x_i,w_i=w_i) for rr in r])
            else:
                x1=((r-low)/2.)*self._x_i+(r+low)/2.    
                x2=((high-r)/2)*self._x_i+(high+r)/2
                for i in range(n):
                    s1=self.density(x1[i])   
                    s2=self.density(x2[i])
                    sum1[i]=(self._w_i[i]*x1[i]**2*s1)
                    sum2[i]=(self._w_i[i]*x2[i]*s2)
                sum_1=np.sum(sum1)
                sum_2=np.sum(sum2)
                return -4*np.pi*self._G*((r-low)/(2*r)*sum_1+(high-r)/2*sum_2)   
        else:
            return self.pot(r)
    
    def _potential_bh(self,r):
        """
        NAME:
            _potential_bh
        PURPOSE:
            calculates potential caused by central black hole
        INPUT:
            r = array of radius coordinates in pc (default: None)
        OUTPUT:
            potential of black hole
        HISTORY:
            2015-12-10 - Written (Milanov, MPIA)
        """
        return -self._G*self._bhmass/r


    def potential(self,r):
        """
        NAME:
            potential
        PURPOSE:
            adds star and black hole potential
        INPUT:
            r = array of radius coordinates in pc (default: None)
        OUTPUT:
            total potential in Globular cluster
        HISTORY:
            2016-01-16 - Written (Milanov, MPIA)
        """
        return self._potential_stars(r)+self._potential_bh(r)
    
    def _r_derivative(self,r):   
        """
        NAME:
            _r_derivative
        PURPOSE:
            calculates derivative of interpolated potential
        INPUT:
            r = array of radius coordinates in pc (default: None)
        OUTPUT:
            r-derivative of potential
        HISTORY:
            2016-01-14 - Written (Milanov, MPIA)
        """    
        der=self._potential_stars.derivative()
        return der(r)+self._G*self._bhmass/(r**2)
        #20. I think having this function is good, but it should do something else. It could have a keyword explicit_derivative=False. In the __init__ function you could call this function once with explicit_derivative=True and calculate the derivative of the potential explicitly at a few points. Everywhere else you use the interpolation analogous to the potential.

    def force(self,x,y,z):    
        """
        NAME:
            force    
        PURPOSE:
            calculates force from potential    
        INPUT:
            x,y,z,r
        OUTPUT:
            force array
        HISTORY:
            2016-01-14 - Written (Milanov, MPIA)
        """
        force=np.array(3)
        r=np.sqrt(x**2+y**2+z**2)
        drdx=x/r
        drdy=y/r
        drdz=z/r
        force[0]=self.r_derivative(r)*drdx 
        force[1]=self.r_derivative(r)*drdy
        force[2]=self.r_derivative(r)*drdz
        return force
        

    def orbit_integration(self,x,y,z,vx,vy,vz,dt=None,t_end=None,method='leapfrog'):     #22. Is this the orbit integration function? I would, in addition to the initial star position, have delta_t and t_end (in useful units) as parameters. N is then calculated in the function from that.
        """
        NAME:
            orbit_integration
        PURPOSE:
            integrates the orbit of the star over time
        INPUT:
            x in pc        
            vx in km/s #noch umrechnen    
            t in years has to be a multiple of dt
            dt in years 
        OUTPUT:
            arrays of star position at each time step
            #wieder in eingabeeinheiten ausgeben
        HISTORY:
            
        """    
        t_end_sec=t_end*np.pi*1e7 #Umrechnung Jahre Sekunden
        dt_sec=dt*np.pi*1e7 #hier auch umrechnen genau
        if method == 'leapfrog':
            if dt is None or t_end is none:
                sys.exit("Error in GCorbit.orbit(): Specify dt and t_end.")
            
            N=int(t_end/dt)
            t=np.linspace(0,t_end_sec,N) #[s]?

            xl=np.zeros(N+1)
            yl=np.zeros(N+1)
            zl=np.zeros(N+1)

            x_l=np.sqrt(xl**2+yl**2+zl**2)

            vxl=np.zeros(N+1)
            vyl=np.zeros(N+1)
            vzl=np.zeros(N+1)

            xl[0]=x
            yl[0]=y
            zl[0]=z

            vxl[0]=vx
            vyl[0]=vy
            vzl[0]=vz

            for i in range(N):
                a=self.force(xl[i],yl[i],zl[i]) 
        
                xl[i+1]=xl[i]+vxl[i]*dt_sec+1./2.*a[0]*dt_sec**2
                yl[i+1]=yl[i]+vyl[i]*dt_sec+1./2.*a[1]*dt_sec**2
                zl[i+1]=zl[i]+vzl[i]*dt_sec+1./2.*a[2]*dt_sec**2
        
                a_1=self.force(xl[i+1],yl[i+1],zl[i+1]) 
        
                vxl[i+1]=vxl[i]+1./2.*(a[0]+a_1[0])*dt_sec
                vyl[i+1]=vyl[i]+1./2.*(a[1]+a_1[1])*dt_sec
                vzl[i+1]=vzl[i]+1./2.*(a[2]+a_1[2])*dt_sec
        
            return xl,yl,zl,vxl,vyl,vzl,t 
        
        elif method == 'rk4':
            return None


    #^-- 34. Da du Runge-Kutta und leapfrog implementiert hast, koenntest du ein keyword method='leapfrog' setzen, dass man auch zu 'rk4' setzen kann, wenn man moechte. Dann kannst du einfach hin und her switchen.


##### Versuch 1 zu actions #####

    def angularmom(self,x,y,z,vx,vy,vz):
        """
        NAME:
            angularmom
        PURPOSE:
            calculates angular momentum 
        INPUT:
            
        OUTPUT:
            L, Lx, Ly, Lz
        HISTORY:
            2016-01-14 - Written (Milanov, MPIA)
        """
        Lx=y*vz-z*vy
        Ly=z*vx-x*vz
        Lz=x*vy-y*vx
        L=np.sqrt(Lx**2+Ly**2+Lz**2)
        return L,Lx,Ly,Lz


    def energy(self,x,y,z,vx,vy,vz):
        """
        NAME:
            energy
        PURPOSE:
            calculates energy of star at its actual position
        INPUT:

        OUTPUT:
            energy
        HISTORY:
            2016-01-14 - Written (Milanov, MPIA)
        """    
        r=np.sqrt(x**2+y**2+z**2)
        pot=self.potential(r)        #
        E=vx**2./2.+vy**2./2.+vz**2./2.+pot
        return E

    def _periapocenter_aux(self,r,E,L):
        """
        NAME:
            _periapocenter_aux
        PURPOSE:
            gives function to solve in periapocenter
        INPUT:
            
        OUTPUT:
            
        HISTORY:
            2016-01-19 - Written (Milanov, MPIA)
        """    
        return(1./r)**2.+2.*(self.potential(r)-E)/L**2.         

    def periapocenter(self,x,y,z,vx,vy,vz):
        """
        NAME:
            periapocenter
        PURPOSE:
            calculates pericenter and apocenter of orbit
        INPUT:
            
        OUTPUT:
            rmin as pericenter, rmax as apocenter
        HISTORY:
            2016-01-16 - Written (Milanov, MPIA)
        """
        r=np.sqrt(x**2.+y**2.+z**2.)
        E=self.energy(x,y,z,vx,vy,vz)
        L=self.angularmom(x,y,z,vx,vy,vz)[0]
        rmin=opt.fsolve(self._periapocenter_aux,x0=np.min(self._r_bin),args=(E,L))
        rmax=opt.fsolve(self._periapocenter_aux,x0=np.max(self._r_bin),args=(E,L))
        if rmin == rmax:
            sys.exit('Error in periapocenter(): rmin=rmax.')
        if rmin > rmax:
            r_temp=rmax
            rmax=rmin
            rmin=r_temp
        return rmin,rmax    
    

    def _j_rint(self,r,E,L):
        """
        NAME:
            _j_rint
        PURPOSE:
            calculates integral needed for J_r action
        INPUT:

        OUTPUT:
            calculated integral
        HISTORY:
            2015-12-04 - Written (Milanov, MPIA)
        """
        pot=self.potential(r)
        return np.sqrt(2.*E-2.*pot-L**2./r**2.)
    #29. j_rint ist doch die funktion die dann in J_r als integrand im Integral aufgerufen wird, oder? Das Integral ist ueber r . Das heisst, du uebergibst dieser Funktion NUR das r und ausserdem die Konstanten E und L, die du schon vorher in J_r ausgerechnet hast.

    def _J_phi(self,x,y,z,vx,vy,vz):
        """
        NAME:
            _J_phi
        PURPOSE:    
            calculates action J_phi
        INPUT:

        OUTPUT:
            J_phi
        HISTORY:
            2015-11-26 - Written (Milanov, MPIA)
        """
        Lz=self.angularmom(x,y,z,vx,vy,vz)[3]
        J_phi=Lz
        return J_phi
    
    def _J_theta(self,x,y,z,vx,vy,vz):
        """
        NAME:
            _J_theta
        PURPOSE:
            calculates action J_theta
        INPUT:
            
        OUTPUT:
            J_theta
        HISTORY:
            2015-11-26 - Written (Milanov, MPIA)
        """
        L=self.angularmom(x,y,z,vx,vy,vz)[0]
        Lz=self.angularmom(x,y,z,vx,vy,vz)[1]
        J_theta=L-np.abs(Lz)
        return J_theta

### J_r beim Integral unsicher wegen Argumenten f�r j_rint und wegen Apo- und Pericenter ###

    def _J_r(self,x,y,z,vx,vy,vz):
        """
        NAME:
            _J_r
        PURPOSE:
            calculates J_r action
        INPUT:

        OUTPUT:
            J_r
        HISTORY:
            2015-12-04 - Written (Milanov, MPIA)
        """
        rmin,rmax=self.periapocenter(x,y,z,vx,vy,vz)           
        E=self.energy(x,y,z,vx,vy,vz)
        L=self.angularmom(x,y,z,vx,vy,vz)[0]
        J_r=1/np.pi*intg.quad(j_rint,rmin,rmax,args=(E,L))[0] 
        return J_r
    
    def actions(self,x,y,z,vx,vy,vz):
        """
        NAME:
            actions
        PURPOSE:
            returns actions
        INPUT:

        OUTPUT:
            actions
        HISTORY:
            2016-01-14 - Written (Milanov, MPIA)
        """
        J_phi=self.J_phi(x,y,z,vx,vy,vz)
        J_theta=self.J_theta(x,y,z,vx,vy,vz)
        J_r=self.J_r(x,y,z,vx,vy,vz)
        return J_phi,J_theta,J_r




