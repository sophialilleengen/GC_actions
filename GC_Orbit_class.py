import numpy as np
from scipy import interpolate
from scipy import integrate as intg
from scipy import optimize as opt
from scipy import constants as cs
from scipy.misc import derivative
from astropy import units as un
import sys



class GCorbit:
    def __init__(self,r_pc=None,rho_M_sunpc3=None,bhmass_M_sun=None,inputfilename=None,n=None): 
        """
        NAME:
            __init__
        PURPOSE:
            initialize a GCorbit object - a Globular cluster object which calculates actions and integrates orbits.
        INPUT:
            r_pc = array of radius coordinates in pc (default: None)
            rho_M_sunpc3 = array of density values depending on r_pc in M_sun/pc³ (default: None)
            bhmass_M_sun = mass of the central black hole in M_sun, (default: None)
            inputfilename = name (and path if not in same repository) of file containing distance and density data (default: None)
            n = degree of Gauss-Legendre quadrature (default: None)
        OUTPUT:
            instance
        HISTORY:
            2016-01-15 - Written (Milanov, MPIA)
        """

        if inputfilename is not None:
            data=np.loadtxt(inputfilename)        
            self._r_bin  = data[0,:]     #[pc]
            self._rho = data[1,:]     #[M_sun/pc³]
        elif r_pc is None or rho_M_sunpc3 is None:
            sys.exit("Error in GCorbit.__init__(): Specify input file or r and rho.")
        else:
            self._r_bin=r_pc        #[pc]
            self._rho=rho_M_sunpc3     #[M_sun/pc³]
        self._n=30 #degree of Gauss-Legendre quadrature
        self.s=interpolate.InterpolatedUnivariateSpline(np.log(self._r_bin[:]),np.log(self._rho[:])) #[M_sun/pc³]
        self._G=(un.m**3/(un.kg*un.s**2)).to(un.pc**3/(un.solMass*un.s**2), cs.G) #[pc³/M_sun*s²]
        if bhmass_M_sun is None:
            self._bhmass=0
        else:
            self._bhmass=bhmass_M_sun    #[M_sun]
        self._x_i=np.polynomial.legendre.leggauss(self._n)[0]        
        self._w_i=np.polynomial.legendre.leggauss(self._n)[1]
        pot_y=self._potential_stars(self._r_bin,density=self.density(self._r_bin),full_integration=True) #[pc²/s²] #statt _r_bin (von der density) r_bin_pot nehmen mit mehr Werten und überprüfen ob logarithmisch oder linear
        self.pot=interpolate.InterpolatedUnivariateSpline(self._r_bin,pot_y)
        return None

    def density(self,r):
        """
        NAME:
            density
        PURPOSE:
            returns density at distance r
        INPUT:
            r = array of radius coordinates in pc       
        OUTPUT:
            density in M_sun/pc³
        HISTORY:
            2016-01-16 - Written (Milanov, MPIA)
        """
        low=np.min(self._r_bin)
        high=np.max(self._r_bin)  #[pc]
        density= np.exp(self.s(np.log(r)))
        if isinstance(r,np.ndarray):
            density[r>high]=0.
            density[r<low]=self._rho[0]
        elif r>high:
            density=0.
        elif r<low:
            density=self._rho[0]
        return  density#[M_sun/pc³]

    def _potential_stars(self,r,density=None,full_integration=False):   
        """
        NAME:
            _potential_stars
        PURPOSE:
            calculates potential caused by stars in Globular cluster
        INPUT:
            r = array of radius coordinates in pc 
            density = density of Globular cluster at distance r in M_sun/pc³ only needed when full_integration=True (default: None)
            full_integration = boolean to have the potential interpolated (default: False)
        OUTPUT:
            potential of the stars in pc²/s²
        HISTORY:
            2015-12-03 - Written (Milanov, MPIA)
        """
        low=0. #np.min(self._r_bin)   #[pc]
        high=np.max(self._r_bin)  #[pc]
        if full_integration==True:
            if density is None:
                sys.exit("Error in GCorbit._potential_stars(): Specify density.")    
            sum1=np.zeros(self._n)    
            sum2=np.zeros(self._n)
            if isinstance(r,np.ndarray):
                return np.array([self._potential_stars(rr,density=density,full_integration=full_integration) for rr in r])
            else:
                x1=((r-low)/2.)*self._x_i+(r+low)/2.    #[pc]
                x2=((high-r)/2.)*self._x_i+(high+r)/2.    #[pc]
                for i in range(self._n):
                    s1=self.density(x1[i])  #[M_sun/pc³]
                    s2=self.density(x2[i])  #[M_sun/pc³]
                    sum1[i]=(self._w_i[i]*x1[i]**2.*s1)  #[M_sun/pc]
                    sum2[i]=(self._w_i[i]*x2[i]*s2)     #[M_sun/pc²]
                sum_1=np.sum(sum1)
                sum_2=np.sum(sum2)
                pot_stars=-4.*np.pi*self._G*((r-low)/(2.*r)*sum_1+(high-r)/2.*sum_2)   #[pc²/s²]
                return pot_stars
        else:
            return self.pot(r)  #[pc²/s²]
    
    def _potential_bh(self,r):
        """
        NAME:
            _potential_bh
        PURPOSE:
            calculates potential caused by central black hole
        INPUT:
            r = array of radius coordinates in pc
        OUTPUT:
            potential of black hole in pc²/s²
        HISTORY:
            2015-12-10 - Written (Milanov, MPIA)
        """
        pot_bh=-self._G*self._bhmass/r #[pc²/s²]
        return pot_bh


    def potential(self,r):
        """
        NAME:
            potential
        PURPOSE:
            adds star and black hole potential
        INPUT:
            r = array of radius coordinates in pc 
        OUTPUT:
            total potential in Globular cluster in pc²/s²
        HISTORY:
            2016-01-16 - Written (Milanov, MPIA)
        """
        return self._potential_stars(r)+self._potential_bh(r) #[pc²/s²]
    
    def _r_derivative(self,r):   
        """
        NAME:
            _r_derivative
        PURPOSE:
            calculates derivative of interpolated potential
        INPUT:
            r = array of radius coordinates in pc 
        OUTPUT:
            r-derivative of potential in pc/s²
        HISTORY:
            2016-01-14 - Written (Milanov, MPIA)
        """    
        der=derivative(self._potential_stars,r)
        return der+self._G*self._bhmass/(r**2.) #[pc/s²]

    def force(self,x,y,z):    
        """
        NAME:
            force    
        PURPOSE:
            calculates force from potential    
        INPUT:
            x,y,z = arrays of distances in x, y and z - direction in pc 
        OUTPUT:
            force array in pc/s²; force[0] in x, force[1] in y and force[2] in z direction
        HISTORY:
            2016-01-14 - Written (Milanov, MPIA)
        """
        force=np.zeros(3)
        r=np.sqrt(x**2+y**2+z**2) #[pc]
        drdx=x/r
        drdy=y/r
        drdz=z/r
        force[0]=self._r_derivative(r)*drdx  #[pc/s²]
        force[1]=self._r_derivative(r)*drdy  #[pc/s²]
        force[2]=self._r_derivative(r)*drdz  #[pc/s²]
        return force
        

    def orbit_integration(self,x,y,z,vx,vy,vz,dt=None,t_end=None,method='leapfrog'):     
        """
        NAME:
            orbit_integration
        PURPOSE:
            integrates the orbit of the star over time
        INPUT:
            x,y,z = distance of one star in x, y and z - direction in pc        
            vx,vy,vz = velocities of the star in x, y and z - direction in km/s    
            t in years; has to be a multiple of dt (default: None)
            dt in years (default: None)
            method: (default: 'leapfrog')
                'leapfrog' for leapfrog integration method 
                'rk4' for Ruge Kutta fourth order integration method (not yet implemented)
        OUTPUT:
            xl,yl,zl = arrays of star position at each time step in x, y and z direction in pc
            vxl_kms,vyl_kms,vzl_kms = arrays of velocities at each time step in x, y and z - direction in km/s
            t_yr = array of each time step in years
        HISTORY:
            
        """  
        if dt is None or t_end is None:
            sys.exit("Error in GCorbit.orbit(): Specify dt and t_end.")  
        t_end_sec=un.yr.to(un.s,t_end)  #[s]
        dt_sec=un.yr.to(un.s,dt)        #[s]  

        vx_pcs=(un.km/un.s).to((un.pc/un.s),vx) #[pc/s]
        vy_pcs=(un.km/un.s).to((un.pc/un.s),vy) #[pc/s]
        vz_pcs=(un.km/un.s).to((un.pc/un.s),vz) #[pc/s]
        
        if method == 'leapfrog':
            
            N=int(t_end/dt)
            t=np.linspace(0,t_end_sec,N) #[s]

            xl=np.zeros(N+1)
            yl=np.zeros(N+1)
            zl=np.zeros(N+1)

            x_l=np.sqrt(xl**2+yl**2+zl**2)

            vxl=np.zeros(N+1)
            vyl=np.zeros(N+1)
            vzl=np.zeros(N+1)

            xl[0]=x #[pc]
            yl[0]=y #[pc]
            zl[0]=z #[pc]

            vxl[0]=vx_pcs   #[pc/s]
            vyl[0]=vy_pcs   #[pc/s]
            vzl[0]=vz_pcs   #[pc/s]

            for i in range(N):
                a=self.force(xl[i],yl[i],zl[i])     #[pc/s²]
        
                xl[i+1]=xl[i]+vxl[i]*dt_sec+1./2.*a[0]*dt_sec**2    #[pc]
                yl[i+1]=yl[i]+vyl[i]*dt_sec+1./2.*a[1]*dt_sec**2    #[pc]
                zl[i+1]=zl[i]+vzl[i]*dt_sec+1./2.*a[2]*dt_sec**2    #[pc]
        
                a_1=self.force(xl[i+1],yl[i+1],zl[i+1])     #[pc/s²]
        
                vxl[i+1]=vxl[i]+1./2.*(a[0]+a_1[0])*dt_sec  #[pc/s]
                vyl[i+1]=vyl[i]+1./2.*(a[1]+a_1[1])*dt_sec  #[pc/s]
                vzl[i+1]=vzl[i]+1./2.*(a[2]+a_1[2])*dt_sec  #[pc/s]
        
            vxl_kms=(un.pc/un.s).to(un.km/un.s,vxl) #[km/s]
            vyl_kms=(un.pc/un.s).to(un.km/un.s,vxl) #[km/s]
            vzl_kms=(un.pc/un.s).to(un.km/un.s,vxl) #[km/s]
            t_yr=un.s.to(un.yr,t)   #[years]
            return xl,yl,zl,vxl_kms,vyl_kms,vzl_kms,t_yr 
        
        elif method == 'rk4':
            return None


    def angularmom(self,x,y,z,vx,vy,vz):
        """
        NAME:
            angularmom
        PURPOSE:
            calculates angular momentum 
        INPUT.
            x,y,z = arrays of distances in x, y and z - direction in pc        
            vx,vy,vz = arrays of velocities in x, y and z - direction in km/s  
        OUTPUT:
            L = total angular momentum in pc²/s
            Lx,Ly,Lz = angular momentum in x, y and z direction in pc²/s
        HISTORY:
            2016-01-14 - Written (Milanov, MPIA)
        """
        vx_pcs=(un.km/un.s).to((un.pc/un.s),vx) #[pc/s]
        vy_pcs=(un.km/un.s).to((un.pc/un.s),vy) #[pc/s]
        vz_pcs=(un.km/un.s).to((un.pc/un.s),vz) #[pc/s]
        
        Lx=y*vz_pcs-z*vy_pcs            #[pc²/s]
        Ly=z*vx_pcs-x*vz_pcs            #[pc²/s]
        Lz=x*vy_pcs-y*vx_pcs            #[pc²/s]
        L=np.sqrt(Lx**2+Ly**2+Lz**2)    #[pc²/s] 
        return L,Lx,Ly,Lz


    def energy(self,x,y,z,vx,vy,vz):
        """
        NAME:
            energy
        PURPOSE:
            calculates energy of star at its actual position
        INPUT:
            x,y,z = arrays of distances in x, y and z - direction in pc        
            vx,vy,vz = arrays of velocities in x, y and z - direction in km/s  
        OUTPUT:
            energy at star position in pc²/s²
        HISTORY:
            2016-01-14 - Written (Milanov, MPIA)
        """  
        vx_pcs=(un.km/un.s).to((un.pc/un.s),vx) #[pc/s]
        vy_pcs=(un.km/un.s).to((un.pc/un.s),vy) #[pc/s]
        vz_pcs=(un.km/un.s).to((un.pc/un.s),vz) #[pc/s]
          
        r=np.sqrt(x**2+y**2+z**2)               #[pc]
        pot=self.potential(r)   #[pc²/s²]
        E=vx_pcs**2./2.+vy_pcs**2./2.+vz_pcs**2./2.+pot     #[pc²/s²]
        return E
    def effective_potential(self,r,L):
        func=self.potential(r)+L**2/(2*r**2)
        return func
        
    def _periapocenter_aux(self,r,E,L):
        """
        NAME:
            _periapocenter_aux
        PURPOSE:
            gives auxiliary function to solve in function periapocenter
        INPUT:
            r = array of radius coordinates in pc
            E = energy at star position in pc²/s²
            L = angular momentum in pc²/s
        OUTPUT:
            auxiliary function for periapocenter
        HISTORY:
            2016-01-19 - Written (Milanov, MPIA)
        """ 
        func=(1./r)**2.+2.*(self.potential(r)-E)/L**2. #[1/pc²]  
        return func

    def periapocenter(self,r,x,y,z,vx,vy,vz):
        """
        NAME:
            periapocenter
        PURPOSE:
            calculates pericenter and apocenter of orbit
        INPUT:
            x,y,z = arrays of distances in x, y and z - direction in pc        
            vx,vy,vz = arrays of velocities in x, y and z - direction in km/s  
        OUTPUT:
            rmin = pericenter of orbit in pc
            rmax = apocenter of orbit in pc
        HISTORY:
            2016-01-16 - Written (Milanov, MPIA)
        """
        r_sqrt=np.sqrt(x**2.+y**2.+z**2.)    #[pc]

        E=self.energy(x,y,z,vx,vy,vz)   #[pc²/s²]
        L=self.angularmom(x,y,z,vx,vy,vz)[0]            #[pc²/s]

        if np.sign(self._periapocenter_aux(r,E,L)) == np.sign(self._periapocenter_aux(1e-7,E,L)): #if star is in peri- or apocenter but can't be calculated due to rounding errors
            if np.sign(self._periapocenter_aux(r_sqrt,E,L)) != np.sign(self._periapocenter_aux(1e-7,E,L)): 
                r_mi=r_sqrt
                rmin=opt.brentq(self._periapocenter_aux,1e-7,r_mi,args=(E,L)) #[pc] 
            elif np.sign(self._periapocenter_aux(r*1.000001,E,L)) != np.sign(self._periapocenter_aux(1e-7,E,L)): 
                r_mi=r*1.000001
                rmin=opt.brentq(self._periapocenter_aux,1e-7,r_mi,args=(E,L)) #[pc] 
            elif np.sign(self._periapocenter_aux(r*0.99999,E,L)) != np.sign(self._periapocenter_aux(1e-7,E,L)):
                r_mi=r*0.99999
                rmin=opt.brentq(self._periapocenter_aux,1e-7,r_mi,args=(E,L)) #[pc] 
            elif np.sign(self._periapocenter_aux(r*1.00001,E,L)) != np.sign(self._periapocenter_aux(1e-7,E,L)): 
                r_mi=r*1.00001
                rmin=opt.brentq(self._periapocenter_aux,1e-7,r_mi,args=(E,L)) #[pc] 
            elif np.sign(self._periapocenter_aux(r*0.9999,E,L)) != np.sign(self._periapocenter_aux(1e-7,E,L)):
                r_mi=r*0.9999
                rmin=opt.brentq(self._periapocenter_aux,1e-7,r_mi,args=(E,L)) #[pc] 
            elif np.sign(self._periapocenter_aux(r*1.0001,E,L)) != np.sign(self._periapocenter_aux(1e-7,E,L)): 
                r_mi=r*1.0001
                rmin=opt.brentq(self._periapocenter_aux,1e-7,r_mi,args=(E,L)) #[pc] 
            elif np.sign(self._periapocenter_aux(r*0.999,E,L)) != np.sign(self._periapocenter_aux(1e-7,E,L)):
                r_mi=r*0.999
                rmin=opt.brentq(self._periapocenter_aux,1e-7,r_mi,args=(E,L)) #[pc] 
        else:
            r_mi=r
            rmin=opt.brentq(self._periapocenter_aux,1e-7,r_mi,args=(E,L)) #[pc] 

        if np.sign(self._periapocenter_aux(r,E,L)) == np.sign(self._periapocenter_aux(np.max(self._r_bin),E,L)):
            if np.sign(self._periapocenter_aux(r_sqrt,E,L)) != np.sign(self._periapocenter_aux(np.max(self._r_bin),E,L)): 
                r_ma=r_sqrt
                rmax=opt.brentq(self._periapocenter_aux,r_ma,np.max(self._r_bin),args=(E,L)) #[pc] 
            elif np.sign(self._periapocenter_aux(r*1.000001,E,L)) != np.sign(self._periapocenter_aux(np.max(self._r_bin),E,L)):
                r_ma=r*1.000001
                rmax=opt.brentq(self._periapocenter_aux,r_ma,np.max(self._r_bin),args=(E,L)) #[pc] 
            elif np.sign(self._periapocenter_aux(r*0.99999,E,L)) != np.sign(self._periapocenter_aux(np.max(self._r_bin),E,L)):
                r_ma=r*0.99999
                rmax=opt.brentq(self._periapocenter_aux,r_ma,np.max(self._r_bin),args=(E,L)) #[pc] 
            elif np.sign(self._periapocenter_aux(r*1.00001,E,L)) != np.sign(self._periapocenter_aux(np.max(self._r_bin),E,L)):
                r_ma=r*1.00001
                rmax=opt.brentq(self._periapocenter_aux,r_ma,np.max(self._r_bin),args=(E,L)) #[pc] 
            elif np.sign(self._periapocenter_aux(r*0.9999,E,L)) != np.sign(self._periapocenter_aux(np.max(self._r_bin),E,L)):
                r_ma=r*0.9999
                rmax=opt.brentq(self._periapocenter_aux,r_ma,np.max(self._r_bin),args=(E,L)) #[pc] 
            elif np.sign(self._periapocenter_aux(r*1.0001,E,L)) != np.sign(self._periapocenter_aux(np.max(self._r_bin),E,L)):
                r_ma=r*1.0001
                rmax=opt.brentq(self._periapocenter_aux,r_ma,np.max(self._r_bin),args=(E,L)) #[pc] 
            elif np.sign(self._periapocenter_aux(r*0.999,E,L)) != np.sign(self._periapocenter_aux(np.max(self._r_bin),E,L)):
                r_ma=r*0.999
                rmax=opt.brentq(self._periapocenter_aux,r_ma,np.max(self._r_bin),args=(E,L)) #[pc] 
            else:
                r_ma=r
                rmax=opt.brentq(self._periapocenter_aux,r_ma,np.max(self._r_bin)*2.,args=(E,L)) #[pc] 
        else:
            r_ma=r
            rmax=opt.brentq(self._periapocenter_aux,r_ma,np.max(self._r_bin),args=(E,L)) #[pc] 
        #rmin=opt.brentq(self._periapocenter_aux,1e-7,r_mi,args=(E,L)) #[pc] 
        #rmax=opt.brentq(self._periapocenter_aux,r_ma,np.max(self._r_bin),args=(E,L)) #[pc] 

        #if rmin == rmax:
         #   if rmin <= 1.01* r and rmin >= 0.99*r and rmax <= 1.01* r and rmax >= 0.99*r: #checks if orbit is circular
          #      return rmin,rmax 
           # else:
            #    print(rmin,rmax,r)
             #   sys.exit('Error in GCorbit.periapocenter(): rmin=rmax; to do: implement con')
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
            calculates integrand needed for J_r action
        INPUT:
            r = array of radius coordinates in pc
            E = energy at star position in pc²/s²
            L = angular momentum in pc²/s
        OUTPUT:
            integrand to integrate in J_r in pc²/s²
        HISTORY:
            2015-12-04 - Written (Milanov, MPIA)
        """
        pot=self.potential(r)   #[pc²/s²]
        return np.sqrt(2.*E-2.*pot-L**2./r**2.) #[pc²/s²]


    def _J_phi(self,x,y,z,vx,vy,vz):
        """
        NAME:
            _J_phi
        PURPOSE:    
            calculates action J_phi
        INPUT:
            x,y,z = arrays of distances in x, y and z - direction in pc        
            vx,vy,vz = arrays of velocities in x, y and z - direction in km/s  
        OUTPUT:
            J_phi in pc*km/s
        HISTORY:
            2015-11-26 - Written (Milanov, MPIA)
        """
        Lz=self.angularmom(x,y,z,vx,vy,vz)[3]   
        J_phi=Lz    #[pc²/s]
        J_phi_pckms=(un.pc**2/un.s).to(un.pc*un.km/un.s,J_phi) #[pc*km/s]
        return J_phi_pckms
    
    def _J_theta(self,x,y,z,vx,vy,vz):
        """
        NAME:
            _J_theta
        PURPOSE:
            calculates action J_theta
        INPUT:
            x,y,z = arrays of distances in x, y and z - direction in pc        
            vx,vy,vz = arrays of velocities in x, y and z - direction in km/s              
        OUTPUT:
            J_theta in pc²/s
        HISTORY:
            2015-11-26 - Written (Milanov, MPIA)
        """
        L=self.angularmom(x,y,z,vx,vy,vz)[0]
        Lz=self.angularmom(x,y,z,vx,vy,vz)[3]
        J_theta=L-np.abs(Lz)    #[pc²/s]
        J_theta_pckms=(un.pc**2/un.s).to(un.pc*un.km/un.s,J_theta) #[pc*km/s]
        return J_theta_pckms

### J_r beim Integral unsicher wegen Argumenten f�r j_rint und wegen Apo- und Pericenter ###

    def _J_r(self,r,x,y,z,vx,vy,vz):
        """
        NAME:
            _J_r
        PURPOSE:
            calculates J_r action
        INPUT:
            x,y,z = arrays of distances in x, y and z - direction in pc        
            vx,vy,vz = arrays of velocities in x, y and z - direction in km/s
        OUTPUT:
            J_r in pc²/s
        HISTORY:
            2015-12-04 - Written (Milanov, MPIA)
        """
        rmin,rmax=self.periapocenter(r,x,y,z,vx,vy,vz)        
        E=self.energy(x,y,z,vx,vy,vz)   #[pc²/s²]
        L=self.angularmom(x,y,z,vx,vy,vz)[0]    #[pc²/s]
        J_r=1/np.pi*intg.quad(self._j_rint,rmin,rmax,args=(E,L))[0] #[pc²/s]
        J_r_pckms=(un.pc**2/un.s).to(un.pc*un.km/un.s,J_r) #[pc*km/s]
        return J_r_pckms,rmin,rmax
    
    def actions(self,r,x,y,z,vx,vy,vz):
        """
        NAME:
            actions
        PURPOSE:
            returns actions
        INPUT:
            x,y,z = arrays of distances in x, y and z - direction in pc        
            vx,vy,vz = arrays of velocities in x, y and z - direction in km/s
        OUTPUT:
            J_phi, J_theta, J_r in pc²/s
        HISTORY:
            2016-01-14 - Written (Milanov, MPIA)
        """
        J_phi=self._J_phi(x,y,z,vx,vy,vz)               #[pc*km/s]
        J_theta=self._J_theta(x,y,z,vx,vy,vz)           #[pc*km/s]
        J_r=self._J_r(r,x,y,z,vx,vy,vz)                 #[pc*km/s]
        actions=J_phi,J_theta,J_r
        return actions

    def v_circ(self,r):
        """
        NAME:
            v_circ
        PURPOSE:
            returns circular velocity
        INPUT:
            r - radius at which circular velocity should be calculated
        OUTPUT:
            v_circ
        HISTORY:
            2016-02-05 - Written (Milanov, MPIA)
        """
        v_circ=np.sqrt(r*np.abs(self._r_derivative(r))) #[pc/s]
        return v_circ

    def _r_guide_aux(self,r,L):
        """
        NAME:
            _r_guide_aux
        PURPOSE:
            returns function to solve in r_guide
        INPUT:
            r - radius at which function is evaluated
        OUTPUT:
            function
        HISTORY:
            2016-02-05 - Written (Milanov, MPIA)
        """
        return r*self.v_circ(r)-np.sqrt(L**2)
        
    def r_guide_min(self,r,x,y,z,vx,vy,vz):
        """
        NAME:
            r_guide
        PURPOSE:
            returns guiding-star radius 
        INPUT:
            x,y,z = arrays of distances in x, y and z - direction in pc        
            vx,vy,vz = arrays of velocities in x, y and z - direction in km/s
        OUTPUT:
            aJ_phi, J_theta, J_r in pc²/s
        HISTORY:
            2016-02-05 - Written (Milanov, MPIA)
        """
        L=self.angularmom(x,y,z,vx,vy,vz)[0] 
        E=self.energy(x,y,z,vx,vy,vz)

        bnds=((self.periapocenter(r,x,y,z,vx,vy,vz)[0],self.periapocenter(r,x,y,z,vx,vy,vz)[1]),)
        r_guide=opt.minimize(self._periapocenter_aux,x0=r,args=(E,L),bounds=bnds)
        r_guide_x=r_guide.x
        return r_guide_x


    def r_guide_root(self,r,x,y,z,vx,vy,vz):
        L=self.angularmom(x,y,z,vx,vy,vz)[0] 
        a=np.min(self._r_bin)
        b=np.max(self._r_bin)
        rg_r=opt.brentq(self._r_guide_aux,a=a,b=b,args=(L))
        return rg_r
