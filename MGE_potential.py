import numpy as np
from astropy import units as un
from scipy import constants as cs
from scipy import integrate as intg
from scipy import optimize as opt
from scipy.misc import derivative
import sys

class MGE_orbit: 
    def __init__(self, counts = None, sigma = None, inputfilename = None, M_BH = None):
        """
        NAME:
            __init__
        PURPOSE:
            initialize a MGE object
        INPUT:
            counts: sol[0, : ] of MGE of density profile
            sigma: sol[1, : ] of MGE of density profile
            inputfilename: file containing mge.sol results
            M_BH: Mass of central IMBH (if one is there)
        OUTPUT:
            instance
        HISTORY:
            2016-08-22 - Written (Milanov, MPIA)
            2016-08-23 - M_BH added (Milanov, MPIA)
            2016-11-09 - Sources to check units added (Milanov, MPIA)
        """
        if inputfilename is not None:
            data = np.loadtxt(inputfilename)        
            self._counts = data[0,:]     #[M_sun / pc^2]
            self._sigma = data[1,:]     #[pc]
            # For [counts] see Cappelari mge_fit_1d documentation in "Example"
            # For [sigma] see Cappellari mge_fit_1d documentation in "Optional Output Keywords" part 2
        elif counts is None or sigma is None:
            sys.exit("Error in MGE_potential.__init__(): Specify input file or counts and sigma.")
        else:
            self._counts = counts       #[M_sun / pc^2]
            self._sigma = sigma     #[pc]

        self._G = (un.m ** 3 / (un.kg * un.s ** 2)).to(un.pc ** 3/(un.solMass * un.s ** 2), cs.G) #[pc^3 / M_sun * s^2]
        self._mass = self._counts * 2. * cs.pi * self._sigma ** 2 #[M_sun]
        # For [mass] and the formula see ... ASK WILMA
        self._bhmass = M_BH #[M_sun]
        self._counts3d = self._counts / (np.sqrt(2. * cs.pi) * self._sigma) #[M_sun / pc^3] 
        # For [counts3d] at the formula see Cappellari mge_fit_1d documentation in "Optional Output Keywords" part 1
        return None

    def _H_j(self, u, R):
        """
        NAME:
            _H_j
        PURPOSE:
            auxiliary function to calculate potential (see Cappellari 2008, (17))
        INPUT:
            u: integration parameter
            R: distance from centre (only input in potential function)
            sigma: sigma given from MGE 
        OUTPUT:
            one factor of integrand
        HISTORY:
            2016-08-22 - Written (Milanov, MPIA)
        """
        sigma = self._sigma #[pc]
        H = np.exp(- u ** 2 * R ** 2 / (2. * sigma ** 2)) #[]
        return H

    def _integrand (self, u, R):
        """
        NAME:
            _integrand
        PURPOSE:
            gives integrand for potential
        INPUT:
            u: integration parameter
            R: distance from centre (only input in potential function)
            sigma: sigma given from MGE
            M: mass calculated from counts given from MGE
        OUTPUT:
            integrand
        HISTORY
            2016-08-22 - Written (Milanov, MPIA)
        """
        sigma = self._sigma #[pc]
        M = self._mass #[M_sun]
        H = self._H_j(u, R) #[]
        return np.sum(M * H / sigma) #[M_sun/pc]

    def _star_potential (self, R):
        """
        NAME:
            _star_potential
        PURPOSE:
            calculates potential of stars at distance R
        INPUT:
            R: distance from centre
        OUTPUT:
            potential of stars in GC
        HISTORY
            2016-08-22 - Written (Milanov, MPIA)
            2016-08-23 - Changed Potential to _star_potential to add in a later function the IMBH potential (Milanov, MPIA)
        """
        a = - np.sqrt( 2. / cs.pi) * self._G    #[pc^3/M_sun*s^2]
        b = intg.quad(self._integrand, 0., 1., args = (R))[0] #[M_sun/pc]
        return a * b #[pc^2/s^2]

    
    def _bh_potential(self, R):
        """
        NAME:
            _bh_potential
        PURPOSE:
            calculates potential of IMBH at distance R_IMBH
        INPUT:
            R: distance from centre
        OUTPUT:
            potential of IMBH in GC
        HISTORY
            2016-08-23 - Written (Milanov, MPIA)
        """
        M_BH = self._bhmass #[M_sun]
        if M_BH == None:
            pot_bh = 0
        else:
            pot_bh = - self._G * M_BH / R #[pc^2/s^2]
        return pot_bh

    def potential(self, R):
        """
        NAME:
            potential
        PURPOSE:
            calculates total potential at distance R
        INPUT:
            R: distance from centre
        OUTPUT:
            potential of IMBH and stars in GC
        HISTORY
            2016-08-23 - Written (Milanov, MPIA)
        """
        pot = self._bh_potential(R) + self._star_potential(R) #[pc^2/s^2]
        return pot

    def _r_derivative(self,r):   
        """
        NAME:
            _r_derivative
        PURPOSE:
            calculates derivative of potential
        INPUT:
            r = array of radius coordinates in pc 
        OUTPUT:
            r-derivative of potential in pc/s^2
        HISTORY:
            2016-02-24 - Written (Milanov, MPIA)
        """    
        der = derivative(self._potential_stars,r)
        return der + self._G * self._bhmass / (r ** 2.) #[pc/s^2]

    def force(self, x, y, z):    
        """
        NAME:
            force    
        PURPOSE:
            calculates force from potential    
        INPUT:
            x, y, z = arrays of distances in x, y and z - direction in pc 
        OUTPUT:
            force array in pc/s^2; force[0] in x, force[1] in y and force[2] in z direction
        HISTORY:
            2016-01-14 - Written (Milanov, MPIA)
        """
        force = np.zeros(3)
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2) #[pc]
        drdx = x / r
        drdy = y / r
        drdz = z / r
        force[0] = self._r_derivative(r) * drdx  #[pc/s^2]
        force[1] = self._r_derivative(r) * drdy  #[pc/s^2]
        force[2] = self._r_derivative(r) * drdz  #[pc/s^2]
        return force
        

    def angularmom(self, x, y, z, vx, vy, vz):
        """
        NAME:
            angularmom
        PURPOSE:
            calculates angular momentum 
        INPUT.
            x,y,z = arrays of distances in x, y and z - direction in pc        
            vx,vy,vz = arrays of velocities in x, y and z - direction in km/s  
        OUTPUT:
            L = total angular momentum in pc^2/s
            Lx,Ly,Lz = angular momentum in x, y and z direction in pc^2/s
        HISTORY:
            2016-08-24 - Written (Milanov, MPIA)
        """
        vx_pcs = (un.km / un.s).to((un.pc / un.s), vx) #[pc/s]
        vy_pcs = (un.km / un.s).to((un.pc / un.s), vy) #[pc/s]
        vz_pcs = (un.km / un.s).to((un.pc / un.s), vz) #[pc/s]
        
        Lx=y*vz_pcs-z*vy_pcs            #[pc^2/s]
        Ly=z*vx_pcs-x*vz_pcs            #[pc^2/s]
        Lz=x*vy_pcs-y*vx_pcs            #[pc^2/s]
        L=np.sqrt(Lx**2+Ly**2+Lz**2)    #[pc^2/s] 
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
            energy at star position in pc^2/s^2
        HISTORY:
            2016-01-14 - Written (Milanov, MPIA)
        """  
        vx_pcs=(un.km/un.s).to((un.pc/un.s),vx) #[pc/s]
        vy_pcs=(un.km/un.s).to((un.pc/un.s),vy) #[pc/s]
        vz_pcs=(un.km/un.s).to((un.pc/un.s),vz) #[pc/s]
          
        r=np.sqrt(x**2+y**2+z**2)               #[pc]
        pot=self.potential(r)   #[pc^2/s^2]
        E=vx_pcs**2./2.+vy_pcs**2./2.+vz_pcs**2./2.+pot     #[pc^2/s^2]
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
            E = energy at star position in pc^2/s^2
            L = angular momentum in pc^2/s
        OUTPUT:
            auxiliary function for periapocenter
        HISTORY:
            2016-01-19 - Written (Milanov, MPIA)
        """ 
        func=(1./r)**2.+2.*(self.potential(r)-E)/L**2. #[1/pc^2]  
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

        E=self.energy(x,y,z,vx,vy,vz)   #[pc^2/s^2]
        L=self.angularmom(x,y,z,vx,vy,vz)[0]            #[pc^2/s]

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

        if np.sign(self._periapocenter_aux(r,E,L)) == np.sign(self._periapocenter_aux(np.max(self._sigma),E,L)):
            if np.sign(self._periapocenter_aux(r_sqrt,E,L)) != np.sign(self._periapocenter_aux(np.max(self._sigma),E,L)): 
                r_ma=r_sqrt
                rmax=opt.brentq(self._periapocenter_aux,r_ma,np.max(self._sigma),args=(E,L)) #[pc] 
            elif np.sign(self._periapocenter_aux(r*1.000001,E,L)) != np.sign(self._periapocenter_aux(np.max(self._sigma),E,L)):
                r_ma=r*1.000001
                rmax=opt.brentq(self._periapocenter_aux,r_ma,np.max(self._sigma),args=(E,L)) #[pc] 
            elif np.sign(self._periapocenter_aux(r*0.99999,E,L)) != np.sign(self._periapocenter_aux(np.max(self._sigma),E,L)):
                r_ma=r*0.99999
                rmax=opt.brentq(self._periapocenter_aux,r_ma,np.max(self._sigma),args=(E,L)) #[pc] 
            elif np.sign(self._periapocenter_aux(r*1.00001,E,L)) != np.sign(self._periapocenter_aux(np.max(self._sigma),E,L)):
                r_ma=r*1.00001
                rmax=opt.brentq(self._periapocenter_aux,r_ma,np.max(self._sigma),args=(E,L)) #[pc] 
            elif np.sign(self._periapocenter_aux(r*0.9999,E,L)) != np.sign(self._periapocenter_aux(np.max(self._sigma),E,L)):
                r_ma=r*0.9999
                rmax=opt.brentq(self._periapocenter_aux,r_ma,np.max(self._sigma),args=(E,L)) #[pc] 
            elif np.sign(self._periapocenter_aux(r*1.0001,E,L)) != np.sign(self._periapocenter_aux(np.max(self._sigma),E,L)):
                r_ma=r*1.0001
                rmax=opt.brentq(self._periapocenter_aux,r_ma,np.max(self._sigma),args=(E,L)) #[pc] 
            elif np.sign(self._periapocenter_aux(r*0.999,E,L)) != np.sign(self._periapocenter_aux(np.max(self._sigma),E,L)):
                r_ma=r*0.999
                rmax=opt.brentq(self._periapocenter_aux,r_ma,np.max(self._sigma),args=(E,L)) #[pc] 
            else:
                r_ma=r
                rmax=opt.brentq(self._periapocenter_aux,r_ma,np.max(self._sigma)*2.,args=(E,L)) #[pc] 
        else:
            r_ma=r
            rmax=opt.brentq(self._periapocenter_aux,r_ma,np.max(self._sigma),args=(E,L)) #[pc] 
        #rmin=opt.brentq(self._periapocenter_aux,1e-7,r_mi,args=(E,L)) #[pc] 
        #rmax=opt.brentq(self._periapocenter_aux,r_ma,np.max(self._sigma),args=(E,L)) #[pc] 

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
            E = energy at star position in pc^2/s^2
            L = angular momentum in pc^2/s
        OUTPUT:
            integrand to integrate in J_r in pc^2/s^2
        HISTORY:
            2015-12-04 - Written (Milanov, MPIA)
        """
        pot=self.potential(r)   #[pc^2/s^2]
        return np.sqrt(2.*E-2.*pot-L**2./r**2.) #[pc^2/s^2]


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
        J_phi=Lz    #[pc^2/s]
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
            J_theta in pc^2/s
        HISTORY:
            2015-11-26 - Written (Milanov, MPIA)
        """
        L=self.angularmom(x,y,z,vx,vy,vz)[0]
        Lz=self.angularmom(x,y,z,vx,vy,vz)[3]
        J_theta=L-np.abs(Lz)    #[pc^2/s]
        J_theta_pckms=(un.pc**2/un.s).to(un.pc*un.km/un.s,J_theta) #[pc*km/s]
        return J_theta_pckms

### J_r beim Integral unsicher wegen Argumenten fuer j_rint und wegen Apo- und Pericenter ###

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
            J_r in pc^2/s
        HISTORY:
            2015-12-04 - Written (Milanov, MPIA)
        """
        rmin,rmax=self.periapocenter(r,x,y,z,vx,vy,vz)        
        E=self.energy(x,y,z,vx,vy,vz)   #[pc^2/s^2]
        L=self.angularmom(x,y,z,vx,vy,vz)[0]    #[pc^2/s]
        J_r=1/np.pi*intg.quad(self._j_rint,rmin,rmax,args=(E,L))[0] #[pc^2/s]
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
            J_phi, J_theta, J_r in pc^2/s
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
            aJ_phi, J_theta, J_r in pc^2/s
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
        a=np.min(self._sigma)
        b=np.max(self._sigma)
        rg_r=opt.brentq(self._r_guide_aux,a=a,b=b,args=(L))
        return rg_r
