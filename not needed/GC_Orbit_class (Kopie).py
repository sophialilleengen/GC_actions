import numpy as np
from scipy import interpolate
from scipy import integrate as intg
from scipy import optimize as opt




class GCorbit:
	def __init__(self,R,rho,bhmass): #filename of object containing interpolation of density profile
        """
        NAME:
           __init__
        PURPOSE:
           initialize a GCorbit object - a Globular cluster object which calculates actions and integrates orbits.
        INPUT:
           R= array of radius coordinates in pc (default: None)
        OUTPUT:
           (none)
        HISTORY:
           2016-01-15 - Written (Milanov, MPIA)
        """
        #^-- 1. I would recommend adding such a header to each function. You can keep track of your changes, make it more usable for others and I believe that this format can be made into an automatic documentation as well...
        #2. It's fine to not use a filename in the initialization. But I believe that you want to have a file with R vs. rho in it anyway (also easier for plotting). So you could for example define alternative input via keywords, something like (not exactly like):
        #   def ___init(self,r_pc=None,rho_Msunpc3=None,bhmass_Msun=None,inputfilename=None):
        #       if inputfilename is not None:
        #           out = numpy.loadtxt(inputfilename)
        #           self._r  =out[0,:]  #[pc]
        #           self._rho=out[1,:]  #[M_sun/pc^3]
        #       elif r is None and rho is None:
        #           sys.exit("Error in GCorbit.__init__(): Specify input file or r and rho.")
        #       else:
        #           proceed with your code...
        #3. I would always write the units that you are using, to make sure, that there are no unit conversion problems due to a wrong input.
		self.r=R
		self.rho=rho
		self.s=interpolate.InterpolatedUnivariateSpline(np.log(self.r[:]),np.log(self.rho[:]))
		pot_y=self.potential(r,density,low=0.003,high=104.64,x_i=None,w_i=None,full_integration=True)   #11. What units??? Write it in code as comment.
        #^-- 3. The limits that you use here are just for your current GC. To make it more flexible, use min(r) and max(r).
        #    4. Maybe you want to calculate the xi and wi already in __init__ as well and pass them to the potential function...
		self.pot=interpolate.InterpolatedUnivariateSpline(self.r,pot_y)
        #^-- 5. Did you test if it makes sense to interpolate the potential in log-space as well?

    #35. I think you need a function def density(self,r) that calls the interpolation object self.s with np.log(r) and returns the result as np.exp(self.s(np.log(r))), or something like that

	def potential(self,r,density,low=0.003,high=104.64,x_i=None,w_i=None,full_integration=False):   
        #^-- 6. I would call this _potential_stars, as you also have a _potential_bh function.
        #    15. density is only needed in case of full_integration. Make it a keyword as well, density=None, and test inside the if condition if it is really set.
		if True:    #7. Why "if True:"??? Do you mean "if full_integration is True:"?
				sum1=np.zeros(n)    #9. What is n???
   				sum2=np.zeros(n)
				if isinstance(r,np.ndarray):
        				return np.array([potential(rr,density,low=low,high=high,x_i=x_i,w_i=w_i) for rr in r])
    			else:
        				if r<low or r>high:
            					sys.exit("r is smaller or bigger than star boundaries") #13. I would put this error message outside of the "if full_integration" bracket, at the beginning of the function.
        				x1=((r-low)/2)*x_i+(r+low)/2    #8. Better write "2." as float. Maybe it works anyway, but it avoids mistakes if integer input. Also everywhere else, where you actually need a float and not an integer.
        				x2=((high-r)/2)*x_i+(high+r)/2
        				for i in range(n):
            					s1=density(x1[i])   #36. This should be self.density(...)
            					s2=density(x2[i])
            					sum1[i]=(w_i[i]*x1[i]**2*s1)
            					sum2[i]=(w_i[i]*x2[i]*s2)
        				sum_1=np.sum(sum1)
        				sum_2=np.sum(sum2)
        				return -4*np.pi*G*((r-low)/(2*r)*sum_1+(high-r)/2*sum_2)    #10. What is G? What units? You could specify it as self.G in __init__. Make sure it has the correct units.
		else:
			return self.pot(r,density,low=0.003,high=104.64,x_i=None,w_i=None)
            #^-- 12. I don't think the interpolation object pot does have low, high etc. as input.
    
	def potential_bh(self,r,bhmass):
		return -G*bhmass/r
        #14. I would set self._bhmass and self._G in the __init__function, where you also make sure that the units are correct. Then potential_bh only needs r as input.

#addiere sternen- und bh-potential vor interpolation?
#^-- 16. Nein. Du koenntest noch eine function potential(self,r) machen, die nix anderes tut, als _potential_stars(r)+_potential_bh(r) zu addieren. Allerdings ist das dann die, die du immer von aussen aufrufen moechtest. 

	def r_derivative(self,potential):   #17. Sorry, no idea, what you're doing in this function.
		return self.potential.derivative
        #20. I think having this function is good, but it should do something else. It could have a keyword explicit_derivative=False. In the __init__ function you could call this function once with explicit_derivative=True and calculate the derivative of the potential explicitly at a few points. Everywhere else you use the interpolation analogous to the potential.
        #21. This function should have a parameter r at which you want to know the potential derivative.

	def force(self,x,y,z,potential):    #18. I don't think, you have to pass potential to the force function. You can use inside force() the function self.potential(r).
		force=np.array(3)
		r=np.sqrt(x**2+y**2+z**2)
		drdx=x/r
		drdy=y/r
		drdz=z/r
		force[0]=self.r_derivative(potential)*drdx  #19. Not sure, if this works...
		force[1]=self.r_derivative(potential)*drdy
		force[2]=self.r_derivative(potential)*drdz
		return force
		

	def orbit(self,x0,y0,z0,vx0,vy0,vz0,N): #noch von force abh�ngig machen?    #22. Is this the orbit integration function? I would, in addition to the initial star position, have delta_t and t_end (in useful units) as parameters. N is then calculated in the function from that.
		xl=np.zeros(N+1)
		yl=np.zeros(N+1)
		zl=np.zeros(N+1)

		x_l=np.sqrt(xl**2+yl**2+zl**2)

		vxl=np.zeros(N+1)
		vyl=np.zeros(N+1)
		vzl=np.zeros(N+1)

		xl[0]=x0
		yl[0]=y0
		zl[0]=z0

		vxl[0]=vx0
		vyl[0]=vy0
		vzl[0]=vz0

		for i in range(N):
		    	xl[0]=x0
    			yl[0]=y0
			zl[0]=z0
    
    			a=self.force(xl[i],yl[i],zl[i]) #hier auch noch potential rein?
    
    			xl[i+1]=xl[i]+vxl[i]*dt+1./2.*a[0]*dt**2
    			yl[i+1]=yl[i]+vyl[i]*dt+1./2.*a[1]*dt**2
    			zl[i+1]=zl[i]+vzl[i]*dt+1./2.*a[2]*dt**2
    
    			a_1=self.force(xl[i+1],yl[i+1],zl[i+1]) #und hier auch potential?
    
    			vxl[i+1]=vxl[i]+1./2.*(a[0]+a_1[0])*dt
    			vyl[i+1]=vyl[i]+1./2.*(a[1]+a_1[1])*dt
   			vzl[i+1]=vzl[i]+1./2.*(a[2]+a_1[2])*dt
		
		return xl,yl,zl,vxl,vyl,vzl #23. You could also return the time array. Easier for plotting...

    #^-- 33. Hab mir die Funktion nicht im Detail angeschaut. Gehe davon aus, dass es stimmt.
    #^-- 34. Da du Runge-Kutta und leapfrog implementiert hast, koenntest du ein keyword method='leapfrog' setzen, dass man auch zu 'rk4' setzen kann, wenn man moechte. Dann kannst du einfach hin und her switchen.


##### Versuch 1 zu actions #####

	def angularmom(self,x,y,z,vx,vy,vz):
    		Lx=y*vz-z*vy
    		Ly=z*vx-x*vz
    		Lz=x*vy-y*vx
    		L=np.sqrt(Lx**2+Ly**2+Lz**2)
    		return L,Lx,Ly,Lz


	def energy(self,x,y,z,vx,vy,vz):
    		pot=self.potential(x,y,z)		#eventuell weitere Parameter im Potential #24. Noe, ich glaub nicht...
    		E=vx**2./2.+vy**2./2.+vz**2./2.+pot
    		return E

	def periapocenter(self,r_ap,x,y,z,vx,vy,vz):
	    	r=np.sqrt(x**2+y**2+z**2)
    		pot=self.potential(x,y,z)		#eventuell weitere Parameter im Potential
		    E=self.energy(x,y,z,vx,vy,vz)
		    L=elf.angularmom(x,y,z,vx,vy,vz)[0] #25. Tippfehler in self
    		return (1/r_ap)**2.+2.*(self.potential(r_ap)-E)/L**2. 	#Potential im Apo- bzw Pericenter noch richtige Argumente/Parameter einsetzen
            #^-- 28. Diese Funktion sollte meiner Meinung nach rmin und rmax returnen.
	

	rmin=opt.fsolve(periapocenter,np.min(r)) #nicht min(rl) sondern einfach kleiner Wert weil ich es erst durch orbit integration weiss
	rmax=opt.fsolve(periapocenter,np.max(r))
    #^-- 26a. Warum steht das ausserhalb einer function? 
    #    26b.Sollte doch in perapocenter drin stehen.
    #27. Ich wuerde noch eine Funktion _periapocenter_aux(E,L) definieren, und die dann in periapocenter in fsolve aufrufen.

	def j_rint(self,x,y,z,vx,vy,vz):
    		r=np.sqrt(x**2+y**2+z**2)
    		pot=self.potential(x,y,z)
    		E=self.energy(x,y,z,vx,vy,vz)
    		L=self.angularmom(x,y,z,vx,vy,vz)[0]
    		return np.sqrt(2.*E-2.*pot-L**2./r**2.)
    #29. j_rint ist doch die funktion die dann in J_r als integrand im Integral aufgerufen wird, oder? Das Integral ist ueber r . Das heisst, du uebergibst dieser Funktion NUR das r und ausserdem die Konstanten E und L, die du schon vorher in J_r ausgerechnet hast.

	def J_phi(self,x,y,z,vx,vy,vz):
    		Lz=self.angularmom(x,y,z,vx,vy,vz)[3]
    		J_phi=Lz
    		return J_phi
    
	def J_theta(self,x,y,z,vx,vy,vz):
    		L=self.angularmom(x,y,z,vx,vy,vz)[0]
    		Lz=self.angularmom(x,y,z,vx,vy,vz)[1]
    		J_theta=L-np.abs(Lz)
    		return J_theta

### J_r beim Integral unsicher wegen Argumenten f�r j_rint und wegen Apo- und Pericenter ###

	def J_r(self,x,y,z,vx,vy,vz):

	### Peri- und Apocenter Suche noch verbessern, nicht min(rl)/max(rl) sondern irgendwie kleine bzw gro�e Werte finden abhaengig von r ###
		rmin=opt.fsolve(periapocenter,np.min(r)) #nicht min(rl) sondern einfach kleiner Wert weil ich es erst durch orbit integration weiss
		rmax=opt.fsolve(periapocenter,np.max(r))
        #^-- 30. So kann man es durchaus auch machen (dann kannst du meine Kommentare 28., 26b. und 27. ignorieren. Allerdings waere es ja auch huebsch, eine Funktion zu haben, die von aussen aufgerufen werden kann und dir fuer einen gegebenen Stern pericenter und apocenter berechnet, unabhaengig von allem anderen. Hier kannst du die Funktion dann aufrufen.

    		J_r=1/np.pi*intg.quad(j_rint,rmin,rmax)[0] 
    		return J_r
	
	def actions(self,x,y,z,vx,vy,vz):
		J_phi=self.J_phi(x,y,z,vx,vy,vz)
		J_theta=self.J_theta(x,y,z,vx,vy,vz)
		J_r=self.J_r(x,y,z,vx,vy,vz)
		return J_phi,J_theta,J_r



##### Versuch 2 zu actions, tendiere zu Versuch 1, das hier ist fast urspr�nglicher Code von mir ###### #32. Ich finde, es hat beides Vor- und Nachteile, ob du nun diese Version oder die oben verwendest. Geschmackssache.

	def actions(self,xl,yl,zl,vxl,vyl,vzl): #hier auch potential in die klammer?
		actions=np.array(3)

		rl=np.sqrt(xl**2+yl**2+zl**2)

		#angular moments without mass
		Lx=yl*vzl-zl*vyl
		Ly=zl*vxl-xl*vzl
		Lz=xl*vyl-yl*vxl

		L=np.sqrt(Lx**2+Ly**2+Lz**2)

		phipot=potential(rl)

		E=vxl**2./2.+vyl**2./2.+vzl**2./2.+phipot

		def periapocenter(self,r):
    			pot=potential(r)
    			return (1/r)**2.+2.*(pot-E[0])/L[0]**2.

		rmin=opt.fsolve(periapocenter,np.min(rl)) #nicht min(rl) sondern einfach kleienr wert weil ich es erst durch orbit integration weiss
		rmax=opt.fsolve(periapocenter,np.max(rl))

		def jrint(self,r,E,L):
		    	pot=potential(r)
		    	return np.sqrt(2.*E-2.*pot-L**2./r**2.)

		J_phi=Lz
		J_theta=L-np.abs(Lz)

		J_r=np.zeros(len(J_phi))
		for i in range(len(J_r)):
    			J_r[i]=1/np.pi*intg.quad(jrint,rmin,rmax)[0]

		#J_ri=cs.G*mges/np.sqrt(-2.*E)-1./2.*(L+np.sqrt(L**2.-4.*cs.G*mges*b))
		
		
		actions[0]=J_phi
		actions[1]=J_theta
		actions[2]=J_r

		return actions


			
