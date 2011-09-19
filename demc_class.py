# -*- coding: utf-8 -*-
import numpy
import numpy.linalg
import scipy.stats
import sys, pdb
from time import time

MAX_ITERATIONS = 3000000
def full_gauss_den(x, mu, va, log):
    """ This function is the actual implementation
    of gaussian pdf in full matrix case.

    It assumes all args are conformant, so it should
    not be used directly Call gauss_den instead

    Does not check if va is definite positive (on inversible
    for that matter), so the inverse computation and/or determinant
    would throw an exception."""
    d       = mu.size
    inva    = numpy.linalg.inv(va)
    fac     = 1 / numpy.sqrt( (2*numpy.pi) ** d * numpy.fabs(numpy.linalg.det(va)))

    # we are using a trick with sum to "emulate"
    # the matrix multiplication inva * x without any explicit loop
    #y   = -0.5 * N.sum(N.dot((x-mu), inva) * (x-mu), 1)
    y   = -0.5 * numpy.dot(numpy.dot((x-mu), inva) * (x-mu),
            numpy.ones((mu.size, 1), x.dtype))[:, 0]

    if not log:
        y   = fac * numpy.exp(y)
    else:
        y   = y + numpy.log(fac)

    return y

class DEMC_sampler(object):
    """
    =======================
    Python DEMC sampler
    =======================
    A python implementation of the Differential Evoluation MCMC sampler of ter Braak (2008) (ter  Braak C. J. F., and Vrugt J. A. (2008). Differential Evolution Markov Chain
    with snooker updater and fewer chains. Statistics and Computing http://dx.doi.org/10.1007/s11222-008-9104-9). It appears that this implementation (and its daughter, DREAM) provide a really nice and efficient MCMC sampling scheme. An additional plus is that the algorithm can be easily parallelised.

    The DEMC_sampler object *should not be used directly* (only for testing purposes), but it should be subclassed. Subclassing implies adding your own likelihood function, rather than the provided likelihoood function. The following python packages are required
        - numpy ( & linalg, usually comes as standard)
        - scipy (we need the stats objects from there)

    An example run
    ~~~~~~~~~~~~~~

    Class usage is very simple. The following example defines a DEMC sampler

  code-block:: python
    DEMC = DEMC_sampler ( 10, n_generations=1000 )
    parameter_list=[['x1', 'scipy.stats.uniform(-20, 40)'], ['x2', 'scipy.stats.uniform(-20, 40)']]
    parameters = ['x1','x2']
    DEMC.prior_distributions ( parameter_list, parameters )
    Z = DEMC.ProposeStartingMatrix ( 150 )
    (Z_out, accept_rate) = DEMC.demc_zs ( Z )

    The first call instantiates the DEMC_sampler class. It is initialised with population size (10 in this example), and with the number of generations (or iterations).  The second line defines a list of pairs of parameter names, distributions. The distributions are distribution objects from scipy.stats, or code that implements the same methods that the user provides. Note that we do not provide a reference to these objects, but write the reference as a string (the reference is obtained internally). We also provide a list with the parameter names in the same order they are required in the M{\theta} parameter vector. These two lists are then included in the object by calling the prior_distributions() method. The sampler is initialised with a M{Z} matrix (num_params x [...]), sampled from the prior distributions. Or, you can provide your own. Finally, the sampler is started and it returns the Z_out matrix (the samples for the different parameters) and the accepctance rate (try to get it ~0.2-0.4)
    """
    def __init__ ( self, num_population, logger=True, CR=1.0, F=2.38, pSnooker=0.1, pGamma1=0.1, n_generations=1000, n_thin=5, n_burnin=200, eps_mult=0.1, eps_add=0) :
        """
        The class creator. You should pass it the populations (say between 10 or 100), and the sampler parameters, such as n_generations, n_thing and n_burnin. The other parameters are DEMC-related parameters and shouldn't need changing accroding to Vrugt and ter Braak. CR is a crossover rate, and at present, is not used in this implementation of the code (but could be used in a future implementation that does DREAM)
        """
        self.num_population = num_population
        self.CR = CR
        self.F = F
        self.pSnooker = pSnooker
        self.pGamma1 = pGamma1
        self.n_generations = n_generations
        self.n_thin = n_thin
        self.n_burnin =n_burnin
        self.eps_mult = eps_mult
        self.eps_add = eps_add
        if logger==True:
            print "Logging by default to ./DEMC_ZS.log"
            logFile = "./DEMC_ZS.log"
            self.logger = open(logFile,"w")
        if type(logger)==str:
            logFile = logger
            self.logger = open(logFile,"w")
        if logger==None:
            self.logger = None
        self._tweet ("Started simulation!")
    def _tweet ( self, message ):
        if self.logger<>None:
            import time
            self.logger.write ("%s - [ %s ]\n"%(time.asctime(), message ))
            self.logger.flush()
    
    def choose_without_replacement(self, m,n,repeats=None):
        """Choose n nonnegative integers less than m without replacement
        Returns an array of shape n, or (n,repeats).

        This code is from Anne Archiebald, as numpy/scipy don't seem to have this facility.... It looks like it might not be the most efficient code in the world
        """
        if repeats is None:
            r = 1
        else:
            r = repeats
        if n>m:
            raise ValueError, "Cannot find %d nonnegative integers less than %d" %  (n,m)
        if n>m/2:
            res = numpy.sort(numpy.random.rand(m,r).argsort(axis=0)[:n,:],axis=0)
        else:
            res = numpy.random.random_integers(m,size=(n,r))
            while True:
                res = numpy.sort(res,axis=0)
                w = numpy.nonzero(numpy.diff(res,axis=0)==0)
                nr = len(w[0])
                if nr==0:
                    break
                res[w] = numpy.random.random_integers(m,size=nr)
        if repeats is None:
            return res[:,0]
        else:
            return res

    def prior_distributions ( self, parameter_list, parameters ):
        """This function defines the prior distributions from a (python) list. Useful to quickly add/update these parameters. The parameters list (2nd argument) is there to have the parameter names in the same order that vector M{\Theta} will have. So if theta required by the model is [ par1, par2, par3], then parameters=['par1','par2','par3']

         Example parameter list:
         parameter_list = [['fuel_1hr',"scipy.stats.uniform ( 0,100.)"],\
         ['fuel_10hr', "scipy.stats.uniform ( 0,100.)"],\
         ['fuel_100hr', "scipy.stats.uniform ( 0,100.)"] ]

         The way this method works is by using setattr to add the prior methods to the self object. Note that in this particular implementation, the parameter distributions are independent (i.e., you can't have correlation between parameters). If this is an issue, you'll need to change the code.

         In Bayesian statistics, the prior distributions encapsulate whatever knowledge we have about the distribution of a parameter prior to doing any experiment. On the one hand, they add a degree of subjectiveness, as different people may translate their knowledge in different distribution shapes. Also, it is sometimes useful to use non-informative priors to indicate ignorance or "objectivity" (yeah, right...). In practice, broad uniform distributions, truncated normals, lognormal distributions are often used. In sequential applications of a calibration exercise, the posterior of the previous run can be used as the prior for the current state.
        """
        self.parameters = parameters # List is in the order of the parameter vector
        for [k,v] in parameter_list:
            setattr(self, k, eval( v ))
            self._tweet("Parameter: %s. Dist: %s"%(k,v))
    
    
    def prior_probabilities ( self, theta ):
        """ The method that calculates the prior (log) probabilities. This is based on the prior distributions given in prior_distributions, and assumes independence, so we just add them up.
        """
        p = numpy.array([ numpy.log ( getattr ( self, self.parameters[i]).pdf ( theta[i])) for i in xrange(len(self.parameters)) ]).sum()
        
        #if numpy.isneginf(p):
            #p = numpy.log(1.0E-300)
        return p

    def likelihood_function ( self, theta ):
        """For example! This function ought to be overridden by the user, and maybe extended with whatever extra parameters you need to get hold of your observations, or model driver parameters.
        This function method calculates the likelihood function for a vector M{\theta}. Usually, you have a model you run with these parameters as inputs (+ some driver data), and some observations that go with the output of the forward model output. These two sets of values are combined in some sort of cost function/likelihood function. A common criterion is to assume that the model is able to perfectly replicate the observations (given a proper parametrisation). The only mismatch between model output and observations is then given by the uncertainty with which the measurement is performed, and we can encode this as a zero-mean Normal distribution. The variance of this distribution is then related to the observational error. If different measurements are used, a multivariate normal is useful, and correlation between observations can also be included, if needs be.
        """
        means = numpy.matrix([-3.0, 2.8])
        means = numpy.matrix([-5.0, 5])
        sigma1 = 1.0
        sigma2 = 2#0.5
        rho = -0.5#-0.1
        covar = numpy.matrix([[sigma1*sigma1,rho*sigma1*sigma2],[rho*sigma1*sigma2,sigma2*sigma2]])
        inv_covar = numpy.linalg.inv ( covar ) # numpy.matrix([[  5.26315789,   9.47368421],\
                        #[  9.47368421,  21.05263158]])
        det_covar = numpy.linalg.det( covar ) #0.047499999999999987
        N = means.shape[0]
        X = numpy.matrix(means- theta)
        #X = theta
        #p = full_gauss_den(X, means, covar, True)
        #This is just lazy... Using libraries to invert a 2x2 matrix & calc. its determinant....
        #Also, the log calculations could be done more efficiently and stored, but...
        p = pow(1.0/(2*numpy.pi), N/2.)
        p = p / numpy.sqrt ( numpy.linalg.det (covar))
        #p = 0.73025296137109341 # Precalc'ed
        #p = p *    numpy.exp (-0.5*X*inv_covar*X.transpose())
        a = X*inv_covar*X.T
        p = p*numpy.exp(-0.5*a)
        #pdb.set_trace()
        p = numpy.log(p)
        if numpy.isneginf(p):
            p = numpy.log(1.0E-300)
        return p

    def fitness ( self, theta ):
        """
        The new posterior probability in log. Convenience, really
        """
        prior = self.prior_probabilities ( theta )
        if numpy.isneginf( prior ):
            return numpy.log(1.0E-300)
        else:
            return self.likelihood_function ( theta ) + prior
    

    def MonitorChains ( self, X ):
        (Npar, I, T2) = X.shape # (number of parameters, number of chains, number of iterations)
        T = T2/2 # Only use the latter half of the itarations. Deals with burn-in issues
        #We do it for each parameter, I guess?
        rhat = 0.
        quantiles = numpy.zeros ((Npar, 5))
        param_r = numpy.zeros(Npar)
        for par in xrange(Npar):
            x = X[par, :, (T+1):] # For convenience, subset per parameter. x is now (chains, iterations)
            psi_i = numpy.mean ( x,axis=1) # Check axis!
            psi = numpy.mean ( psi_i )
            B = numpy.sum ( (psi_i-psi)*(psi_i-psi))*(1/(I-1.))
            S = numpy.mean( (x-psi_i[:,numpy.newaxis])**2,axis=1)
            W = numpy.mean(S)
            V_hat = ((T-.1)/T)*W + (1+1./I)*B
            #pdb.set_trace()
            if rhat<numpy.sqrt(V_hat/W):
                rhat = numpy.sqrt(V_hat/W)
            quantiles_p =[ scipy.stats.scoreatpercentile ( x.flatten(1), percentile) for percentile in [2.5, 25., 50., 75., 97.5] ]
            quantiles[par,:] = numpy.array ( quantiles_p )
            param_r[par] = numpy.sqrt(V_hat/W)
        return ( rhat, param_r, quantiles )
    
    
    
    def _dump_diags ( self, rhat, param_r, quantiles, iteration ):
        self._tweet ("Convergence iagnostics @ iteration %s"%iteration)
        self._tweet ("Total rhat: %f"%rhat)
        for par in xrange(len(self.parameters)):
            self._tweet ("Param: %s; rhat: %f"%(self.parameters[par],param_r[par]))
        self._tweet("+----------------------------------------------------------------+")
        self._tweet("|Parameter|  2.5%    |  25%     |   50%    |  75 %    |  97.5%   |")
        for par in xrange(len(self.parameters)):
            self._tweet ("|%s|%10.2g|%10.2g|%10.2g|%10.2g|%10.2g|"%(self.parameters[par].rjust(9),quantiles[par,0], quantiles[par,1], quantiles[par,2],quantiles[par,3], quantiles[par,4]) )
        self._tweet("+----------------------------------------------------------------+")
    def ProposeStartingMatrix ( self, m0):
        """
        The  proposed starting matrix. We initialise the chain with this matrix. Usually, a draw from the prior distribution is suitable
        """
        if m0<=(self.num_population+len(self.parameters)):
            print "The size of the Z matrix needs to be larger than the population size + the number of parameters"
            sys.exit()
        Z = []
        for i in xrange(len(self.parameters)):
            Z.append(getattr ( self, self.parameters[i]).rvs ( size = m0))
        return numpy.array(Z)

    def demc_zs ( self,  Z):
        #CR=self.CR, F=self.F, pSnooker=0.1, pGamma1=0.1, n_generations=10000, n_thin=5, n_burnin=2000, eps_mult=0.1, eps_add=0
        #Z = numpy.zeros ( d, m0)
        #X = numpy.zeros (d, num_population)

        d = 2
        npass = 0
        X = Z[:,:self.num_population]
        m0 = Z.shape[1]
        self.discard = int(m0+self.num_population*numpy.floor(self.n_burnin/self.n_thin))
        mZ = Z.shape[1]
        Npar = X.shape[0]
        Npar12 = (Npar-1)/2. # Factor for Metropolis ratio DE snooker update
        F2 = self.F/numpy.sqrt ( 2.*Npar)
        F1 = 1.0
        accept = numpy.zeros ( self.n_generations )
        #iseq = numpy.arange(1, num_population)
        rr = 0.0 ; r_extra = 0
        #print int(m0+num_population*numpy.floor(n_burnin/n_thin))
        #Calculate the starting posteriors...
        Z_diagnostic = numpy.zeros ( (Npar, self.num_population, self.n_generations))
        logfitness_x = [ self.fitness(X[:,i]) for i in xrange(self.num_population) ]
        #Start of main loop
        iteration = -1
        T0 = time()
        while True:
            # We clear the acceptance counter
            accepti = 0
            for i in xrange(self.num_population):
                #Start of different chains loop
                #First decide whether this is a snooker update or not
                if (numpy.random.random()<self.pSnooker):
                    #Snooker update
                    #Select three chains
                    rr = self.choose_without_replacement(mZ-1, 3, repeats=None)
                    z = Z[:,rr[2]]
                    x_z = X[:,i] - z
                    #Difference between the current point and one of the 3 chains
                    D2 = max(numpy.sum(x_z*x_z), 1.0e-300) # This is the distance. Could do it with dot?
                    gamma_snooker = numpy.random.random()+1.2 # Snooker stochastic
                    proj_diff = numpy.sum((Z[:,rr[0]] - Z[:,rr[1]])*x_z)/D2 # Project the difference onto x_z. Normalize by x_z's norm
                    x_prop = X[:,i] + (gamma_snooker * proj_diff) * x_z # Proposed point
                    x_z = x_prop - z # update x_z
                    #pdb.set_trace()
                    D2prop = max( numpy.dot(x_z, x_z), 1.0e-30) # Calculate D2prop
                    r_extra = Npar12*(numpy.log(D2prop) - numpy.log(D2))
                else:
                    if (numpy.random.random()<self.pGamma1):
                        gamma_par = F1
                    else:
                        gamma_par = F2 * numpy.random.uniform( low=1-self.eps_mult, high=1+self.eps_mult, size=Npar)
                    rr = self.choose_without_replacement(mZ-1, 2, repeats=None)
                    if (self.eps_add==0):
                        x_prop = X[:,i] + gamma_par * ( Z[:,rr[0]] - Z[:,rr[1]])
                    else:
                        x_prop = X[:,i] + gamma_par * ( Z[:,rr[0]] - Z[:,rr[1]]) + self.eps_add*numpy.random.randn(Npar)
                    r_extra = 0
                logfitness_x_prop = self.fitness ( x_prop )
                logr = logfitness_x_prop - logfitness_x[i]
                if (logr + r_extra)>numpy.log ( numpy.random.random()):
                    accepti += 1
                    X[:,i] = x_prop
                    logfitness_x[i] = logfitness_x_prop
                Z_diagnostic [:, i, iteration] = X[:,i]
            accept[iteration] = accepti
            iteration += 1
            if iteration%100==0:
                T1 = time()
                self._tweet("Run 100 iterations more. Now at %d"%iteration)
                self._tweet("Speed: %f sims/s"%( 100./(T1-T0)))
                T0 = time()
            #print logfitness_x[i], logfitness_x_prop ,logr,r_extra,accepti,x_prop
            if iteration%self.n_thin==0:
                Z = numpy.c_[X,Z]
                mZ = Z.shape[1]
                #print "iteration ",iteration
                #print "\t",X[0,:]
                #print "\t",X[1,:]
            if (iteration% (self.n_generations)==0) and iteration>0:
                ( rhat, param_r, quantiles ) = self.MonitorChains ( Z_diagnostic)
                self._dump_diags ( rhat, param_r, quantiles, iteration )
                new_pop = iteration
                if rhat<=1.2:
                        npass += 1
                else:
                    if npass>0:
                        npass = 0
                if npass>2: break
                if iteration>MAX_ITERATIONS:
                    self._tweet ( "Maximum number of iterations exceeded. Bailing out...")
                    break
                accept = numpy.append ( accept, numpy.zeros (self.n_generations+1), 0)
                Z_diagnostic = numpy.append(Z_diagnostic, numpy.zeros ((Npar, self.num_population, self.n_generations+1)), 2)

        self._tweet ("Finished simulation!")
        self._tweet ( "Returned %d samples"%int(m0+iteration*numpy.floor(self.n_burnin/self.n_thin)))
        return (Z[:,:int(m0+iteration*numpy.floor(self.n_burnin/self.n_thin))], accept/self.num_population)

if __name__=="__main__":
    DEMC = DEMC_sampler ( 4, n_generations=400, n_burnin=1, n_thin=1, logger="test_me.log")
    parameter_list=[['x1', 'scipy.stats.norm(-100, 20)'], ['x2', 'scipy.stats.norm(-100, 20)']]
    parameters = ['x1','x2']
    DEMC.prior_distributions ( parameter_list, parameters )
    Z = DEMC.ProposeStartingMatrix ( 150 )
    (Z_out, accept_rate) = DEMC.demc_zs ( Z )
    import pylab
    pylab.figure();pylab.plot ( Z_out[0,:], Z_out[1,:], 'k,')
    pylab.figure();pylab.hist(Z_out[0,:],bins=10);pylab.hist(Z_out[1,:],bins=10);
    pylab.figure();pylab.hexbin(Z_out[0,:], Z_out[1,:], bins='log')
    pylab.show()
