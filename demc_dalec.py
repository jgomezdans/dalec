#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import numpy as np
import scipy.stats
import pdb
import matplotlib.pyplot as plt
from demc_class import DEMC_sampler
# Actually, the next class maybe ought to be imported in its own method
# rather than for the whole script here... 
from dalec import dalec

"""
The DEMC class modified for DALEC fun'n'games. The major changes are that
it reads the data (drivers and observations), and defines the prior and
likelihood functions. Everything else is from DEMC_sampler (demc_class.py).
"""
class jeffreys (object):
    def __init__ ( self ):
        return
    def pdf ( self, x ):
        return 1./x
        
class normal_trunc:
    def __init__ ( self, hi, lo, mean=0., sigma=1. ):
        self.mean = mean
        self.sigma = sigma
        self.hi = hi
        self.lo = lo
        
    def pdf ( self, x ):
        if lo <= x <= hi:
            return 1./(np.sqrt(2*np.pi)*self.sigma)*np.exp(-0.5*(x-self.mu)**2/(self.sigma*self.sigma))
        else:
            return -np.inf
            
class DEMC_dalec ( DEMC_sampler ):
    """
    This class assumes that there are global parameters, and per-cell parameters
    The code is fairly general, and although the focus in on SHITEFIRE here,
    only a function that calls the model is needed. Typically, you could
    subclass this object, or use it as boilerplate code for other models, as
    this object implements the DEMC sampler.
    """
    def __init__ ( self, meteo_drivers, observations, num_years=1,\
                truncate=True, num_population=4, obs_thin=1, \
                CR=1.0, F=2.38, pSnooker=0.1, pGamma1=0.1, \
                n_generations=1000,
                n_thin=1, n_burnin=0, eps_mult=0.1, eps_add=0, \
                    logger="./NONAME.log" ):
        """
        The constructor. Most of the variables are to do with the setting up
        of the DEMC sampler. grids and year are to do with the grid ids and
        year we are considering, and they are required here so as to setup
        the problem (reading drivers, calibration data etc.). The latter is
        accomplished in self._prepare_calibration() (unsurprisingly!)
        """
        DEMC_sampler.__init__ ( self, num_population, CR = CR, F = F, \
                pSnooker = pSnooker, pGamma1 = pGamma1, \
                n_generations = n_generations, n_thin = n_thin, \
                n_burnin = n_burnin, eps_mult = eps_mult,  \
                eps_add = eps_add, logger = logger )

        self.meteo_data = np.loadtxt( meteo_drivers, delimiter="," )
        self.obs_nee = np.loadtxt ( observations, delimiter="," )
        self.missing_obs = self.obs_nee[ :, 1] < -9998
        passer = self.meteo_data[:,0] <= 365*num_years
        self.meteo_data = self.meteo_data[ passer, :]
        self.obs_nee = self.obs_nee[ passer ]
        self.missing_obs = self.missing_obs[ passer ]
        
        self.truncate = truncate
        self.obs_thin = obs_thin
        self.x_init = np.array ( [0.0007393872, 0.6310229, \
                    0.3276926,  0.2932357, 0.04744657, 1e-006, 0.006676273, \
                    0.03251259, 3.667811e-005, 0.1475122, 10.52553, 345.7663, \
                    14.08511, 0.7, 0.02686367, 0.1724897, 350.4624, 0.513303, \
                    4891.44,134.927,82.27539,74.74379,12526.28 ])
                    
        self.lo_val = np.array( [ 1e-006, 0.2, 0.01, 0.01, 0.0001, 1e-006, \
                0.0001, 1e-005, 1e-006, 0.05, 5, 2e+002, 8, 0.1, 0.0001, \
                0.01, 1e+002, 0, 0, 0, 0, 0, 0 ] )
        self.hi_val = np.array( [ 0.01, 0.7, 0.5, 0.5, 0.1, 0.01, \
                0.01, 0.1, 0.01, 0.2, 20, 4e+002, 15, 0.7, 0.1, 0.5, \
                5e+002, 4e+002, 2.5e+004, 3e+002, 2e+002, 2e+002, 4e+004 ] )
        self.med_val = ( self.hi_val + self.lo_val )/2.
        self.x_init = self.med_val
        self.flux_unc = 0.58
        self.it = 0
        self.psid = -2.0
        self.rtot = 1.
        self.lma = 60.
        self.nit = 2.7
        self.lat = 42.2
    def prior_probabilities ( self, theta ):
        """ The method that calculates the prior (log) probabilities. This is based on the prior distributions given in prior_distributions, and assumes independence, so we just add them up.
        """
        
        p = np.array([ getattr ( self, self.parameters[i]).pdf ( theta[i]) \
                    for i in xrange(len(self.parameters)) ])
        
        if np.any ( p == 0):
            return -np.inf
        else:
            return np.log(p).sum()
                
    def _transform_variables_lognorm ( self, theta ):
        retval = np.exp( theta - 1. ) * self.x_init
        return retval
                    
                    
        
    def _transform_variables_uniform ( self, theta ):
        """
        Do a quantile transformation for a uniform transformation ;)
        """
        retval = theta*( self.hi_val - self.lo_val ) + self.lo_val
        return retval   

    def likelihood_function ( self, theta ):
        """
        This calculates the likelihood function. It first transforms the
        theta vector into a parameters per grid cell that are then passed on to
        the model object.
        """
        # Load the parameters into a dictionary indexed by grid_ids
        parameters = self._transform_variables_lognorm ( theta )
        if self.truncate:
            if np.any ( parameters > self.hi_val ) or \
                        np.any ( parameters < self.lo_val ):
                return -np.inf
        # Parameter have been loaded nto parameter_list for all grid cells
        # All that needs doing now is to run the model forward for each
        # grid cell and calculate the likelihood function
        retval = dalec ( parameters[17:], parameters[:17], self.meteo_data, \
            self.psid, self.rtot, self.lma, self.lat, self.nit )
        model_nee =  retval[ :, -4 ]
        
        delta = (self.obs_nee[~self.missing_obs, 1] - \
                    model_nee[~self.missing_obs] )**2
        delta = delta / (self.flux_unc**2 )            
        delta = delta[::self.obs_thin]
        p = -0.5*np.sum ( delta )

        p -= delta.shape[0]*( np.log ( np.sqrt(2.*np.pi)*self.flux_unc ))
        self.it += 1
        #pdb.set_trace()
        if self.it%500 == 0:
            print self.it, np.sum(delta), p
            self._plot_fit ( model_nee, p )
        return p

    def _plot_fit ( self, model_nee, lklihood, flag="simple" ):
        """Diagnostic plots
        """
        plt.clf()
        plt.plot ( self.meteo_data[ :, 0 ], model_nee, '-r' )
        plt.plot( self.meteo_data[ ~self.missing_obs, 0], \
                    self.obs_nee[~self.missing_obs, 1 ], '-b', lw=0.3 )
        plt.plot( self.meteo_data[ ~self.missing_obs, 0][::self.obs_thin], \
            self.obs_nee[~self.missing_obs, 1 ][::self.obs_thin], 'o', \
                markerfacecolor='none',markeredgecolor='b' )
        plt.vlines ( self.meteo_data[ ~self.missing_obs, 0], \
                    self.obs_nee[~self.missing_obs, 1 ] - 3*self.flux_unc, \
                    self.obs_nee[~self.missing_obs, 1 ] + 3*self.flux_unc, \
                    color='0.8' )
        plt.title ("Iteration %d. Loglikelihood: %f" % ( self.it, lklihood ) )
        plt.axis ([0, 365, -10, 10])
        plt.grid ( True )
        plt.savefig ( "/media/My Passport/inversions/%s_%d_iter%08d.png" % \
                    ( flag, self.obs_thin, self.it), dpi=150 )
        plt.close()
        self._tweet ( "Saved plot " + \
            "to /media/My\ Passport/inversions/%s_%d_iter%08d.png" \
            % (flag, self.obs_thin, self.it ) )
            
    def fitness ( self, theta ):
        """
        The new posterior probability in log. Convenience, really
        """
        
        prior = self.prior_probabilities ( theta )
        if np.isneginf( prior ):
            return -np.inf #np.log(1.0E-300)
        else:
            likelihood = self.likelihood_function ( theta )
            
            return  likelihood + prior

class DEMC_dalec_thin ( DEMC_dalec ):
    def likelihood_function ( self, theta ):
        """
        This calculates the likelihood function. It first transforms the
        theta vector into a parameters per grid cell that are then passed on to
        the model object.
        """
        # Load the parameters into a dictionary indexed by grid_ids
        parameters = self._transform_variables_uniform ( theta )
        # Parameter have been loaded nto parameter_list for all grid cells
        # All that needs doing now is to run the model forward for each
        # grid cell and calculate the likelihood function
        retval = dalec ( self.x_init, parameters, self.meteo_data )
        model_nee =  retval[ :, -4 ]
        
        delta = (self.obs_nee[~self.missing_obs, 1] - \
        model_nee[~self.missing_obs] )**2
        p = 0.5*np.sum ( delta[::10] ) / (self.flux_unc**2 )
        p -= np.sum ( np.log ( 2.*np.pi*self.flux_unc ))
        return p

class DEMC_dalec_arf ( DEMC_dalec ):
    def likelihood_function ( self, theta ):
        """
        This calculates the likelihood function. It first transforms the
        theta vector into a parameters per grid cell that are then passed on to
        the model object.
        """
        # Load the parameters into a dictionary indexed by grid_ids
        tau = theta[0]
        #parameters = self._transform_variables_lognorm ( theta[1:] )
        parameters = theta[1:]
        if self.truncate:
            if np.any ( parameters > self.hi_val ) or \
                    np.any ( parameters < self.lo_val ):
                return -np.inf
        #print tau
        # Parameter have been loaded nto parameter_list for all grid cells
        # All that needs doing now is to run the model forward for each
        # grid cell and calculate the likelihood function
        retval = dalec ( parameters[17:], parameters[:17], self.meteo_data, \
                self.psid, self.rtot, self.lma, self.lat, self.nit )
        model_nee =  retval[ :, -4 ]
        first = True
        p = 0.0
        
        n_obs = self.missing_obs.shape[0]
        for isample in np.arange ( self.missing_obs.shape[0] ):
            if not self.missing_obs [ isample ]:
                if first:
                    # First observation, edge condition, so asymptotic behaviour
                        
                    first = False
                    t_prev = self.meteo_data[ isample, 0 ]

                    p -= np.sum ( np.log ( 2.*np.pi*self.flux_unc ))
                    delta = (self.obs_nee[isample, 1] - \
                            model_nee[isample] )
                    p -= 0.5*(delta**2)/self.flux_unc**2
                    delta_prev = delta
                    # Remember to store t_prev and  delta_prev
                else:
                    # Autoregressive part
                    t_curr = self.meteo_data [isample, 0]
                    # sigma temperated by the temporal correlation
                    sigma_dash = self.flux_unc * np.sqrt ( \
                        (1. - np.exp (-2.*(t_curr-t_prev)/tau))  )
                    delta = (self.obs_nee[isample, 1] - \
                            model_nee[isample] )
                    arf = delta - np.exp (-(t_curr-t_prev)/tau)*delta_prev
                    arf2 = arf*arf 
                    p -= np.log ( np.sqrt(2.*np.pi)*sigma_dash )
                    p -= 0.5*arf2/(sigma_dash*sigma_dash)
                    t_prev = t_curr
                    delta_prev = delta
        self.it += 1
        if self.it % 500 == 0:
            print self.it, p
            print "**", parameters
            self._plot_fit ( model_nee, p, flag="arf" )
                
        return p
                    

        
        

def mcmc_arf ( num_years ):
    cal_dalec = DEMC_dalec_arf ( "sim04_met_l10_g6_n58.csv", \
            "sim04_obs_l10_g6_n58.csv", num_years=num_years, truncate=True,\
            n_generations=1000, n_burnin=1, n_thin=1, \
                logger="dalec_arf.log")
    
    # Need to define priors here...
    
    #parameter_list=[ \
        #['tau','scipy.stats.reciprocal(0.01, 2.00)'], \
        #['p1', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        #['p2', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        #['p3', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        #['p4', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        #['p5', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        #['p6', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        #['p7', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        #['p8', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        #['p9', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        #['p10', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        #['p11', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        #['p12', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        #['p13', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        #['p14', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        #['p15', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        #['p16', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        #['p17', 'scipy.stats.uniform( 0.01, 0.99 )'] , \
        #['Cf_init', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        #['Cw_init', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        #['Cr_init', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        #['Clab_init', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        #['Clit_init', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        #['Csom_init', 'scipy.stats.uniform( 0.01, 0.99 )'] \
        #]
    #parameter_list=[ \
        #['tau', 'scipy.stats.reciprocal( 1, 100)'], \
        #['p1', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        #['p2', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        #['p3', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        #['p4', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        #['p5', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        #['p6', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        #['p7', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        #['p8', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        #['p9', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        #['p10', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        #['p11', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        #['p12', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        #['p13', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        #['p14', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        #['p15', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        #['p16', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        #['p17', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'] , \
        #['Cf_init', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        #['Cw_init', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        #['Cr_init', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        #['Clab_init', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        #['Clit_init', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        #['Csom_init', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'] \
        #]
        
    parameters = ['tau','p1','p2','p3','p4','p5','p6','p7','p8','p9', \
            'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', \
            'Cf_init', 'Cw_init', 'Cr_init', 'Clab_init', 'Clit_init', \
            'Csom_init' ]
    true_vals = np.array([5.00E-04,0.45,0.4,0.4,6.00E-02,7.00E-05,\
            8.00E-03,3.00E-02,3.00E-05,7.30E-02,1.40E+01,240,9,0.48,\
            9.00E-02,0.15,300,0,5,5,100,5,9900])
    
    lo_val = np.array( [ 1e-006, 0.2, 0.01, 0.01, 0.0001, 1e-006, \
        0.0001, 1e-005, 1e-006, 0.05, 5, 2e+002, 8, 0.1, 0.0001, \
        0.01, 1e+002, 0, 0, 0, 0, 0, 0 ] ) + 1.e-6
    hi_val = np.array( [ 0.01, 0.7, 0.5, 0.5, 0.1, 0.01, \
        0.01, 0.1, 0.01, 0.2, 20, 4e+002, 15, 0.7, 0.1, 0.5, \
        5e+002, 4e+002, 2.5e+004, 3e+002, 2e+002, 2e+002, 4e+004 ] )
    med_val = ( hi_val + lo_val )/2.
    parameter_list = [['tau', 'scipy.stats.reciprocal( 1, 100)']]
    for ( i, param ) in enumerate ( parameters[1:] ):
        u = [ param, 'scipy.stats.reciprocal( %g, %g)' % \
                ( lo_val[i], hi_val[i] ) ]
        parameter_list.append ( u )
    print parameter_list
    cal_dalec.prior_distributions ( parameter_list, parameters )
    Z = cal_dalec.ProposeStartingMatrix ( 200 )
    #result = np.load ( "truncated_results/mcmc_results_full_OK.npz" )
    #Z[1:, :1 ] = np.ones_like (Z[1:, :1])* true_vals[:,np.newaxis]
    #Z[1:, 1: ] = np.ones_like (Z[1:, 1: ])* med_val[:,np.newaxis]
    #result = None
    (Z_out, accept_rate) = cal_dalec.demc_zs ( Z )
    
    if Z_out.shape[0] == 17:
        zt = Z_out*0.0
        selected = np.arange ( 17 )
        for p in selected:
            zt [ p, :] = Z_out[ p, : ]*(hi_val[p] - lo_val[p]) + lo_val[p]
            
    else:
        selected = np.arange ( 1, 18 )
        zt = Z_out*0.0
        for p in selected:
            zt [ p, :] = Z_out[ p, : ]*(hi_val[p-1] - lo_val[p-1]) + lo_val[p-1]
        zt[0,:] = Z_out[0,:]
    np.savez("mcmc_arf_results.npz", Z_out=Z_out, zt=zt )


def mcmc ( ):
    cal_dalec = DEMC_dalec ( "sim04_met_l10_g6_n58.csv", \
            "sim04_obs_l10_g6_n58.csv", num_population=4, \
            n_generations=1000, n_burnin=1, n_thin=1, \
                logger="dalec_plain.log")
    
    # Need to define priors here...
    #parameter_list=[ \
        #['p1', 'scipy.stats.uniform( 0.01, 0.98 )'], \
        #['p2', 'scipy.stats.uniform( 0.01, 0.98 )'], \
        #['p3', 'scipy.stats.uniform( 0.01, 0.98 )'], \
        #['p4', 'scipy.stats.uniform( 0.01, 0.98 )'], \
        #['p5', 'scipy.stats.uniform( 0.01, 0.98 )'], \
        #['p6', 'scipy.stats.uniform( 0.01, 0.98 )'], \
        #['p7', 'scipy.stats.uniform( 0.01, 0.98 )'], \
        #['p8', 'scipy.stats.uniform( 0.01, 0.98 )'], \
        #['p9', 'scipy.stats.uniform( 0.01, 0.98 )'], \
        #['p10', 'scipy.stats.uniform( 0.01, 0.98 )'], \
        #['p11', 'scipy.stats.uniform( 0.01, 0.98 )'], \
        #['p12', 'scipy.stats.uniform( 0.01, 0.98 )'], \
        #['p13', 'scipy.stats.uniform( 0.01, 0.98 )'], \
        #['p14', 'scipy.stats.uniform( 0.01, 0.98 )'], \
        #['p15', 'scipy.stats.uniform( 0.01, 0.98 )'], \
        #['p16', 'scipy.stats.uniform( 0.01, 0.98 )'], \
        #['p17', 'scipy.stats.uniform( 0.01, 0.98 )'] , \
        #['Cf_init', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        #['Cw_init', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        #['Cr_init', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        #['Clab_init', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        #['Clit_init', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        #['Csom_init', 'scipy.stats.uniform( 0.01, 0.99 )'] \
        #]
    parameter_list=[ \
        ['p1', 'scipy.stats.lognorm (    0.5*0.5  )'], \
        ['p2', 'scipy.stats.lognorm (    0.5*0.5  )'], \
        ['p3', 'scipy.stats.lognorm (    0.5*0.5  )'], \
        ['p4', 'scipy.stats.lognorm (    0.5*0.5  )'], \
        ['p5', 'scipy.stats.lognorm (    0.5*0.5  )'], \
        ['p6', 'scipy.stats.lognorm (    0.5*0.5  )'], \
        ['p7', 'scipy.stats.lognorm (    0.5*0.5  )'], \
        ['p8', 'scipy.stats.lognorm (    0.5*0.5  )'], \
        ['p9', 'scipy.stats.lognorm (    0.5*0.5  )'], \
        ['p10', 'scipy.stats.lognorm (   0.5*0.5  )'], \
        ['p11', 'scipy.stats.lognorm (   0.5*0.5  )'], \
        ['p12', 'scipy.stats.lognorm (   0.5*0.5  )'], \
        ['p13', 'scipy.stats.lognorm (   0.5*0.5  )'], \
        ['p14', 'scipy.stats.lognorm (   0.5*0.5  )'], \
        ['p15', 'scipy.stats.lognorm (   0.5*0.5  )'], \
        ['p16', 'scipy.stats.lognorm (   0.5*0.5  )'], \
        ['p17', 'scipy.stats.lognorm (   0.5*0.5  )'] , \
        ['Cf_init', 'scipy.stats.lognorm (   0.5*0.5  )'], \
        ['Cw_init', 'scipy.stats.lognorm (   0.5*0.5  )'], \
        ['Cr_init', 'scipy.stats.lognorm (   0.5*0.5  )'], \
        ['Clab_init', 'scipy.stats.lognorm ( 0.5*0.5  )'], \
        ['Clit_init', 'scipy.stats.lognorm ( 0.5*0.5  )'], \
        ['Csom_init', 'scipy.stats.lognorm ( 0.5*0.5  )'] \
        ]
        

    parameters = ['p1','p2','p3','p4','p5','p6','p7','p8','p9', \
            'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', \
            'Cf_init', 'Cw_init', 'Cr_init', 'Clab_init', 'Clit_init', \
            'Csom_init' ]
    cal_dalec.prior_distributions ( parameter_list, parameters )
    Z = cal_dalec.ProposeStartingMatrix ( 50 )
    result = np.load ( "mcmc_results.npz" )
    (Z_out, accept_rate) = cal_dalec.demc_zs ( result['Z_out'][:,:200] )
    (Z_out, accept_rate) = cal_dalec.demc_zs ( Z )
    lo_val = np.array( [ 1e-006, 0.2, 0.01, 0.01, 0.0001, 1e-006, \
        0.0001, 1e-005, 1e-006, 0.05, 5, 2e+002, 8, 0.1, 0.0001, \
        0.01, 1e+002, 0, 0, 0, 0, 0, 0 ] )
    hi_val = np.array( [ 0.01, 0.7, 0.5, 0.5, 0.1, 0.01, \
        0.01, 0.1, 0.01, 0.2, 20, 4e+002, 15, 0.7, 0.1, 0.5, \
        5e+002, 4e+002, 2.5e+004, 3e+002, 2e+002, 2e+002, 4e+004 ] )
    
    if Z_out.shape[0] == 23:
        zt = Z_out*0.0
        selected = np.arange ( 23 )
        for p in selected:
            zt [ p, :] = Z_out[ p, : ]*(hi_val[p] - lo_val[p]) + lo_val[p]
            
    else:
        selected = np.arange ( 1, 24 )
        zt = Z_out*0.0
        for p in selected:
            zt [ p, :] = Z_out[ p, : ]*(hi_val[p-1] - lo_val[p-1]) + lo_val[p-1]
        zt[0,:] = Z_out[0,:]
    np.savez("mcmc_results.npz", Z_out=Z_out, zt=zt )

def mcmc2 ( ):
    cal_dalec = DEMC_dalec ( "sim04_met_l10_g6_n58.csv", \
            "sim04_obs_l10_g6_n58.csv", num_population=4, \
            obs_thin=40, \
            n_generations=1000, n_burnin=1, n_thin=1, \
                logger="dalec_plain_thinned.log")
    
    # Need to define priors here...
    #parameter_list=[ \
        #['p1', 'scipy.stats.uniform( 0.01, 0.98 )'], \
        #['p2', 'scipy.stats.uniform( 0.01, 0.98 )'], \
        #['p3', 'scipy.stats.uniform( 0.01, 0.98 )'], \
        #['p4', 'scipy.stats.uniform( 0.01, 0.98 )'], \
        #['p5', 'scipy.stats.uniform( 0.01, 0.98 )'], \
        #['p6', 'scipy.stats.uniform( 0.01, 0.98 )'], \
        #['p7', 'scipy.stats.uniform( 0.01, 0.98 )'], \
        #['p8', 'scipy.stats.uniform( 0.01, 0.98 )'], \
        #['p9', 'scipy.stats.uniform( 0.01, 0.98 )'], \
        #['p10', 'scipy.stats.uniform( 0.01, 0.98 )'], \
        #['p11', 'scipy.stats.uniform( 0.01, 0.98 )'], \
        #['p12', 'scipy.stats.uniform( 0.01, 0.98 )'], \
        #['p13', 'scipy.stats.uniform( 0.01, 0.98 )'], \
        #['p14', 'scipy.stats.uniform( 0.01, 0.98 )'], \
        #['p15', 'scipy.stats.uniform( 0.01, 0.98 )'], \
        #['p16', 'scipy.stats.uniform( 0.01, 0.98 )'], \
        #['p17', 'scipy.stats.uniform( 0.01, 0.98 )'] , \
        #['Cf_init', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        #['Cw_init', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        #['Cr_init', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        #['Clab_init', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        #['Clit_init', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        #['Csom_init', 'scipy.stats.uniform( 0.01, 0.99 )'] \
        #]
    parameter_list=[ \
        ['p1', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        ['p2', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        ['p3', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        ['p4', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        ['p5', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        ['p6', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        ['p7', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        ['p8', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        ['p9', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        ['p10', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        ['p11', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        ['p12', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        ['p13', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        ['p14', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        ['p15', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        ['p16', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        ['p17', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'] , \
        ['Cf_init', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        ['Cw_init', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        ['Cr_init', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        ['Clab_init', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        ['Clit_init', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'], \
        ['Csom_init', 'scipy.stats.norm ( loc=1, scale=0.5*0.5  )'] \
        ]
        

    parameters = ['p1','p2','p3','p4','p5','p6','p7','p8','p9', \
            'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', \
            'Cf_init', 'Cw_init', 'Cr_init', 'Clab_init', 'Clit_init', \
            'Csom_init' ]
    cal_dalec.prior_distributions ( parameter_list, parameters )
    Z = cal_dalec.ProposeStartingMatrix ( 250 )
    #result = np.load ( "mcmc_results.npz" )
    #(Z_out, accept_rate) = cal_dalec.demc_zs ( result['Z_out'][:,:200] )
    (Z_out, accept_rate) = cal_dalec.demc_zs ( Z )
    lo_val = np.array( [ 1e-006, 0.2, 0.01, 0.01, 0.0001, 1e-006, \
        0.0001, 1e-005, 1e-006, 0.05, 5, 2e+002, 8, 0.1, 0.0001, \
        0.01, 1e+002, 0, 0, 0, 0, 0, 0 ] )
    hi_val = np.array( [ 0.01, 0.7, 0.5, 0.5, 0.1, 0.01, \
        0.01, 0.1, 0.01, 0.2, 20, 4e+002, 15, 0.7, 0.1, 0.5, \
        5e+002, 4e+002, 2.5e+004, 3e+002, 2e+002, 2e+002, 4e+004 ] )
    x_init = np.array ( [0.0007393872, 0.6310229, \
        0.3276926,  0.2932357, 0.04744657, 1e-006, 0.006676273, \
        0.03251259, 3.667811e-005, 0.1475122, 10.52553, 345.7663, \
        14.08511, 0.7, 0.02686367, 0.1724897, 350.4624, 0.513303, \
        4891.44,134.927,82.27539,74.74379,12526.28 ])
    if Z_out.shape[0] == 23:
        zt = Z_out*0.0
        selected = np.arange ( 23 )
        for p in selected:
            zt [ p, :] = np.exp(Z_out[ p, : ]-1)*x_init[p]#*(hi_val[p] - lo_val[p]) + lo_val[p]
            
    else:
        selected = np.arange ( 1, 24 )
        zt = Z_out*0.0
        for p in selected:
            zt [ p, :] = Z_out[ p, : ]*(hi_val[p-1] - lo_val[p-1]) + lo_val[p-1]
        zt[0,:] = Z_out[0,:]
    np.savez("mcmc_results_thinned.npz", Z_out=Z_out, zt=zt )


if __name__ == "__main__":
    print sys.argv
    if len ( sys.argv ) == 1:
        print "Doing simple MCMC"
        mcmc2()
    else:
        if sys.argv[1] == "arf":
            print "Doing autoregressive likelihood"
        if 1 <= int(sys.argv[2]) <= 10:
            num_years = int ( sys.argv[2] )
        else:
            print "Number of years to calibrate can only be between 1 and 10"
        mcmc_arf( num_years=num_years )