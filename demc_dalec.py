#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats

from demc_class import DEMC_sampler
# Actually, the next class maybe ought to be imported in its own method
# rather than for the whole script here... 
from DALEC import dalec

"""
The DEMC class modified for DALEC fun'n'games. The major changes are that
it reads the data (drivers and observations), and defines the prior and
likelihood functions. Everything else is from DEMC_sampler (demc_class.py).
"""
class DEMC_dalec ( DEMC_sampler ):
    """
    This class assumes that there are global parameters, and per-cell parameters
    The code is fairly general, and although the focus in on SHITEFIRE here,
    only a function that calls the model is needed. Typically, you could
    subclass this object, or use it as boilerplate code for other models, as
    this object implements the DEMC sampler.
    """
    def __init__ ( self, meteo_drivers, observations, num_population=6, \
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

        self.meteo_data = np.loadtxt( meteo_drivers, delimiter="," )[:365]
        self.obs_nee = np.loadtxt ( observations, delimiter="," )[:365,:]
        self.missing_obs = self.obs_nee[ :, 1] < -9998
        self.x_init = np.array ( [ 0.513303, 4891.44, 134.927, 82.27539, \
                    74.74379, 12526.28 ] )
        self.lo_val = np.array( [ 1e-006, 0.2, 0.01, 0.01, 0.0001, 1e-006, \
                0.0001, 1e-005, 1e-006, 0.05, 5, 2e+002, 8, 0.1, 0.0001, \
                0.01, 1e+002, 0, 0, 0, 0, 0, 0 ] )
        self.hi_val = np.array( [ 0.01, 0.7, 0.5, 0.5, 0.1, 0.01, \
                0.01, 0.1, 0.01, 0.2, 20, 4e+002, 15, 0.7, 0.1, 0.5, \
                5e+002, 4e+002, 2.5e+004, 3e+002, 2e+002, 2e+002, 4e+004 ] )
        self.flux_unc = np.sqrt ( 0.3364 )
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
        parameters = self._transform_variables_uniform ( theta )
        # Parameter have been loaded nto parameter_list for all grid cells
        # All that needs doing now is to run the model forward for each
        # grid cell and calculate the likelihood function
        retval = dalec ( self.x_init, parameters, self.meteo_data )
        model_nee =  retval[ :, -4 ]
        
        delta = (self.obs_nee[~self.missing_obs, 1] - \
                    model_nee[~self.missing_obs] )**2
        p = 0.5*np.sum ( delta ) / (self.flux_unc**2 )
        p -= np.sum ( np.log ( 2.*np.pi*self.flux_unc ))
        return p

    def fitness ( self, theta ):
        """
        The new posterior probability in log. Convenience, really
        """
        
        prior = self.prior_probabilities ( theta )
        if np.isneginf( prior ):
            return np.log(1.0E-300)
        else:
            return self.likelihood_function ( theta ) + prior
                
if __name__ == "__main__":
    cal_dalec = DEMC_dalec ( "sim04_met_l10_g6_n58.csv", \
            "sim04_obs_l10_g6_n58.csv",  \
            n_generations=400, n_burnin=1, n_thin=1, \
                logger="dalec_test.log")
    
    # Need to define priors here...
    parameter_list=[ \
        ['p1', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        ['p2', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        ['p3', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        ['p4', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        ['p5', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        ['p6', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        ['p7', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        ['p8', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        ['p9', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        ['p10', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        ['p11', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        ['p12', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        ['p13', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        ['p14', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        ['p15', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        ['p16', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        ['p17', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        ['Cf_init', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        ['Cw_init', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        ['Cr_init', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        ['Clab_init', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        ['Clit_init', 'scipy.stats.uniform( 0.01, 0.99 )'], \
        ['Csom_init', 'scipy.stats.uniform( 0.01, 0.99 )'] \
        ]
    parameters = ['p1','p2','p3','p4','p5','p6','p7','p8','p9', \
            'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', \
            'Cf_init', 'Cw_init', 'Cr_init', 'Clab_init', 'Clit_init', \
            'Csom_init' ]
    cal_dalec.prior_distributions ( parameter_list, parameters )
    Z = cal_dalec.ProposeStartingMatrix ( 150 )
    (Z_out, accept_rate) = cal_dalec.demc_zs ( Z )
    lo_val = np.array( [ 1e-006, 0.2, 0.01, 0.01, 0.0001, 1e-006, \
        0.0001, 1e-005, 1e-006, 0.05, 5, 2e+002, 8, 0.1, 0.0001, \
        0.01, 1e+002, 0, 0, 0, 0, 0, 0 ] )
    hi_val = np.array( [ 0.01, 0.7, 0.5, 0.5, 0.1, 0.01, \
        0.01, 0.1, 0.01, 0.2, 20, 4e+002, 15, 0.7, 0.1, 0.5, \
        5e+002, 4e+002, 2.5e+004, 3e+002, 2e+002, 2e+002, 4e+004 ] )
    zt = Z_out*0.0
    for p in np.arange ( Z_out.shape[0] ):
        zt [ p, :] = Z_out[ p, : ]*(hi_val[p] - lo_val[p]) + lo_val[p]
    np.savez("mcmc_results.npz", Z_out=Z_out, zt=zt )