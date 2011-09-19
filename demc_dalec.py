#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import scipy.stats

from demc_class import DEMC_sampler
# Actually, the next class maybe ought to be imported in its own method
# rather than for the whole script here... 
from DALEC import dale

"""
The DEMC class modified for DALEC fun'n'games. The major changes are that
it reads the data (drivers and observations), and defines the prior and
likelihood functions. Everything else is from DEMC_sampler (demc_class.py).
"""
class DEMC_spitfire ( DEMC_sampler ):
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
                n_thin=5, n_burnin=200, eps_mult=0.1, eps_add=0, \
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
        self.nee_obs = np.loadtxt ( observations, delimiter="," )
        self.missing_obs = self.nee_obs[ :, 1] < -9998
        self.x_init = np.array ( [ 0.513303, 4891.44, 134.927, 82.27539, \
                    74.74379, 12526.28 ] )

    def calc_likelihood ( self, theta ):
        """
        This calculates the likelihood function. It first transforms the
        theta vector into a parameters per grid cell that are then passed on to
        the model object.
        """
        # Load the parameters into a dictionary indexed by grid_ids
        parameters = self.load_parameters ( theta )
        # Parameter have been loaded nto parameter_list for all grid cells
        # All that needs doing now is to run the model forward for each
        # grid cell and calculate the likelihood function
        retval = dalec ( self.x_init, parameters, self.meteo_data )
        model_nee =  retval[ :, -4 ]
        pt = 0.
        delta = (self.obs_nee[~self.missing] - model_nee[~self.missing] )**2
        p = 0.5*np.sum ( delta ) / (self.flux_unc**2 )
        p -= np.sum ( np.log ( 2*np.pi*self.flux_unc ))
        return p

    def fitness ( self, theta ):
        """
        The new posterior probability in log. Convenience, really
        """
        
        prior = self.prior_probabilities ( theta )
        if np.isneginf( prior ):
            return np.log(1.0E-300)
        else:
            return self.calc_likelihood ( theta ) + prior
                
if __name__ == "__main__":
    # Need to define priors here...