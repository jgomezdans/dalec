import numpy as np
import matplotlib.pyplot as plt

def PreparePlotsParams ( small=True):
    import pylab
    import numpy
    fig_size = [8.3, 11.7]
    params = {'backend': 'ps',
    'ps.papersize': 'a4',
    'axes.formatter.limits' : [-2, 2], #No large numbers with loads of 0s
    'figure.subplot.left'  : 0.05,  # the left side of the subplots of the figure
    'figure.subplot.right' : 0.95,    # the right side of the subplots of the figure
    'figure.subplot.bottom' : 0.05,   # the bottom of the subplots of the figure
    'figure.subplot.top' : 0.95,      # the top of the subplots of the figure
    'figure.subplot.wspace' : 0.32,   # the amount of width reserved for blank space between subplots
    'figure.subplot.hspace' : 0.32,   # the amount of height reserved for white space between subplots
    'text.usetex': True ,
    'figure.figsize': fig_size}
    pylab.rcParams.update(params)
    if small:
        pylab.rcParams.update ({'axes.labelsize': 10,
        'text.fontsize': 6,
        'legend.fontsize': 8,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6})
    else:
        pylab.rcParams.update ({'axes.labelsize': 12,
        'text.fontsize': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12})
        

def summarise_posterior ( mcmc_result, parameter_names, true_vals ):
    """
    This function summarises the posterior distribution in terms of
    mean, standard deviation, and 95CI interval.

    Returns a string with an RST table.
    """
    from scipy.stats import scoreatpercentile
    summary = ""
    #summary = summary + "+--------------------+--------------------+" + \
        #"--------------------" + \
        #"+--------------------+--------------------+\n"
    #summary = summary + "+     Parameter      +     Mean           +" + \
        #"     Std Dev        " +\
        #"+     2.5% CI        +    97.5% CI        +\n"
    #summary = summary + "+====================+====================+" + \
        #"====================" + \
        #"+====================+====================+\n"
    nsamples = mcmc_result.shape[1]/2
    
    for ( i, param ) in enumerate( parameter_names ):
        posterior_dist = mcmc_result[i, :nsamples]
        summary = summary + "%s & %20.2G & %20.2G & %20.2G & %20.2G & %20.2G & %s\\\\\n" % ( \
                param.center(20), \
                posterior_dist.mean(), \
                posterior_dist.std(), \
                scoreatpercentile ( posterior_dist, 2.5 ), \
                scoreatpercentile ( posterior_dist, 97.5 ), 
                true_vals[i], scoreatpercentile ( posterior_dist, 2.5 ) <= true_vals[i] <= scoreatpercentile ( posterior_dist, 97.5 ), 
                )
        #summary = summary + "+--------------------+--------------------+" + \
                #"--------------------" + \
                #"+--------------------+--------------------+\n"

    return summary

def trace_plots ( mcmc_result, parameter_names ):
    plt.figure ()
    for ( i, param ) in enumerate ( parameter_names ):
        plt.subplot ( 5, 5, i+1 )
        plt.plot ( mcmc_result[ i, :][::-1], '-k', lw=0.5)
        plt.title ( param )

def parameter_histograms ( mcmc_results, parameter_names, lo_val, hi_val,       
            x_init=None, true_vals=None ):
    plt.figure ()
    nsamples = mcmc_results.shape[1]/2
    for ( i, param ) in enumerate ( parameter_names ):
        plt.subplot ( 5, 5, i+1 )
        plt.hist ( mcmc_results[ i, :nsamples], bins=20, histtype="stepfilled", ec='r',fc='0.8' )
        if true_vals is not None:
            plt.axvline ( true_vals[i], ymin=0, color='orange', lw=2.2 )
        if x_init is not None:
            plt.axvline ( x_init[i], ymin=0, color='g', lw=2.2 )
            
        plt.xlim ( lo_val[i], hi_val[i] )
        plt.title ( param )
        
def forward_model ( mcmc_results, parameter_names, \
        observations="sim04_obs_l10_g6_n58.csv", \
        meteo_drivers="sim04_met_l10_g6_n58.csv"):
    from dalec import dalec
    meteo_data = np.loadtxt( meteo_drivers, delimiter="," )
    obs_nee = np.loadtxt ( observations, delimiter="," )
    missing_obs = obs_nee[ :, 1] < -9998
    nsamples = mcmc_results.shape[1]/2
    parameters = mcmc_results[:, :nsamples]
    sel = np.random.randint ( 0, nsamples-1, size=25 )
    #x_init = np.array ( [ 0.513303, 4891.44, 134.927, 82.27539, \
    #    74.74379, 12526.28 ] )
    i = 0
    model_nee = np.zeros ( ( 25, meteo_data.shape[0] ) )
    for params in sel:
        retval = dalec ( parameters[17:, params], parameters[:17, params], meteo_data, \
        -2, 1, 60, 42.2, 2.7)#self.psid, self.rtot, self.lma, self.lat, self.nit )
        
        model_nee[ i, : ] =  retval[ :, -4 ]
        i = i + 1
    #plt.plot ( meteo_data[~missing_obs,0], obs_nee[~missing_obs,1], 'ro' )
    #plt.plot 
    return (model_nee, meteo_data[:,0], obs_nee[:,1], missing_obs, meteo_data )
    
def transform_variables_lognorm ( theta, x_init ):
    retval = theta*.0
    if theta.shape[0] == 23:
        istart = 0
    else:
        istart = 1
        retval[0,:] = theta[0,:]
    for i in xrange(istart, theta.shape[0]):
        retval[i,:] = np.exp( theta[i,:] - 1. ) * x_init[i]
    return retval
        
if __name__ == "__main__":
    from dalec import dalec
    PreparePlotsParams ()
    result = np.load ( "mcmc_results.npz" )
    parameters = ['p1','p2','p3','p4','p5','p6','p7','p8','p9', \
        'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', \
        'Cf init', 'Cw init', 'Cr init', 'Clab init', 'Clit init', \
        'Csom init' ]
        
    true_vals = np.array([5.00E-04,0.45,0.4,0.4,6.00E-02,7.00E-05,\
        8.00E-03,3.00E-02,3.00E-05,7.30E-02,1.40E+01,240,9,0.48,\
        9.00E-02,0.15,300,0,5,5,100,5,9900])
        
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
        4891.44,134.927,82.27539,74.74379,12526.28, 1 ])
    results = transform_variables_lognorm ( result['Z_out'], x_init )
    print summarise_posterior ( results, parameters, true_vals )
    trace_plots ( result['Z_out'], parameters )
    parameter_histograms ( results, parameters, lo_val, hi_val, x_init, true_vals )
    (model_nee, doys, obs_nee, missing, meteo_data )= forward_model (results[:,:], \
            parameters  )
    mu_ensemble = model_nee.mean( axis=0 )
    std_ensemble = model_nee.std ( axis=0 )
    plt.figure()
    PreparePlotsParams ( small=False)
    
    #true_nee = dalec ( x_init[17:], x_init[:17], meteo_data,-2, 1, 60, 42.2, 2.7 )
    
    plt.vlines ( doys, mu_ensemble-3*std_ensemble, mu_ensemble+3*std_ensemble, \
                color="0.6" )
    plt.plot ( doys[~missing], obs_nee[~missing], '-b' )
    #plt.plot ( doys, true_nee[:,-4], '-r' )
    plt.grid (True)
    plt.figure()
    boxes=plt.boxplot(result['Z_out'][:,:20000].T, notch=1, sym="" )
    plt.axis([0, 24, 0, 2])
    plt.grid ( True)
    #plt.show()
    for i in xrange(23):
        boxes['boxes'][i].set_color( 'k' )
        boxes['boxes'][i].set_mfc( '0.8' )