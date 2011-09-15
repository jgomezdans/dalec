#!/usr/bin/env pythpn

import numpy as np

def dalec ( x, p, meteo_data, site_info ):
    """
    The DALEC ecosystem model. In glorious python
    """
    # First extract the model parameters from the ``x`` vector

    Cf = x[0]
    Cw = x[1]
    Cr = x[2]
    Clab = x[3]
    Clit = x[4]
    Csom = x[5]
    # ``d`` is the number of days, ``e`` of meteo parameters
    ( d, e ) = meteo_data.shape

    # Get the site information
    lat = site_info[0] 
    lat = lat*np.pi/180. # To radians
    nit = site_info[1]
    lma = site_info[2]
    psid = -2.0
    rtot = 1.0
    #ACM parameters
    a1 = p[10]
    a2=0.0156935
    a3=4.22273
    a4=208.868
    a5=0.0453194
    a6=0.37836
    a7=7.19298
    a8=0.011136
    a9=2.1001
    a10=0.789798

    
    # Define the output array
    # Elements (24 of 'em) are
    # 'day,' 'G,' 'Ra,' 'Af,' 'Aw,' 'Ar,' 'Atolab,' 'Afrlab,' 'Lf,' 'Lw,' 'Lr,' 
    # 'Rh1,' 'Rh2,' 'D,' 'Cf,' 'Cw,' 'Cr,' 'Clab,' 'Clit,' 'Csom,' 'NEE,' 'LAI,' 
    # 'iNEE'
    output_array = np.zeros ( (d, 24) )
    Afromlab = 0. # Needs to be initialised
    # Day loop
    for k in np.arange ( d ):
        ( projectday, mint, maxt, rad, ca, yearday ) = meteo_data[ k, : ]
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #% Step 3 - Run ACM to determine GPP %
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        trange=maxt-mint;
        LAI=max(0.1,(Cf/lma));

        gs = np.pow( np.abs(psid), a10)/((0.5*trange) + (a6*rtot))
        pp = (a1*LAI*nit*np.exp( a8*maxt )) / gs
        qq = a3 - a4
        ci = 0.5*( ca + qq - pp + np.sqrt ((( ca+qq-pp)**2) - \
                (4*((ca*qq) - (pp*a3)))))
        e0 = ( a7*LAI*LAI)/((LAI*LAI) + a9 )
        dec = -23.4*np.cos ( ( 360*(yearday + 10.)/365.)*np.pi/180.)*np.pi/180.
        mult = np.tan ( lat ) * np.tan ( dec )
        if mult >= 1:
            dayl = 24.
        elif mult <= -1:
            dayl = 0.
        else:
            dayl = 24.*np.acos ( -mult )/np.pi
        cps = (e0*rad*gs*(ca-ci))/((e0*rad)+(gs*(ca-ci)))
        G = cps * ( (a2*dayl) + a5 )
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #% Step 4 - Run DALEC model equations %
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if yearday <= 100:
            gdd = 0
            max_fol = 1
        # time switches
        multtf = 1               # defaults
        multtl = 0
        gdd = gdd + 0.5* ( maxt + mint )    #growing degreeday sum from day 100
        if gdd < p[11]:           #winter
            multtf = 1           # turnover of foliage on
            multtl = 0           # turnover of labile C off
        else:
            if max_fol == 1:      # spring
                multtl = 1
                multtf = 0
            else:                # summer
                multtl = 0
                multtf = 0
          
        if ( (Cf >= p[16]) or ( yearday >= 200 ) ):
            max_fol = 0
            multtl = 0
        
        if ( ( yearday >= 200 ) and ( mint < p[12] ) ): 
            # Drop leaves N hemisphere
            multtf = 1


        Trate=0.5*np.exp( p[9]*0.5*( maxt + mint ) )
        ralabfrom = p[14]*Clab*p[15]*multtl*Trate
        ralabto = (1. - p[13])*p[4]*Cf*p[15]*multtf*Trate
        Ra = p[1]*G + ralabfrom + ralabto
        npp = G* ( 1. - p[1] )
        Af = min ( p[17] - Cf, npp*p[2] )*multtl + Afromlab
        npp = npp - min ( p[17] - Cf, npp*p[2] )*multtl
        Ar = npp*p[3]
        Aw = ( 1. - p[3] )*npp
        Lf = p[4]*Cf*p[13]*multtf
        Lw = p[5]*Cw
        Lr = p[6]*Cr
        Rh1 = p[7]*Clit*Trate
        Rh2 = p[8]*Csom*Trate
        D = p[0]*Clit*Trate
        Atolab = ( 1.0 - p[13] ) * p[4]*Cf*( 1. - p[15] )*multtf*Trate
        Afromlab = p[14]*Clab*( 1. - p[15] )*multtf*Trate
        # Pools:
        Cf = Cf + Af - Lf - Atolab - ralabto
        Cw = Cw + Aw - Lw
        Cr = Cr + Ar - Lr                           
        Clit = Clit + Lf + Lr - Rh1 - D
        Csom = Csom+ D - Rh2 + Lw               
        Clab = Clab + Atolab - Afromlab - ralabfrom
        Rtot = Ra + Rh1 + Rh2
        NEE = Ra + Rh1 + Rh2 - G 
        output_array [ k, : ] = [ k, G, Ra, Af, Aw, Ar, Atolab, Afromlab, \
                                Lf, Lw, Lr, Rh1, Rh2, D, Cf, Cw, Cr, Clab, \
                                Clit, Csom, NEE, ralabfrom, ralabto, Rtot ]
    return output_array

        