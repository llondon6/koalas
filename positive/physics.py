#
from positive import *

#
def pnw0(m1,m2,D=10.0):
    # https://arxiv.org/pdf/1310.1528v4.pdf
    # Equation 228
    # 2nd Reference: arxiv:0710.0614v1
    # NOTE: this outputs orbital frequency
    from numpy import sqrt,zeros,pi,array,sum
    #
    G = 1.0
    c = 1.0
    r = float(D)
    M = float( m1+m2 )
    v = m1*m2/( M**2 )
    gamma = G*M/(r*c*c)     # Eqn. 225
    #
    trm = zeros((4,))
    #
    trm[0] = 1.0
    trm[1] = v - 3.0
    trm[2] = 6 + v*41.0/4.0 + v*v
    trm[3] = -10.0 + v*( -75707.0/840.0 + pi*pi*41.0/64.0 ) + 19.0*0.5*v*v + v*v*v
    #
    w0 = sqrt( (G*M/(r*r*r)) * sum( array([ term*(gamma**k) for k,term in enumerate(trm) ]) ) )

    #
    return w0


#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#
# Here are some phenomenological fits used in PhenomD                               #
#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#

# Formula to predict the final spin. Equation 3.6 arXiv:1508.07250
# s is defined around Equation 3.6.
''' Copied from LALSimulation Version '''
def FinalSpin0815_s(eta,s):
    eta = round(eta,8)
    eta2 = eta*eta
    eta3 = eta2*eta
    eta4 = eta3*eta
    s2 = s*s
    s3 = s2*s
    s4 = s3*s
    return 3.4641016151377544*eta - 4.399247300629289*eta2 +\
    9.397292189321194*eta3 - 13.180949901606242*eta4 +\
    (1 - 0.0850917821418767*eta - 5.837029316602263*eta2)*s +\
    (0.1014665242971878*eta - 2.0967746996832157*eta2)*s2 +\
    (-1.3546806617824356*eta + 4.108962025369336*eta2)*s3 +\
    (-0.8676969352555539*eta + 2.064046835273906*eta2)*s4

#Wrapper function for FinalSpin0815_s.
''' Copied from LALSimulation Version '''
def FinalSpin0815(eta,chi1,chi2):
    from numpy import sqrt
    eta = round(eta,8)
    if eta>0.25:
        error('symmetric mass ratio greater than 0.25 input')
    # Convention m1 >= m2
    Seta = sqrt(abs(1.0 - 4.0*float(eta)))
    m1 = 0.5 * (1.0 + Seta)
    m2 = 0.5 * (1.0 - Seta)
    m1s = m1*m1
    m2s = m2*m2
    # s defined around Equation 3.6 arXiv:1508.07250
    s = (m1s * chi1 + m2s * chi2)
    return FinalSpin0815_s(eta, s)

# Formula to predict the total radiated energy. Equation 3.7 and 3.8 arXiv:1508.07250
# Input parameter s defined around Equation 3.7 and 3.8.
def EradRational0815_s(eta,s):
    eta = round(eta,8)
    eta2 = eta*eta
    eta3 = eta2*eta
    eta4 = eta3*eta
    return ((0.055974469826360077*eta + 0.5809510763115132*eta2 - 0.9606726679372312*eta3 + 3.352411249771192*eta4)*\
    (1. + (-0.0030302335878845507 - 2.0066110851351073*eta + 7.7050567802399215*eta2)*s))/(1. + (-0.6714403054720589 \
    - 1.4756929437702908*eta + 7.304676214885011*eta2)*s)


# Wrapper function for EradRational0815_s.
def EradRational0815(eta, chi1, chi2):
    from numpy import sqrt,round
    eta = round(eta,8)
    if eta>0.25:
        error('symmetric mass ratio greater than 0.25 input')
    # Convention m1 >= m2
    Seta = sqrt(1.0 - 4.0*eta)
    m1 = 0.5 * (1.0 + Seta)
    m2 = 0.5 * (1.0 - Seta)
    m1s = m1*m1
    m2s = m2*m2
    # arXiv:1508.07250
    s = (m1s * chi1 + m2s * chi2) / (m1s + m2s)
    return EradRational0815_s(eta,s)


#
def rISCO_14067295(a):
    """
    Calculate the ISCO radius of a Kerr BH as a function of the Kerr parameter using eqns. 2.5 and 2.8 from Ori and Thorne, Phys Rev D 62, 24022 (2000)

    Parameters
    ----------
    a : Kerr parameter

    Returns
    -------
    ISCO radius
    """

    import numpy as np
    a = np.array(a)

    # Ref. Eq. (2.5) of Ori, Thorne Phys Rev D 62 124022 (2000)
    z1 = 1.+(1.-a**2.)**(1./3)*((1.+a)**(1./3) + (1.-a)**(1./3))
    z2 = np.sqrt(3.*a**2 + z1**2)
    a_sign = np.sign(a)
    return 3+z2 - np.sqrt((3.-z1)*(3.+z1+2.*z2))*a_sign

# https://arxiv.org/pdf/1406.7295.pdf
def Mf14067295( m1,m2,chi1,chi2,chif=None ):

    import numpy as np

    if np.any(abs(chi1>1)):
      raise ValueError("chi1 has to be in [-1, 1]")
    if np.any(abs(chi2>1)):
      raise ValueError("chi2 has to be in [-1, 1]")


    # Swapping inputs to conform to fit conventions
    # NOTE: See page 2 of https://arxiv.org/pdf/1406.7295.pdf
    if m1>m2:
        #
        m1_,m2_ = m1,m2
        chi1_,chi2_ = chi1,chi2
        #
        m1,m2 = m2_,m1_
        chi1,chi2 = chi2_,chi1_

    # binary parameters
    m = m1+m2
    q = m1/m2
    eta = q/(1.+q)**2.
    delta_m = (m1-m2)/m
    S1 = chi1*m1**2 # spin angular momentum 1
    S2 = chi2*m2**2 # spin angular momentum 2
    S = (S1+S2)/m**2 # symmetric spin (dimensionless -- called \tilde{S} in the paper)
    Delta = (S2/m2-S1/m1)/m # antisymmetric spin (dimensionless -- called tilde{Delta} in the paper

    #
    if chif is None:
        chif = jf14067295(m1, m2, chi1, chi2)
    r_isco = rISCO_14067295(chif)

    # fitting coefficients - Table XI of Healy et al Phys Rev D 90, 104004 (2014)
    # [fourth order fits]
    M0  = 0.951507
    K1  = -0.051379
    K2a = -0.004804
    K2b = -0.054522
    K2c = -0.000022
    K2d = 1.995246
    K3a = 0.007064
    K3b = -0.017599
    K3c = -0.119175
    K3d = 0.025000
    K4a = -0.068981
    K4b = -0.011383
    K4c = -0.002284
    K4d = -0.165658
    K4e = 0.019403
    K4f = 2.980990
    K4g = 0.020250
    K4h = -0.004091
    K4i = 0.078441

    # binding energy at ISCO -- Eq.(2.7) of Ori, Thorne Phys Rev D 62 124022 (2000)
    E_isco = (1. - 2./r_isco + chif/r_isco**1.5)/np.sqrt(1. - 3./r_isco + 2.*chif/r_isco**1.5)

    # final mass -- Eq. (14) of Healy et al Phys Rev D 90, 104004 (2014)
    mf = (4.*eta)**2*(M0 + K1*S + K2a*Delta*delta_m + K2b*S**2 + K2c*Delta**2 + K2d*delta_m**2 \
        + K3a*Delta*S*delta_m + K3b*S*Delta**2 + K3c*S**3 + K3d*S*delta_m**2 \
        + K4a*Delta*S**2*delta_m + K4b*Delta**3*delta_m + K4c*Delta**4 + K4d*S**4 \
        + K4e*Delta**2*S**2 + K4f*delta_m**4 + K4g*Delta*delta_m**3 + K4h*Delta**2*delta_m**2 \
        + K4i*S**2*delta_m**2) + (1+eta*(E_isco+11.))*delta_m**6.

    return mf*m

#
def jf14067295_diff(a_f,eta,delta_m,S,Delta):
    """ Internal function: the final spin is determined by minimizing this function """

    #
    import numpy as np

    # calculate ISCO radius
    r_isco = rISCO_14067295(a_f)

    # angular momentum at ISCO -- Eq.(2.8) of Ori, Thorne Phys Rev D 62 124022 (2000)
    J_isco = (3*np.sqrt(r_isco)-2*a_f)*2./np.sqrt(3*r_isco)

    # fitting coefficients - Table XI of Healy et al Phys Rev D 90, 104004 (2014)
    # [fourth order fits]
    L0  = 0.686710
    L1  = 0.613247
    L2a = -0.145427
    L2b = -0.115689
    L2c = -0.005254
    L2d = 0.801838
    L3a = -0.073839
    L3b = 0.004759
    L3c = -0.078377
    L3d = 1.585809
    L4a = -0.003050
    L4b = -0.002968
    L4c = 0.004364
    L4d = -0.047204
    L4e = -0.053099
    L4f = 0.953458
    L4g = -0.067998
    L4h = 0.001629
    L4i = -0.066693

    a_f_new = (4.*eta)**2.*(L0  +  L1*S +  L2a*Delta*delta_m + L2b*S**2. + L2c*Delta**2 \
        + L2d*delta_m**2. + L3a*Delta*S*delta_m + L3b*S*Delta**2. + L3c*S**3. \
        + L3d*S*delta_m**2. + L4a*Delta*S**2*delta_m + L4b*Delta**3.*delta_m \
        + L4c*Delta**4. + L4d*S**4. + L4e*Delta**2.*S**2. + L4f*delta_m**4 + L4g*Delta*delta_m**3. \
        + L4h*Delta**2.*delta_m**2. + L4i*S**2.*delta_m**2.) \
        + S*(1. + 8.*eta)*delta_m**4. + eta*J_isco*delta_m**6.

    daf = a_f-a_f_new
    return daf*daf

#
def jf14067295(m1, m2, chi1, chi2):
    """
    Calculate the spin of the final BH resulting from the merger of two black holes with non-precessing spins using fit from Healy et al Phys Rev D 90, 104004 (2014)

    Parameters
    ----------
    m1, m2 : component masses
    chi1, chi2 : dimensionless spins of two BHs

    Returns
    -------
    dimensionless final spin, chif
    """
    import numpy as np
    import scipy.optimize as so

    if np.any(abs(chi1>1)):
      raise ValueError("chi1 has to be in [-1, 1]")
    if np.any(abs(chi2>1)):
      raise ValueError("chi2 has to be in [-1, 1]")

    # Vectorize the function if arrays are provided as input
    if np.size(m1) * np.size(m2) * np.size(chi1) * np.size(chi2) > 1:
        return np.vectorize(bbh_final_spin_non_precessing_Healyetal)(m1, m2, chi1, chi2)

    # binary parameters
    m = m1+m2
    q = m1/m2
    eta = q/(1.+q)**2.
    delta_m = (m1-m2)/m

    S1 = chi1*m1**2 # spin angular momentum 1
    S2 = chi2*m2**2 # spin angular momentum 2
    S = (S1+S2)/m**2 # symmetric spin (dimensionless -- called \tilde{S} in the paper)
    Delta = (S2/m2-S1/m1)/m # antisymmetric spin (dimensionless -- called tilde{Delta} in the paper

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # compute the final spin
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    x, cov_x = so.leastsq(jf14067295_diff, 0., args=(eta, delta_m, S, Delta))
    chif = x[0]

    return chif



#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#
# Post-Newtonian methods
#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#

#
def mishra( f, m1,m2, X1,X2, lm,    # Intrensic parameters and l,m
            lnhat   = None,         # unit direction of orbital angular momentum
            vmax    = None,         # maximum power (integer) allwed for V parameter
            verbose = False ):      # toggle for letting the people know
    '''
    PN formulas from "Ready-to-use post-Newtonian gravitational waveforms for binary black holes with non-precessing spins: An update"
    *   https://arxiv.org/pdf/1601.05588.pdf
    *   https://arxiv.org/pdf/0810.5336.pdf
    '''

    # Import usefuls
    from numpy import pi,array,dot,sqrt,log,inf,ones

    # Handle zero values
    f[f==0] = 1e-5 * min(abs(f))

    #
    l,m = lm

    #
    M = m1+m2

    # Normalied mass difference
    delta = (m1-m2)/(m1+m2)

    # Symmetric mass ratio
    eta = float(m1)*m2/(M*M)

    # Frequency parameter (note that this is the same as the paper's \nu)
    V = lambda m: pow(2.0*pi*M*f/m,1.0/3)

    # Here we handle the vmax input using a lambda function
    if vmax is None: vmax = inf
    e = ones(12)        # This identifier will be used to turn off terms
    e[(vmax+1):] = 0    # NOTE that e will effectively be indexed starting
                        # from 1 not 0, so it must have more elements than needed.


    # Handle default behavior for
    lnhat = array([0,0,1]) if lnhat is None else lnhat

    # Symmetric and Anti-symmettric spins
    Xs = 0.5 * ( X1+X2 )
    Xa = 0.5 * ( X1-X2 )

    # Dictionary for FD multipole terms (eq. 12)
    H = {}

    #
    H[2,2] = lambda v: -1 + v**2  * e[2] *  ( (323.0/224.0)-(eta*451.0/168.0) ) \
                          + v**3  * e[3] *  ( -(27.0/8)*delta*dot(Xa,lnhat) + dot(Xs,lnhat)*((-27.0/8)+(eta*11.0/6)) ) \
                          + v**4  * e[4] *  ( (27312085.0/8128512)+(eta*1975055.0/338688) - (105271.0/24192)*eta*eta + dot(Xa,lnhat)**2 * ((113.0/32)-eta*14) + delta*(113.0/16)*dot(Xa,lnhat)*dot(Xs,lnhat) + dot(Xs,lnhat)**2 * ((113.0/32) - (eta/8)) )

    #
    H[2,1] = lambda v: -(sqrt(2)/3) * ( v    * delta \
                                      - v**2 * e[2] * 1.5*( dot(Xa,lnhat)+delta*dot(Xs,lnhat) ) \
                                      + v**3 * e[3] * delta*( (335.0/672)+(eta*117.0/56) ) \
                                      + v**4 * e[4] * ( dot(Xa,lnhat)*(4771.0/1344 - eta*11941.0/336) + delta*dot(Xs,lnhat)*(4771.0/1344 - eta*2549.0/336) + delta*(-1j*0.5-pi-2*1j*log(2)) ) \
                                      )

    #
    H[3,3] = lambda v: -0.75*sqrt(5.0/7) \
                        *(v           * delta \
                        + v**3 * e[3] * delta * ( -(1945.0/672) + eta*(27.0/8) )\
                        + v**4 * e[4] * ( dot(Xa,lnhat)*( (161.0/24) - eta*(85.0/3) ) + delta*dot(Xs,lnhat)*( (161.0/24) - eta*(17.0/3) ) + delta*(-1j*21.0/5 + pi + 6j*log(3.0/2)) ) \
                        )

    #
    H[3,2] = lambda v: -(1.0/3)*sqrt(5.0/7) * (\
                        v**2   * e[2] * (1-3*eta) \
                        + v**3 * e[3] * 4*eta*dot(Xs,lnhat) \
                        + v**4 * e[4] * (-10471.0/10080 + eta*12325.0/2016 - eta*eta*589.0/72) \
                        )

    #
    H[4,4] = lambda v: -(4.0/9)*sqrt(10.0/7) \
                        * ( v**2 * e[2] * (1-3*eta) \
                        +   v**4 * e[4] * (-158383.0/36960 + eta*128221.0/7392 - eta*eta*1063.0/88) \
                        )

    #
    H[4,3] = lambda v: -(3.0/4)*sqrt(3.0/35) * (
                        v**3 * e[3] * delta*(1-2*eta) \
                        + v**4 * e[4] * (5.0/2)*eta*( dot(Xa,lnhat) - delta*dot(Xs,lnhat) )\
                        )

    #
    hlm_amp = M*M*pi * sqrt(eta*2.0/3)*(V(m)**-3.5) * H[l,m]( V(m) )

    #
    return abs(hlm_amp)


# leading order amplitudes in freq fd strain via spa
def lamp_spa(f,eta,lm=(2,2)):
    # freq domain amplitude from leading order in f SPA
    # made using ll-LeadingOrderAmplitudes.nb in PhenomHM repo
    from numpy import pi,sqrt

    #
    warning('This function has a bug related the a MMA bug of unknown origin -- ampliutdes are off by order 1 factors!!')

    # Handle zero values
    f[f==0] = 1e-5 * min(abs(f))

    #
    hf = {}
    #
    hf[2,2] = (sqrt(0.6666666666666666)*sqrt(eta))/(f**1.1666666666666667*pi**0.16666666666666666)
    #
    hf[2,1] = (sqrt(0.6666666666666666)*sqrt(eta - 4*eta**2)*pi**0.16666666666666666)/(3.*f**0.8333333333333334)
    #
    hf[3,3] = (3*sqrt(0.7142857142857143)*sqrt(eta - 4*eta**2)*pi**0.16666666666666666)/(4.*f**0.8333333333333334)
    #
    hf[3,2] = (sqrt(0.47619047619047616)*eta*sqrt(pi))/(3.*sqrt(eta*f)) - (sqrt(0.47619047619047616)*eta**2*sqrt(pi))/sqrt(eta*f)
    #
    hf[3,1] = (sqrt(eta - 4*eta**2)*pi**0.16666666666666666)/(12.*sqrt(21)*f**0.8333333333333334)
    #
    hf[4,4] = (8*sqrt(0.47619047619047616)*eta*sqrt(pi))/(9.*sqrt(eta*f)) - (8*sqrt(0.47619047619047616)*eta**2*sqrt(pi))/(3.*sqrt(eta*f))
    #
    hf[4,3] = (3*sqrt(0.08571428571428572)*sqrt(eta - 4*eta**2)*pi**0.8333333333333334)/(4.*f**0.16666666666666666) - (3*sqrt(0.08571428571428572)*eta*sqrt(eta - 4*eta**2)*pi**0.8333333333333334)/(2.*f**0.16666666666666666)
    #
    hf[4,2] = (sqrt(3.3333333333333335)*eta*sqrt(pi))/(63.*sqrt(eta*f)) - (sqrt(3.3333333333333335)*eta**2*sqrt(pi))/(21.*sqrt(eta*f))
    #
    hf[4,1] = (sqrt(eta - 4*eta**2)*pi**0.8333333333333334)/(84.*sqrt(15)*f**0.16666666666666666) - (eta*sqrt(eta - 4*eta**2)*pi**0.8333333333333334)/(42.*sqrt(15)*f**0.16666666666666666)
    #
    hf[5,5] = (625*sqrt(eta - 4*eta**2)*pi**0.8333333333333334)/(288.*sqrt(11)*f**0.16666666666666666) - (625*eta*sqrt(eta - 4*eta**2)*pi**0.8333333333333334)/(144.*sqrt(11)*f**0.16666666666666666)
    #
    hf[6,6] = 3.6
    #
    return hf[lm]
