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
