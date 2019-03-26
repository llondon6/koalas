'''
Module written by "mmrdns_write_python_freq_fit_eqns.m". Here be fitting functions for QNM frequencies for gravitational perturbations of kerr (spin weight -2).
'''

# Import useful things
from numpy import log,exp

# Domain map jf --> kappa( x =def [jf,l,m] ). 
# NOTE that the function below MUST be consistent with the domain_map of "fit_qnm_Mw.m"
kappa = lambda x: ( log(2.0-x[0])/log(3.0) )**( 1 / (2.0+x[1]-abs(x[2])) )

# Fit Equations for QNM complex frequencies (CW). Note that while these are the zero-damped modes in the extremal kerr limit, these are the effectively unique solutions throughout non-extremal kerr.
CW = {
    (2,2,0) : lambda jf: 1.0 + kappa([jf,2,2])*(  1.557847*exp(2.903124*1j) + 1.95097051*exp(5.920970*1j)*kappa([jf,2,2]) + 2.09971716*exp(2.760585*1j)*(kappa([jf,2,2])**2) + 1.41094660*exp(5.914340*1j)*(kappa([jf,2,2])**3) + 0.41063923*exp(2.795235*1j)*(kappa([jf,2,2])**4)  ),
    (2,-2,0): lambda jf: -CW[(2,2,0)](jf).conj(),
    (2,2,1) : lambda jf: 1.0 + kappa([jf,2,2])*(  1.870939*exp(2.511247*1j) + 2.71924916*exp(5.424999*1j)*kappa([jf,2,2]) + 3.05648030*exp(2.285698*1j)*(kappa([jf,2,2])**2) + 2.05309677*exp(5.486202*1j)*(kappa([jf,2,2])**3) + 0.59549897*exp(2.422525*1j)*(kappa([jf,2,2])**4)  ),
    (2,-2,1): lambda jf: -CW[(2,2,1)](jf).conj(),
    (3,2,0) : lambda jf:   1.022464*exp(0.004870*1j) + 0.24731213*exp(0.665292*1j)*kappa([jf,3,2]) + 1.70468239*exp(3.138283*1j)*(kappa([jf,3,2])**2) + 0.94604882*exp(0.163247*1j)*(kappa([jf,3,2])**3) + 1.53189884*exp(5.703573*1j)*(kappa([jf,3,2])**4) + 2.28052668*exp(2.685231*1j)*(kappa([jf,3,2])**5) + 0.92150314*exp(5.841704*1j)*(kappa([jf,3,2])**6)  ,
    (3,-2,0): lambda jf: -CW[(3,2,0)](jf).conj(),
    (4,4,0) : lambda jf: 2.0 + kappa([jf,4,4])*(  2.658908*exp(3.002787*1j) + 2.97825567*exp(6.050955*1j)*kappa([jf,4,4]) + 3.21842350*exp(2.877514*1j)*(kappa([jf,4,4])**2) + 2.12764967*exp(5.989669*1j)*(kappa([jf,4,4])**3) + 0.60338186*exp(2.830031*1j)*(kappa([jf,4,4])**4)  ),
    (4,-4,0): lambda jf: -CW[(4,4,0)](jf).conj(),
    (2,1,0) : lambda jf:   0.589113*exp(0.043525*1j) + 0.18896353*exp(2.289868*1j)*kappa([jf,2,1]) + 1.15012965*exp(5.810057*1j)*(kappa([jf,2,1])**2) + 6.04585476*exp(2.741967*1j)*(kappa([jf,2,1])**3) + 11.12627777*exp(5.844130*1j)*(kappa([jf,2,1])**4) + 9.34711461*exp(2.669372*1j)*(kappa([jf,2,1])**5) + 3.03838318*exp(5.791518*1j)*(kappa([jf,2,1])**6)  ,
    (2,-1,0): lambda jf: -CW[(2,1,0)](jf).conj(),
    (3,3,0) : lambda jf: 1.5 + kappa([jf,3,3])*(  2.095657*exp(2.964973*1j) + 2.46964352*exp(5.996734*1j)*kappa([jf,3,3]) + 2.66552551*exp(2.817591*1j)*(kappa([jf,3,3])**2) + 1.75836443*exp(5.932693*1j)*(kappa([jf,3,3])**3) + 0.49905688*exp(2.781658*1j)*(kappa([jf,3,3])**4)  ),
    (3,-3,0): lambda jf: -CW[(3,3,0)](jf).conj(),
    (3,3,1) : lambda jf: 1.5 + kappa([jf,3,3])*(  2.339070*exp(2.649692*1j) + 3.13988786*exp(5.552467*1j)*kappa([jf,3,3]) + 3.59156756*exp(2.347192*1j)*(kappa([jf,3,3])**2) + 2.44895997*exp(5.443504*1j)*(kappa([jf,3,3])**3) + 0.70040804*exp(2.283046*1j)*(kappa([jf,3,3])**4)  ),
    (3,-3,1): lambda jf: -CW[(3,3,1)](jf).conj(),
    (4,3,0) : lambda jf: 1.5 + kappa([jf,4,3])*(  0.205046*exp(0.595328*1j) + 3.10333396*exp(3.016200*1j)*kappa([jf,4,3]) + 4.23612166*exp(6.038842*1j)*(kappa([jf,4,3])**2) + 3.02890198*exp(2.826239*1j)*(kappa([jf,4,3])**3) + 0.90843949*exp(5.915164*1j)*(kappa([jf,4,3])**4)  ),
    (4,-3,0): lambda jf: -CW[(4,3,0)](jf).conj(),
    (5,5,0) : lambda jf: 2.5 + kappa([jf,5,5])*(  3.240455*exp(3.027869*1j) + 3.49056455*exp(6.088814*1j)*kappa([jf,5,5]) + 3.74704093*exp(2.921153*1j)*(kappa([jf,5,5])**2) + 2.47252790*exp(6.036510*1j)*(kappa([jf,5,5])**3) + 0.69936568*exp(2.876564*1j)*(kappa([jf,5,5])**4)  ),
    (5,-5,0): lambda jf: -CW[(5,5,0)](jf).conj()
}

# Cleaning up
# del log,exp
