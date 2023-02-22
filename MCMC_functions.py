'''Auxilliary functions for running the MCMC code'''
import numpy as np
import pymc as pm
import aesara.tensor as at
import aesara.tensor.extra_ops as ate
from astropy import units as u
from astropy import cosmology as cosmo
from astropy.cosmology import Planck18
from pylab import *
import warnings
from scipy import integrate



def P_m(m, alpha=-2.35, mmin=5):
    '''Converts X~Uniform[0,1] to bounded Power law
        m: [0,1] random value to distribute
        alpha: slope of exponential
        mmin: minimum mass value, >0 (needed to prevent diverging at 0)'''    
    ma = mmin**alpha
    return (ma + (-ma)*m)**(1/alpha)



def generate_masses(N_samples=int(1E4), f=0.3, m_b=35,
                    sigma=5, alpha=-2.35, mmin=5):
    '''Generate masses based on mass function for black holes.
       Distribution follows power law with gaussian bump
       N_samples: number of points to generate
       f: fraction of points that are within the gaussian bump
       m_b: mean of gaussian bump
       sigma: std. dev. of gaussian bump
       alpha: slope of exponential
       mmin: lower bound of exponential'''
    
    mass_array = np.zeros(N_samples)
    in_bump = np.random.rand(N_samples) < f #pick which distribution each point goes in

    for i, point in enumerate(in_bump):
        if point: #if point is going into the bump distribution
            # random seed
            mass_array[i] = np.random.normal(m_b, sigma)

        else: # going into exponential instead
            mass_array[i] = P_m(np.random.rand(), alpha=alpha, mmin=mmin)
    return mass_array



def generate_dLs(N_samples=int(1E4), R=Planck18.luminosity_distance(5)):
    '''Sample from uniform sphere
       N_samples: number of points to generate
       R: radius of sphere in which to populate (Mpc)'''
    return np.cbrt(np.random.rand(N_samples))*R.value



def dLs_to_zs(dLs, cosmology=Planck18):
    '''returns array of zs for corresponding dLs
       dLs: luminosity distance points
       cosmology: model in which to base conversion'''
    return np.array([cosmo.z_at_value(cosmology.luminosity_distance, x*u.Mpc).value for x in dLs])







def at_interp(x, xs, ys):
    '''Custom linear interpolator'''
    x  = at.as_tensor(x)
    xs = at.as_tensor(xs)
    ys = at.as_tensor(ys)

    n = xs.shape[0]
    
    ind = ate.searchsorted(xs, x)
    ind = at.where(ind >= n, n-1, ind)
    ind = at.where(ind < 0, 0, ind)
    r = (x - xs[ind-1])/(xs[ind]-xs[ind-1]) 
    return r*ys[ind] + (1-r)*ys[ind-1]

def Ez(z, Om, w):
    '''Integrand in d_L z relation'''
    z = at.as_tensor(z)

    opz = 1 + z
    
    return at.sqrt(Om*opz*opz*opz + (1-Om)*opz**(3*(1+w)))

def dCs(zs, Om, w):
    '''Integrating over range of z'''
    dz = zs[1:] - zs[:-1]
    fz = 1/Ez(zs, Om, w)
    I = 0.5*dz*(fz[:-1] + fz[1:]) #trapazoidally integrating
    return at.concatenate([at.as_tensor([0.0]), at.cumsum(I)])

def dLs(zs, dCs):
    '''Combining constants and integral to get d_L'''
    return dCs*(1+zs)

def make_model(ms_obs, sigma_ms_obs, dls, zmin=0, zmax=100):
    '''Make PyMC4 MCMC model using custom helper functions'''
    zinterp = expm1(linspace(log(1+zmin), log(1+zmax), 1024))

    with pm.Model() as model:
        w = pm.Normal('w', mu=-1, sigma=0.25) 

        Om = pm.Bound('Om', pm.Normal.dist(mu=0.3, sigma=0.15), lower=0, upper=1)

        h = pm.Bound('h', pm.Lognormal.dist(mu=log(0.7), sigma=0.2), lower=0.35, upper=1.4)
#         h = pm.Lognormal('h', mu=log(0.7), sigma=0.2)
        Ode = pm.Deterministic('Ode', 1-Om)
        om = pm.Deterministic('om', Om*h*h)
        ode = pm.Deterministic('ode', Ode*h*h)

        dH= pm.Deterministic('dH', 2.99792/h)
        m0 = pm.Lognormal('m0', mu=log(35), sigma=0.5)

        dCinterp = dH*dCs(zinterp, Om, w)
        dLinterp = dLs(zinterp, dCinterp)
        
        zs = pm.Deterministic('zs', at_interp(dls, dLinterp, zinterp))

        pm.Normal('m_likelihood', mu=m0*(1+zs), sigma=sigma_ms_obs, observed=ms_obs)
    return model

def find_argmax_gridsearch(xs, fxs):
    '''Custom max finder for arrays'''
    imax = np.argmax(fxs)

    if imax == 0 or imax == len(xs)-1:
        warnings.warn('max occurs at grid boundary')
        return xs[imax]

    x0, x1, x2 = xs[imax-1], xs[imax], xs[imax+1]
    f0, f1, f2 = fxs[imax-1], fxs[imax], fxs[imax+1]

    dx01 = x0-x1
    dx12 = x1-x2
    dx20 = x2-x0

    sx01 = x0+x1
    sx12 = x1+x2
    sx20 = x2+x0

    xmax = (f2*dx01*sx01 + f0*dx12*sx12 + f1*dx20*sx20)/\
            (2*(f2*dx01 + f0*dx12 + f1*dx20))

    return xmax




def H(z, Ode, Om):
    epsilon = lambda x: (Om*(1+x)**3 + Ode)**-0.5
    return (1+z)*integrate.quad(epsilon, 0, z)[0]