import numpy as np
import scipy as scp
import pandas as pd
from mssm.models import *

################################## Contains simulations to simulate for GAMM & GAMMLSS models ##################################

def sim3(n,scale,c=1,family=Gaussian()):
    """
    First Simulation performed by Wood et al., (2016): 4 smooths, 1 is really zero everywhere.
    Based on the original functions of Gu & Whaba (1991).

    This is also the first simulation performed by gamSim() - except for the fact that f(x_0) can also be set to
    zero, as was done by Wood et al., (2016)

    References:

     - Gu, C. & Whaba, G., (1991). Minimizing GCV/GML scores with multiple smoothing parameters via the Newton method.
     - Wood, S. N., Pya, N., Saefken, B., (2016). Smoothing Parameter and Model Selection for General Smooth Models
     - mgcv source code: gam.sim.r

    :param scale: Standard deviation for `family='Gaussian'` else scale parameter
    :type scale: float
    :param c: Effect strength for x3 effect - 0 = No effect, 1 = Maximal effect
    :type c: float
    :param family: Distribution for response variable, must be: `Gaussian()`, `Gamma()`, or `Binomial()`. Defaults to `Gaussian()`
    :type family: Family, optional
    """
    x0 = np.random.rand(n)
    x1 = np.random.rand(n)
    x2 = np.random.rand(n)
    x3 = np.random.rand(n)

    f0 = 2* np.sin(np.pi*x0)
    f1 = np.exp(2*x1)
    f2 = 0.2*np.power(x2,11)*np.power(10*(1-x2),6)+10*np.power(10*x2,3)*np.power(1-x2,10)
    f3 = np.zeros_like(x3)

    mu = c*f0 + f1 + f2 + f3 # eta in truth for non-Gaussian

    if isinstance(family,Gaussian):
        y = scp.stats.norm.rvs(loc=mu,scale=scale,size=n)
    
    elif isinstance(family,Gamma):
        # Need to transform from mean and scale to \alpha & \beta
        # From Wood (2017), we have that
        # \phi = 1/\alpha
        # so \alpha = 1/\phi
        # From https://en.wikipedia.org/wiki/Gamma_distribution, we have that:
        # \mu = \alpha/\beta
        # \mu = 1/\phi/\beta
        # \beta = 1/\phi/\mu
        # scipy docs, say to set scale to 1/\beta.
        # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html
        mu = family.link.fi(mu)
        alpha = 1/scale
        beta = alpha/mu  
        y = scp.stats.gamma.rvs(a=alpha,scale=(1/beta),size=n)
    
    elif isinstance(family,Binomial):
        mu = family.link.fi(mu*0.1)
        y = scp.stats.binom.rvs(1, mu, size=n)

    dat = pd.DataFrame({"y":y,
                        "x0":x0,
                        "x1":x1,
                        "x2":x2,
                        "x3":x3})
    return dat


def sim4(n,scale,c=1,family=Gaussian()):
    """
    Like ``sim3``, except that a random factor is added - second simulation performed by Wood et al., (2016).

    This is also the sixth simulation performed by gamSim() - except for the fact that c is used here to scale the contribution
    of the random factor, as was also done by Wood et al., (2016)

    References:

     - Gu, C. & Whaba, G., (1991). Minimizing GCV/GML scores with multiple smoothing parameters via the Newton method.
     - Wood, S. N., Pya, N., Saefken, B., (2016). Smoothing Parameter and Model Selection for General Smooth Models
     - mgcv source code: gam.sim.r

    :param scale: Standard deviation for `family='Gaussian'` else scale parameter
    :type scale: float
    :param c: Effect strength for random effect - 0 = No effect (sd=0), 1 = Maximal effect (sd=1)
    :type c: float
    :param family: Distribution for response variable, must be: `Gaussian()`, `Gamma()`, or `Binomial()`. Defaults to `Gaussian()`
    :type family: Family, optional
    """

    x0 = np.random.rand(n)
    x1 = np.random.rand(n)
    x2 = np.random.rand(n)
    x3 = np.random.rand(n)
    x4 = np.random.randint(0,high=40,size=n)

    if c > 0:
        rind = scp.stats.norm.rvs(size=40,scale=c)
    else:
        rind = np.zeros(40)

    f0 = 2* np.sin(np.pi*x0)
    f1 = np.exp(2*x1)
    f2 = 0.2*np.power(x2,11)*np.power(10*(1-x2),6)+10*np.power(10*x2,3)*np.power(1-x2,10)
    f3 = np.zeros_like(x3)
    f4 = rind[x4]

    mu = f0 + f1 + f2 + f3 + f4 # eta in truth for non-Gaussian

    if isinstance(family,Gaussian):
        y = scp.stats.norm.rvs(loc=mu,scale=scale,size=n)
    
    elif isinstance(family,Gamma):
        # Need to transform from mean and scale to \alpha & \beta
        # From Wood (2017), we have that
        # \phi = 1/\alpha
        # so \alpha = 1/\phi
        # From https://en.wikipedia.org/wiki/Gamma_distribution, we have that:
        # \mu = \alpha/\beta
        # \mu = 1/\phi/\beta
        # \beta = 1/\phi/\mu
        # scipy docs, say to set scale to 1/\beta.
        # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html
        mu = family.link.fi(mu)
        alpha = 1/scale
        beta = alpha/mu  
        y = scp.stats.gamma.rvs(a=alpha,scale=(1/beta),size=n)
    
    elif isinstance(family,Binomial):
        mu = family.link.fi(mu*0.1)
        y = scp.stats.binom.rvs(1, mu, size=n)

    dat = pd.DataFrame({"y":y,
                        "x0":x0,
                        "x1":x1,
                        "x2":x2,
                        "x3":x3,
                        "x4":[f"f_{fl}" for fl in x4]})
    return dat


def sim5(n):
    """
    Simulates `n` data-points for a Multi-nomial model - probability of Y_i being one of K=5 classes changes smoothly as a function of variable
    x and differently so for each class - based on slightly modified versions of the original functions of Gu & Whaba (1991).

    References:
    - Gu, C. & Whaba, G., (1991). Minimizing GCV/GML scores with multiple smoothing parameters via the Newton method.
    - Wood, S. N., Pya, N., Saefken, B., (2016). Smoothing Parameter and Model Selection for General Smooth Models
    - mgcv source code: gam.sim.r
    """
    x0 = np.random.rand(n)

    f0 = 2* np.sin(np.pi*x0)
    f1 = np.exp(2*x0)*0.2
    f2 = 1e-4*np.power(x0,11)*np.power(10*(1-x0),6)+10*np.power(10*x0,3)*np.power(1-x0,10)
    f3 = 1*x0 + 0.03*x0**2

    family = MULNOMLSS(4)

    mus = [np.exp(f0),np.exp(f1),np.exp(f2),np.exp(f3)]
    
    ps = np.zeros((n,5))

    for k in range(5):
        lpk = family.lp(np.zeros(n)+k, *mus)
        ps[:,k] += lpk
    
    y = np.zeros(n,dtype=int)
    
    for i in range(n):
        y[i] = int(np.random.choice([0,1,2,3,4],p=np.exp(ps[i,:]),size=1)[0])

    dat = pd.DataFrame({"y":y,
                        "x0":x0})
    return dat


def sim6(n,family=GAUMLSS([Identity(),LOG()])):
    """
    Simulates `n` data-points for a Gaussian or Gamma GAMLSS model - mean and standard deviation/scale change based on 
    the original functions of Gu & Whaba (1991).

    References:
    - Gu, C. & Whaba, G., (1991). Minimizing GCV/GML scores with multiple smoothing parameters via the Newton method.
    - Wood, S. N., Pya, N., Saefken, B., (2016). Smoothing Parameter and Model Selection for General Smooth Models
    - mgcv source code: gam.sim.r

    :param family: Distribution for response variable, must be: `GAUMLSS()`, `GAMMALS()`. Defaults to `GAUMLSS([Identity(),LOG()])`
    :type family: GAMLSSFamily, optional
    """
    x0 = np.random.rand(n)
    mu_sd = 2* np.sin(np.pi*x0)
    mu_mean = 0.2*np.power(x0,11)*np.power(10*(1-x0),6)+10*np.power(10*x0,3)*np.power(1-x0,10)

    mus = [mu_mean,mu_sd]

    if isinstance(family,GAUMLSS):
        y = scp.stats.norm.rvs(loc=mus[0],scale=mus[1],size=n)

    elif isinstance(family,GAMMALS):
        # Need to transform from mean and scale to \alpha & \beta
        # From Wood (2017), we have that
        # \phi = 1/\alpha
        # so \alpha = 1/\phi
        # From https://en.wikipedia.org/wiki/Gamma_distribution, we have that:
        # \mu = \alpha/\beta
        # \mu = 1/\phi/\beta
        # \beta = 1/\phi/\mu
        # scipy docs, say to set scale to 1/\beta.
        # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html

        mus[0] += 1
        mus[1] += 1

        alpha = 1/mus[1]
        beta = alpha/mus[0]  
        y = scp.stats.gamma.rvs(a=alpha,scale=(1/beta),size=n)
    
    dat = pd.DataFrame({"y":y,
                        "x0":x0})
    return dat