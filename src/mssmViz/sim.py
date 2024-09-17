import numpy as np
import scipy as scp
import pandas as pd
from mssm.models import *

################################## Contains simulations to simulate for GAMM & GAMMLSS models ##################################

def sim1(sim_size,sim_sigma = 5.5,sim_lam = 1e-4,sim_weak_nonlin = 0.5,random_seed=None,fixed_seed=42*3):
    """First simulation for an additive time-series model with trial-level non-linear random effects. Data-set contains covariates time & x,
    for which the effect is different for three levels of factor variable fact.

    :param sim_size: Number of trials, defaults to 1000
    :type sim_size: int, optional
    :param sim_sigma: Standard error for residuals, defaults to 5.5
    :type sim_sigma: float, optional
    :param sim_lam: Lambda parameter for trial-level non-linear effect complexity, defaults to 1e-4
    :type sim_lam: _type_, optional
    :param sim_weak_nonlin: Strength of weakly non-linear covariate x effect, defaults to 0.5
    :type sim_weak_nonlin: float, optional
    :param random_seed: Seed for random parts of the simulation - should differ between repeated simulations, defaults to None
    :type random_seed: int, optional
    :param fixed_seed: Seed for fixed effects in the simulation - should NOT differ between repeated simulations, defaults to None
    :type fixed_seed: int, optional
    :return: Tuple, first element contains a ``pd.DataFrame`` with simulated data, second element is again a tuple containing: a ``np.array`` with the trial-level deviations, design matrices used for simulation, true coefficients used for simulation, and true intercepts used for simulation.
    :rtype: (pd.Dataframe,(np.array,np.array,np.array,np.array,np.array,np.array))
    """
    global i
    global f
    global lhs


    # Set up fixed and random effects
    time_pred = np.array([t for t in range(0,3000,20)])
    x_pred = np.linspace(0,25,len(time_pred))

    # Get matrix for time effects
    sim_dat = pd.DataFrame({"Time":time_pred,
                            "x":x_pred,
                            "y":scp.stats.norm.rvs(size=len(time_pred))})

    sim_formula = Formula(lhs("y"),
                        [i(),f(["Time"],nk=15)],
                        data=sim_dat,
                        print_warn=False)

    sim_model = GAMM(sim_formula,Gaussian())
    sim_model.fit(progress_bar=False)
    sim_mat = sim_model.get_mmat()

    sim_S = sim_formula.penalties[0].S_J_emb * sim_lam

    # Get fixed time effects (+intercept)
    fixed1 = np.array([5,*scp.stats.norm.rvs(size=(sim_S.shape[1]-1),scale=5,random_state=fixed_seed)]).reshape(-1,1)
    fixed2 = np.array([-5,*scp.stats.norm.rvs(size=(sim_S.shape[1]-1),scale=5,random_state=fixed_seed*3)]).reshape(-1,1)
    fixed3 = np.zeros_like(fixed2)
    fixed_sim_time_coefs = [fixed1,fixed2,fixed3]

    # Also get intercepts alone
    true_offsets = [fixed1[0],fixed2[0],fixed3[0]]

    # Prepare random smooth sampler
    # Based on Wood (2017, 6.10)
    V = scp.sparse.linalg.spsolve(sim_mat.T @ sim_mat + sim_S,scp.sparse.eye((sim_S.shape[1]),format='csc')) * sim_sigma

    # Get matrix for x effects
    sim_formula2 = Formula(lhs("y"),
                        [i(),f(["x"],nk=5)],
                        data=sim_dat,
                        print_warn=False)

    sim_model2 = GAMM(sim_formula2,Gaussian())
    sim_model2.fit(progress_bar=False)
    sim_mat2 = sim_model2.get_mmat()

    # Get fixed x effects
    fixedX1 = np.array([0,*scp.stats.norm.rvs(size=(5),scale=5,random_state=fixed_seed*6)]).reshape(-1,1)
    fixedX2 = np.array([0,*scp.stats.norm.rvs(size=(5),scale=5,random_state=fixed_seed*9)]).reshape(-1,1)
    fixedX3 = np.array([0,*np.linspace(-sim_weak_nonlin,sim_weak_nonlin,len(fixedX2)-1)]).reshape(-1,1)
    fixed_sim_x_coefs = [fixedX1,fixedX2,fixedX3]
    
    ft = [] # series specific effect for each data point
    time = [] # time of each data point
    x = [] # x covariate of each data point
    il = [] # id of each data point
    fact = [] # group of each data point

    # Simulation seed
    np_gen = np.random.default_rng(random_seed)

    # Group assignment
    fl = np_gen.choice([1,2,3],size=sim_size,replace=True,p=[0.5,0.2,0.3])

    # x values for each trial
    xl = np_gen.choice(x_pred,size=sim_size,replace=True)

    # Sample trial-level smooths

    # random offsets
    rand_int = scp.stats.norm.rvs(size=sim_size,scale=2.5,random_state=random_seed)

    # random drifts
    rand_slope = scp.stats.norm.rvs(size=sim_size,scale=0.0025,random_state=random_seed)

    rand_matrix = np.zeros((100,len(time_pred)))
    for sim_idx in range(sim_size):
        sample = scp.stats.multivariate_normal.rvs(mean=scp.stats.norm.rvs(size=(sim_S.shape[1]),scale=5,random_state=random_seed+sim_idx),cov=V.toarray(),size=1,random_state=random_seed+sim_idx)
        sample[0] = 0
        take = np_gen.integers(int(len(time_pred)/4),len(time_pred)+1)
        fact.extend(np.repeat(fl[sim_idx],take))
        time.extend(time_pred[0:take])
        x.extend(np.repeat(xl[sim_idx],take))
        il.extend(np.repeat(sim_idx,take))
        ft.extend(((sim_mat @ sample) + rand_int[sim_idx] + time_pred*rand_slope[sim_idx])[0:take])
        
        if sim_idx < rand_matrix.shape[0]:
            rand_matrix[sim_idx,:] += ((sim_mat @ sample) + rand_int[sim_idx] + time_pred*rand_slope[sim_idx])

    time = np.array(time)
    x = np.array(x)
    fact = np.array(fact)
    ft = np.array(ft).reshape(-1,1)

    # Get fixed predictions
    f0 = np.zeros((len(time))) # time
    f1 = np.zeros((len(time))) # x

    for fi in [1,2,3]:
        sim_cond_dat = pd.DataFrame({"Time":time[fact == fi]})
        sim_condX_dat = pd.DataFrame({"x":x[fact == fi]})
        _,sim_mat_cond,_ = sim_model.predict([0,1],sim_cond_dat)
        _,sim_matX_cond,_ = sim_model2.predict([0,1],sim_condX_dat)

        f0[fact == fi] = np.ndarray.flatten(sim_mat_cond@fixed_sim_time_coefs[fi-1])
        f1[fact == fi] = np.ndarray.flatten(sim_matX_cond@fixed_sim_x_coefs[fi-1])
    
    f0 = np.array(f0).reshape(-1,1)
    f1 = np.array(f1).reshape(-1,1)

    # Now build sim dat and define formula
    sim_fit_dat = pd.DataFrame({"y":np.ndarray.flatten(f0 + f1 + ft + scp.stats.norm.rvs(size=len(f0),scale=sim_sigma,random_state=random_seed).reshape(-1,1)),
                                "truth":np.ndarray.flatten(f0 + f1),
                                "time":time,
                                "x":x,
                                "fact":[f"fact_{fc}" for fc in fact],
                                "series":[f"series_{ic}" for ic in il]})

    return sim_fit_dat,(rand_matrix,sim_mat,sim_mat2,fixed_sim_time_coefs,fixed_sim_x_coefs,true_offsets)

def sim2(sim_size,sim_sigma = 5.5,sim_lam = 1e-4,set_zero = 1,random_seed=None,fixed_seed=42*3):
    """Second simulation for an additive time-series model with trial-level non-linear random effects. Data contains two
    additional covariates apart from time (x & z) - values for z vary within and between series, x only between series.

    Ground-truth for x or z can be set to zero.

    :param sim_size: Number of trials, defaults to 1000
    :type sim_size: int, optional
    :param sim_sigma: Standard error for residuals, defaults to 5.5
    :type sim_sigma: float, optional
    :param sim_lam: Lambda parameter for trial-level non-linear effect complexity, defaults to 1e-4
    :type sim_lam: _type_, optional
    :param set_zero: Which covariate (1 or 2 for x and z respectively) to set to zero, defaults to 1
    :type set_zero: int, optional
    :param random_seed: Seed for random parts of the simulation - should differ between repeated simulations, defaults to None
    :type random_seed: int, optional
    :param fixed_seed: Seed for fixed effects in the simulation - should NOT differ between repeated simulations, defaults to None
    :type fixed_seed: int, optional
    :return: Tuple, first element contains a ``pd.DataFrame`` with simulated data, second element is again a tuple containing: a ``np.array`` with the trial-level deviations, design matrices used for simulation, true effects used for simulation, and true offset used for simulation.
    :rtype: (pd.Dataframe,(np.array,np.array,np.array,np.array,np.array,np.array,np.array,float))
    """
    global i
    global f
    global lhs

    # Set up fixed and random effects
    time_pred = np.array([t for t in range(0,3000,20)])
    x_pred = np.linspace(0,25,len(time_pred))
    z_pred = np.linspace(-1,1,len(time_pred))

    # Get matrix for time effects
    sim_dat = pd.DataFrame({"Time":time_pred,
                            "x":x_pred,
                            "z":z_pred,
                            "y":scp.stats.norm.rvs(size=len(time_pred))})

    sim_formula = Formula(lhs("y"),
                        [i(),f(["Time"],nk=15)],
                        data=sim_dat,
                        print_warn=False)

    sim_model = GAMM(sim_formula,Gaussian())
    sim_model.fit(progress_bar=False)
    sim_mat = sim_model.get_mmat()

    sim_S = sim_formula.penalties[0].S_J_emb * sim_lam

    # Get fixed time effects
    fixed_time = np.array([5,*scp.stats.norm.rvs(size=(sim_S.shape[1]-1),scale=5,random_state=fixed_seed)]).reshape(-1,1)

    # Also get intercept alone
    true_offset = 5

    # Prepare random smooth sampler
    # Based on Wood (2017, 6.10)
    V = scp.sparse.linalg.spsolve(sim_mat.T @ sim_mat + sim_S,scp.sparse.eye((sim_S.shape[1]),format='csc')) * sim_sigma

    # Get matrix for x effects
    sim_formula2 = Formula(lhs("y"),
                        [i(),f(["x"],nk=5)],
                        data=sim_dat,
                        print_warn=False)

    sim_model2 = GAMM(sim_formula2,Gaussian())
    sim_model2.fit(progress_bar=False)
    sim_mat2 = sim_model2.get_mmat()

    # Get fixed x effects
    fixed_x = np.array([0,*scp.stats.norm.rvs(size=(5),scale=5,random_state=fixed_seed*6)]).reshape(-1,1)

    # Get matrix for z effects
    sim_formula3 = Formula(lhs("y"),
                        [i(),f(["z"],nk=10)],
                        data=sim_dat,
                        print_warn=False)

    sim_model3 = GAMM(sim_formula3,Gaussian())
    sim_model3.fit(progress_bar=False)
    sim_mat3 = sim_model3.get_mmat()

    # Get fixed z effects
    fixed_z = np.array([0,*scp.stats.norm.rvs(size=(10),scale=5,random_state=fixed_seed*15)]).reshape(-1,1)

    # Simulation seed
    np_gen = np.random.default_rng(random_seed)
    
    ft = [] # series specific effect for each data point
    time = [] # time of each data point
    x = [] # x covariate of each data point
    il = [] # id of each data point

    # x values for each trial
    xl = np_gen.choice(x_pred,size=sim_size,replace=True)

    # Sample trial-level smooths

    # random offsets
    rand_int = scp.stats.norm.rvs(size=sim_size,scale=2.5,random_state=random_seed)
    
    # random drifts
    rand_slope = scp.stats.norm.rvs(size=sim_size,scale=0.0025,random_state=random_seed)
    
    rand_matrix = np.zeros((100,len(time_pred)))
    for sim_idx in range(sim_size):
        sample = scp.stats.multivariate_normal.rvs(mean=scp.stats.norm.rvs(size=(sim_S.shape[1]),scale=5,random_state=random_seed+sim_idx),cov=V.toarray(),size=1,random_state=random_seed+sim_idx)
        sample[0] = 0
        take = np_gen.integers(int(len(time_pred)/4),len(time_pred)+1)
        
        time.extend(time_pred[0:take])
        x.extend(np.repeat(xl[sim_idx],take))
        il.extend(np.repeat(sim_idx,take))
        ft.extend(((sim_mat @ sample) + rand_int[sim_idx] + time_pred*rand_slope[sim_idx])[0:take])

        if sim_idx < rand_matrix.shape[0]:
            rand_matrix[sim_idx,:] + ((sim_mat @ sample) + rand_int[sim_idx] + time_pred*rand_slope[sim_idx])

    time = np.array(time)
    x = np.array(x)
    z = np_gen.choice(z_pred,size=len(time),replace=True) # z covariate of each data point
    ft = np.array(ft).reshape(-1,1)

    # Get fixed predictions
    sim_time_dat = pd.DataFrame({"Time":time})
    sim_X_dat = pd.DataFrame({"x":x})
    sim_Z_dat = pd.DataFrame({"z":z})

    _,sim_mat_time,_ = sim_model.predict([0,1],sim_time_dat)
    _,sim_mat_x,_ = sim_model2.predict([0,1],sim_X_dat)
    _,sim_mat_z,_ = sim_model3.predict([0,1],sim_Z_dat)

    f0 = sim_mat_time @ fixed_time # time
    f1 = sim_mat_x @ fixed_x # x
    f2 = sim_mat_z @ fixed_z # z

    # Set co-variate effects to zero
    if set_zero == 1:
        f1 = np.zeros_like(f1)

    if set_zero == 2:
        f2 = np.zeros_like(f2)

    # Now build sim dat and define formula
    sim_fit_dat = pd.DataFrame({"y":np.ndarray.flatten(f0 + f1 + f2 + ft + scp.stats.norm.rvs(size=len(f0),scale=sim_sigma,random_state=random_seed).reshape(-1,1)),
                                "truth":np.ndarray.flatten(f0 + f1 + f2),
                                "time":time,
                                "x":x,
                                "z":z,
                                "series":[f"series_{ic}" for ic in il]})
    return sim_fit_dat,(rand_matrix,sim_mat,sim_mat2,sim_mat3,fixed_time,fixed_x,fixed_z,true_offset)

def sim3(n,scale,c=1,family=Gaussian(),seed=None):
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
    np_gen = np.random.default_rng(seed)

    x0 = np_gen.random(n)
    x1 = np_gen.random(n)
    x2 = np_gen.random(n)
    x3 = np_gen.random(n)

    f0 = 2* np.sin(np.pi*x0)
    f1 = np.exp(2*x1)
    f2 = 0.2*np.power(x2,11)*np.power(10*(1-x2),6)+10*np.power(10*x2,3)*np.power(1-x2,10)
    f3 = np.zeros_like(x3)

    mu = c*f0 + f1 + f2 + f3 # eta in truth for non-Gaussian

    if isinstance(family,Gaussian):
        y = scp.stats.norm.rvs(loc=mu,scale=scale,size=n,random_state=seed)
    
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
        y = scp.stats.gamma.rvs(a=alpha,scale=(1/beta),size=n,random_state=seed)
    
    elif isinstance(family,Binomial):
        mu = family.link.fi(mu*0.1)
        y = scp.stats.binom.rvs(1, mu, size=n,random_state=seed)

    dat = pd.DataFrame({"y":y,
                        "x0":x0,
                        "x1":x1,
                        "x2":x2,
                        "x3":x3})
    return dat


def sim4(n,scale,c=1,family=Gaussian(),seed=None):
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
    np_gen = np.random.default_rng(seed)

    x0 = np_gen.random(n)
    x1 = np_gen.random(n)
    x2 = np_gen.random(n)
    x3 = np_gen.random(n)
    x4 = np_gen.integers(low=0,high=40,size=n)

    if c > 0:
        rind = scp.stats.norm.rvs(size=40,scale=c,random_state=seed)
    else:
        rind = np.zeros(40)

    f0 = 2* np.sin(np.pi*x0)
    f1 = np.exp(2*x1)
    f2 = 0.2*np.power(x2,11)*np.power(10*(1-x2),6)+10*np.power(10*x2,3)*np.power(1-x2,10)
    f3 = np.zeros_like(x3)
    f4 = rind[x4]

    mu = f0 + f1 + f2 + f3 + f4 # eta in truth for non-Gaussian

    if isinstance(family,Gaussian):
        y = scp.stats.norm.rvs(loc=mu,scale=scale,size=n,random_state=seed)
    
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
        y = scp.stats.gamma.rvs(a=alpha,scale=(1/beta),size=n,random_state=seed)
    
    elif isinstance(family,Binomial):
        mu = family.link.fi(mu*0.1)
        y = scp.stats.binom.rvs(1, mu, size=n,random_state=seed)

    dat = pd.DataFrame({"y":y,
                        "x0":x0,
                        "x1":x1,
                        "x2":x2,
                        "x3":x3,
                        "x4":[f"f_{fl}" for fl in x4]})
    return dat


def sim5(n,seed=None):
    """
    Simulates `n` data-points for a Multi-nomial model - probability of Y_i being one of K=5 classes changes smoothly as a function of variable
    x and differently so for each class - based on slightly modified versions of the original functions of Gu & Whaba (1991).

    References:
    - Gu, C. & Whaba, G., (1991). Minimizing GCV/GML scores with multiple smoothing parameters via the Newton method.
    - Wood, S. N., Pya, N., Saefken, B., (2016). Smoothing Parameter and Model Selection for General Smooth Models
    - mgcv source code: gam.sim.r
    """
    np_gen = np.random.default_rng(seed)

    x0 = np_gen.random(n)

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
        y[i] = int(np_gen.choice([0,1,2,3,4],p=np.exp(ps[i,:]),size=1)[0])

    dat = pd.DataFrame({"y":y,
                        "x0":x0})
    return dat


def sim6(n,family=GAUMLSS([Identity(),LOG()]),seed=None):
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
    np_gen = np.random.default_rng(seed)

    x0 = np_gen.random(n)
    mu_sd = 2* np.sin(np.pi*x0)
    mu_mean = 0.2*np.power(x0,11)*np.power(10*(1-x0),6)+10*np.power(10*x0,3)*np.power(1-x0,10)

    mus = [mu_mean,mu_sd]

    if isinstance(family,GAUMLSS):
        y = scp.stats.norm.rvs(loc=mus[0],scale=mus[1],size=n,random_state=seed)

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
        y = scp.stats.gamma.rvs(a=alpha,scale=(1/beta),size=n,random_state=seed)
    
    dat = pd.DataFrame({"y":y,
                        "x0":x0})
    return dat