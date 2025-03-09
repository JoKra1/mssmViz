import numpy as np
import scipy as scp
import pandas as pd
from mssm.models import *
from mssm.src.python.smooths import convolve_event
from mssm.src.python.gamm_solvers import cpp_cholP,compute_Linv,apply_eigen_perm
from mssm.src.python.formula import reparam

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
                            "y":scp.stats.norm.rvs(size=len(time_pred),random_state=20)})

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
    fixed2 = np.array([-5,*scp.stats.norm.rvs(size=(sim_S.shape[1]-1),scale=5,random_state=int(fixed_seed*3))]).reshape(-1,1)
    fixed3 = np.zeros_like(fixed2)
    fixed_sim_time_coefs = [fixed1,fixed2,fixed3]

    # Also get intercepts alone
    true_offsets = [fixed1[0],fixed2[0],fixed3[0]]

    # Prepare random smooth sampler
    # Based on Wood (2017, 6.10)
    nH = sim_mat.T @ sim_mat + sim_S
    Lp, Pr, code = cpp_cholP(nH.tocsc())

    if code != 0:
        raise ValueError("Cholesky failed.")
    
    LVp = compute_Linv(Lp,1)
    LV = apply_eigen_perm(Pr,LVp)
    V = (LV.T @ LV) * sim_sigma
    V = V.toarray()
    V[np.abs(V) < 1e-4] = 0 # Minimize numerical inaccuracies between different systems to ensure similar behavior

    #V = scp.sparse.linalg.spsolve(sim_mat.T @ sim_mat + sim_S,scp.sparse.eye((sim_S.shape[1]),format='csc')) * sim_sigma

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
    sub = [] # subject of each data point

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
        if not random_seed is None:
            sample = scp.stats.multivariate_normal.rvs(mean=scp.stats.norm.rvs(size=(sim_S.shape[1]),scale=5,random_state=random_seed+sim_idx),cov=V,size=1,random_state=random_seed+sim_idx)
        else:
            sample = scp.stats.multivariate_normal.rvs(mean=scp.stats.norm.rvs(size=(sim_S.shape[1]),scale=5,random_state=None),cov=V,size=1,random_state=None)
        sample[0] = 0
        take = np_gen.integers(int(len(time_pred)/4),len(time_pred)+1)
        fact.extend(np.repeat(fl[sim_idx],take))
        sub.extend([f"sub_{sim_idx%20}" for _ in range(take)])
        time.extend(time_pred[0:take])
        x.extend(np.repeat(xl[sim_idx],take))
        il.extend(np.repeat(sim_idx,take))
        ft.extend(((sim_mat @ sample) + rand_int[sim_idx] + time_pred*rand_slope[sim_idx])[0:take])
        
        if sim_idx < rand_matrix.shape[0]:
            rand_matrix[sim_idx,:] += ((sim_mat @ sample) + rand_int[sim_idx] + time_pred*rand_slope[sim_idx])

    time = np.array(time)
    x = np.array(x)
    fact = np.array(fact)
    sub = np.array(sub)
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
                                "sub":sub,
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
                            "y":scp.stats.norm.rvs(size=len(time_pred),random_state=20)})

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
    nH = sim_mat.T @ sim_mat + sim_S
    Lp, Pr, code = cpp_cholP(nH.tocsc())
    
    if code != 0:
        raise ValueError("Cholesky failed.")
    
    LVp = compute_Linv(Lp,1)
    LV = apply_eigen_perm(Pr,LVp)
    V = (LV.T @ LV) * sim_sigma
    V = V.toarray()
    V[np.abs(V) < 1e-4] = 0 # Minimize numerical inaccuracies between different systems to ensure similar behavior

    #V = scp.sparse.linalg.spsolve(sim_mat.T @ sim_mat + sim_S,scp.sparse.eye((sim_S.shape[1]),format='csc')) * sim_sigma

    # Get matrix for x effects
    sim_formula2 = Formula(lhs("y"),
                        [i(),f(["x"],nk=5)],
                        data=sim_dat,
                        print_warn=False)

    sim_model2 = GAMM(sim_formula2,Gaussian())
    sim_model2.fit(progress_bar=False)
    sim_mat2 = sim_model2.get_mmat()

    # Get fixed x effects
    fixed_x = np.array([0,*scp.stats.norm.rvs(size=(5),scale=5,random_state=int(fixed_seed*6))]).reshape(-1,1)

    # Get matrix for z effects
    sim_formula3 = Formula(lhs("y"),
                        [i(),f(["z"],nk=10)],
                        data=sim_dat,
                        print_warn=False)

    sim_model3 = GAMM(sim_formula3,Gaussian())
    sim_model3.fit(progress_bar=False)
    sim_mat3 = sim_model3.get_mmat()

    # Get fixed z effects
    fixed_z = np.array([0,*scp.stats.norm.rvs(size=(10),scale=5,random_state=int(fixed_seed*15))]).reshape(-1,1)

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
        if not random_seed is None:
            sample = scp.stats.multivariate_normal.rvs(mean=scp.stats.norm.rvs(size=(sim_S.shape[1]),scale=5,random_state=random_seed+sim_idx),cov=V,size=1,random_state=random_seed+sim_idx)
        else:
            sample = scp.stats.multivariate_normal.rvs(mean=scp.stats.norm.rvs(size=(sim_S.shape[1]),scale=5,random_state=None),cov=V,size=1,random_state=None)
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

def sim3(n,scale,c=1,binom_offset=0,family=Gaussian(),prop_q=0.95,correlate=False,seed=None):
    """
    First Simulation performed by Wood et al., (2016): 4 smooths, 1 is really zero everywhere.
    Based on the original functions of Gu & Whaba (1991).

    This is also the first simulation performed by gamSim() - except for the fact that f(x_0) can also be set to
    zero, as was done by Wood et al., (2016)

    Covariates can also be simulated to correlate with each other, following the steps outlined in supplementary materials
    E of Wood et al., (2016).

    References:

     - Gu, C. & Whaba, G., (1991). Minimizing GCV/GML scores with multiple smoothing parameters via the Newton method.
     - Wood, S. N., Pya, N., Saefken, B., (2016). Smoothing Parameter and Model Selection for General Smooth Models
     - mgcv source code: gam.sim.r

    :param scale: Standard deviation for `family='Gaussian'` else scale parameter
    :type scale: float
    :param c: Effect strength for x3 effect - 0 = No effect, 1 = Maximal effect
    :type c: float
    :param binom_offset: Additive adjustment to log-predictor for Binomial and Poisson model (5 in mgcv) and baseline hazard parameter for Propoprtional Hazard model. Defaults to 0.
    :type binom_offset: float
    :param prop_q: Simulated times exceeding ``q(prop_q)'' where ``q`` is the quantile function of a Weibull model parameterized with ``b=np.min(max(binom_offset,0.01)*np.exp(eta))`` are treated as censored for Propoprtional Hazard model. Defaults to 0.
    :type prop_q: float
    :param correlate: Whether predictor covariates should correlate or not. Defaults to False
    :type correlate: bool
    :param family: Distribution for response variable, must be: `Gaussian()`, `Gamma()`, `Binomial()`, or `Poisson()`. Defaults to `Gaussian()`
    :type family: Family, optional
    """
    np_gen = np.random.default_rng(seed)

    if correlate:
        # Following steps by Wood et al. (2016)
        Sigma = np.zeros((4,4)) + 0.9

        for ij in range(4):
            Sigma[ij,ij] = 1

        z = scp.stats.multivariate_normal.rvs(mean=[0 for _ in range(4)],cov=Sigma,size=n,random_state=seed)
        
        # I am a bit confused by notation in Wood et al. (2016) - they say x = cdf^{-1}(z) but I think that's not
        # what they mean, since cdf^{-1} = percent wise/quantile function which expects values between 0-1 which
        # is not the support for z. So I just use the cdf - which I think is what they mean. The resulting marginals
        # for x are uniform and all variables show the expected correlation, so it's probably correct.
        x0 = scp.stats.norm.cdf(z[:,0])
        x1 = scp.stats.norm.cdf(z[:,1])
        x2 = scp.stats.norm.cdf(z[:,2])
        x3 = scp.stats.norm.cdf(z[:,3])
    else:

        x0 = np_gen.random(n)
        x1 = np_gen.random(n)
        x2 = np_gen.random(n)
        x3 = np_gen.random(n)

    f0 = 2* np.sin(np.pi*x0)
    f1 = np.exp(2*x1)
    f2 = 0.2*np.power(x2,11)*np.power(10*(1-x2),6)+10*np.power(10*x2,3)*np.power(1-x2,10)
    f3 = np.zeros_like(x3)

    eta = c*f0 + f1 + f2 + f3 # eta in truth for non-Gaussian

    if isinstance(family,Gaussian):
        y = scp.stats.norm.rvs(loc=eta,scale=scale,size=n,random_state=seed)
    
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
        mu = family.link.fi(eta)
        alpha = 1/scale
        beta = alpha/mu  
        y = scp.stats.gamma.rvs(a=alpha,scale=(1/beta),size=n,random_state=seed)
    
    elif isinstance(family,Binomial):
        eta += binom_offset
        mu = family.link.fi(eta*scale)
        y = scp.stats.binom.rvs(1, mu, size=n,random_state=seed)

    elif isinstance(family,Poisson):
        eta += binom_offset
        mu = family.link.fi(eta*scale)
        y = scp.stats.poisson.rvs(mu, size=n,random_state=seed)

    elif isinstance(family,PropHaz):
        # Based on example code for mgcv's coxph family available here:
        # https://github.com/cran/mgcv/blob/aff4560d187dfd7d98c7bd367f5a0076faf129b7/man/coxph.Rd
        # Assumes a Weibull proportional Hazard model so baseline hazard function is
        # simply the Weibull hazard function as defined on Wikipedia, see:
        # https://en.wikipedia.org/wiki/Weibull_distribution#Cumulative_distribution_function
        # parameterization below assumes alternative re-parameterization on Wikipedia with
        # b = \lambda^{-k} and k here is the scale

        # First center eta
        eta -= np.mean(eta)

        # Now compute b parameter of Weibull
        b = max(binom_offset,0.01)*np.exp(eta)

        # And sample from Weibull quantile function as also done by numpy, see:
        # https://numpy.org/doc/2.0/reference/random/generated/numpy.random.weibull.html
        U = np_gen.random(n)

        y = np.power((-np.log(U)/b),scale)

        # Determine censoring cut-off
        b_limit = np.min(max(binom_offset,0.01)*np.exp(eta))
        q_limit = np.power((-np.log(prop_q)/b_limit),scale) 

        # Apply censoring
        delta = y <= q_limit
        y[delta == False] = 0
        delta = 1*delta


    dat = pd.DataFrame({"y":y,
                        "x0":x0,
                        "x1":x1,
                        "x2":x2,
                        "x3":x3,
                        "eta":eta})

    if isinstance(family,PropHaz):
        dat["delta"] = delta

    return dat


def sim4(n,scale,c=1,binom_offset=0,family=Gaussian(),prop_q=0.95,correlate=False,seed=None):
    """
    Like ``sim3``, except that a random factor is added - second simulation performed by Wood et al., (2016).

    This is also the sixth simulation performed by gamSim() - except for the fact that c is used here to scale the contribution
    of the random factor, as was also done by Wood et al., (2016)
    
    Covariates can also be simulated to correlate with each other, following the steps outlined in supplementary materials
    E of Wood et al., (2016).

    References:

     - Gu, C. & Whaba, G., (1991). Minimizing GCV/GML scores with multiple smoothing parameters via the Newton method.
     - Wood, S. N., Pya, N., Saefken, B., (2016). Smoothing Parameter and Model Selection for General Smooth Models
     - mgcv source code: gam.sim.r

    :param scale: Standard deviation for `family='Gaussian'` else scale parameter
    :type scale: float
    :param c: Effect strength for random effect - 0 = No effect (sd=0), 1 = Maximal effect (sd=1)
    :type c: float
    :param binom_offset: Additive adjustment to log-predictor for Binomial and Poisson model (5 in mgcv) and baseline hazard parameter for Propoprtional Hazard model. Defaults to 0.
    :type binom_offset: float
    :param prop_q: Simulated times exceeding ``q(prop_q)'' where ``q`` is the quantile function of a Weibull model parameterized with ``b=np.min(max(binom_offset,0.01)*np.exp(eta))`` are treated as censored for Propoprtional Hazard model. Defaults to 0.
    :type prop_q: float
    :param family: Distribution for response variable, must be: `Gaussian()`, `Gamma()`, `Binomial()`, or `Poisson()`. Defaults to `Gaussian()`
    :type family: Family, optional
    :param correlate: Whether predictor covariates should correlate or not. Defaults to False
    :type correlate: bool
    """
    np_gen = np.random.default_rng(seed)

    if correlate:
        Sigma = np.zeros((4,4)) + 0.9

        for ij in range(4):
            Sigma[ij,ij] = 1

        z = scp.stats.multivariate_normal.rvs(mean=[0 for _ in range(4)],cov=Sigma,size=n,random_state=seed)
        
        # I am a bit confused by notation in Wood et al. (2016) - they say x = cdf^{-1}(z) but I think that's not
        # what they mean, since cdf^{-1} = percent wise/quantile function which expects values between 0-1 which
        # is not the support for z. So I just use the cdf - which I think is what they mean. The resulting marginals
        # for x are uniform and all variables show the expected correlation, so it's probably correct.
        x0 = scp.stats.norm.cdf(z[:,0])
        x1 = scp.stats.norm.cdf(z[:,1])
        x2 = scp.stats.norm.cdf(z[:,2])
        x3 = scp.stats.norm.cdf(z[:,3])
    else:
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

    eta = f0 + f1 + f2 + f3 + f4 # eta in truth for non-Gaussian

    if isinstance(family,Gaussian):
        y = scp.stats.norm.rvs(loc=eta,scale=scale,size=n,random_state=seed)
    
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
        mu = family.link.fi(eta)
        alpha = 1/scale
        beta = alpha/mu  
        y = scp.stats.gamma.rvs(a=alpha,scale=(1/beta),size=n,random_state=seed)
    
    elif isinstance(family,Binomial):
        eta += binom_offset
        mu = family.link.fi(eta*scale)
        y = scp.stats.binom.rvs(1, mu, size=n,random_state=seed)

    elif isinstance(family,Poisson):
        mu = family.link.fi(eta*scale)
        y = scp.stats.poisson.rvs(mu, size=n,random_state=seed)

    elif isinstance(family,PropHaz):
        # Based on example code for mgcv's coxph family available here:
        # https://github.com/cran/mgcv/blob/aff4560d187dfd7d98c7bd367f5a0076faf129b7/man/coxph.Rd
        # Assumes a Weibull proportional Hazard model so baseline hazard function is
        # simply the Weibull hazard function as defined on Wikipedia, see:
        # https://en.wikipedia.org/wiki/Weibull_distribution#Cumulative_distribution_function
        # parameterization below assumes alternative re-parameterization on Wikipedia with
        # b = \lambda^{-k} and k here is the scale

        # First center eta
        eta -= np.mean(eta)

        # Now compute b parameter of Weibull
        b = max(binom_offset,0.01)*np.exp(eta)

        # And sample from Weibull quantile function as also done by numpy, see:
        # https://numpy.org/doc/2.0/reference/random/generated/numpy.random.weibull.html
        U = np_gen.random(n)

        y = np.power((-np.log(U)/b),scale)

        # Determine censoring cut-off
        b_limit = np.min(max(binom_offset,0.01)*np.exp(eta))
        q_limit = np.power((-np.log(prop_q)/b_limit),scale) 

        # Apply censoring
        delta = y <= q_limit
        y[delta == False] = 0
        delta = 1*delta

    dat = pd.DataFrame({"y":y,
                        "x0":x0,
                        "x1":x1,
                        "x2":x2,
                        "x3":x3,
                        "x4":[f"f_{fl}" for fl in x4],
                        "eta":eta})
    
    if isinstance(family,PropHaz):
        dat["delta"] = delta

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
    Simulates `n` data-points for a Gaussian or Gamma GAMMLSS model - mean and standard deviation/scale change based on 
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

def sim7(n,c,scale,family=Gaussian(),seed=None):
    """
    An overlap simulation with a random intercept. Two events are present that differ in their onset across
    different trials - each event triggers a response in the signal (modeled as two functions of Gu & Whaba).

    References:

     - Gu, C. & Whaba, G., (1991). Minimizing GCV/GML scores with multiple smoothing parameters via the Newton method.
     - Wood, S. N., Pya, N., Saefken, B., (2016). Smoothing Parameter and Model Selection for General Smooth Models
     - mgcv source code: gam.sim.r

    :param n: Number of trials to simulate
    :type n: int
    :param c: Effect strength for random effect - 0 = No effect (sd=0), 1 = Maximal effect (sd=1)
    :type c: float
    :param scale: Standard deviation for `family='Gaussian'` else scale parameter
    :type scale: float
    :param family: Distribution for response variable, must be: `Gaussian()`, `Gamma()`, or `Binomial()`. Defaults to `Gaussian()`
    :type family: Family, optional
    """

    np_gen = np.random.default_rng(seed)

    onsets1 = np_gen.integers(5,high=10,size=n)
    onsets2 = onsets1 + np_gen.integers(1,high=10,size=n)
    ends = onsets2 + np_gen.integers(15,high=30,size=n)
    
    if c > 0:
        rind = scp.stats.norm.rvs(size=40,scale=c,random_state=seed)
    else:
        rind = np.zeros(40)

    y = []
    time = []
    fl = []
    series = []

    for tr in range(n):
        x = np.linspace(0,1,20)
        f1 = 2* np.sin(np.pi*x)
        f2 = 0.2*np.power(x,11)*np.power(10*(1-x),6)+10*np.power(10*x,3)*np.power(1-x,10)
        f1_emb = np.zeros(ends[tr]+1)
        f2_emb = np.zeros(ends[tr]+1)
        f1_emb[0:len(f1)] += f1
        f2_emb[0:len(f2)] += f2

        # Convolve to shift event onset
        o1 = convolve_event(f1_emb,onsets1[tr])[0:len(f1_emb)]
        o2 = convolve_event(f2_emb,onsets2[tr])[0:len(f2_emb)]

        # Create & collect all variables
        t = np.arange(len(o1))*10
        y.extend(o1 + o2 + rind[tr%40])
        time.extend(t)
        fl.extend([f"l{tr%40}" for _ in range(len(o1))])
        series.extend([tr for _ in range(len(o1))])

    y = scp.stats.norm.rvs(loc=np.array(y),scale=scale,size=len(y),random_state=seed) 

    dat = pd.DataFrame({"y":y,
                        "time":time,
                        "factor":fl,
                        "series":series})
    
    return dat,onsets1,onsets2

def sim8(n,c,family=GAUMLSS([Identity(),LOG()]),seed=None):
    """
    Like sim6: Simulates `n` data-points for a Gaussian or Gamma GAMMLSS model - mean and standard deviation/scale change based on 
    the original functions of Gu & Whaba (1991). Difference is that the effect strength of the scale smoother can be manipulated by changing c!

    References:
    - Gu, C. & Whaba, G., (1991). Minimizing GCV/GML scores with multiple smoothing parameters via the Newton method.
    - Wood, S. N., Pya, N., Saefken, B., (2016). Smoothing Parameter and Model Selection for General Smooth Models
    - mgcv source code: gam.sim.r

    :param c: Effect strength for random effect - 0 = No effect (sd=0), 1 = Maximal effect (sd=1)
    :type c: float
    :param family: Distribution for response variable, must be: `GAUMLSS()`, `GAMMALS()`. Defaults to `GAUMLSS([Identity(),LOG()])`
    :type family: GAMLSSFamily, optional
    """
    np_gen = np.random.default_rng(seed)

    x0 = np_gen.random(n)
    mu_sd = 2* np.sin(np.pi*x0)
    mu_sd = 3.5 + c*mu_sd
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


def sim9(n,c=1,family=GAUMLSS([Identity(),LOG()]),seed=None):
    """
    Like ``sim4``, except for a GAMMLSS model.

    The random intercept to be selected is included in the model of the mean. I.e. the model of the mean is: f0 + f1 + c*f4 For the sd: f2 + f3.

    References:

     - Gu, C. & Whaba, G., (1991). Minimizing GCV/GML scores with multiple smoothing parameters via the Newton method.
     - Wood, S. N., Pya, N., Saefken, B., (2016). Smoothing Parameter and Model Selection for General Smooth Models
     - mgcv source code: gam.sim.r

    :param scale: Standard deviation for `family='Gaussian'` else scale parameter
    :type scale: float
    :param c: Effect strength for random effect - 0 = No effect (sd=0), 1 = Maximal effect (sd=1)
    :type c: float
    :param family: Distribution for response variable, must be: `GAUMLSS()` or `GAMMALS()`. Defaults to `GAUMLSS([Identity(),LOGb()])`
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

    mu_mean = f0 + f1 + f4
    mu_sd = 5 + f2 + f3

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

        alpha = 1/mus[1]
        beta = alpha/mus[0]  
        y = scp.stats.gamma.rvs(a=alpha,scale=(1/beta),size=n,random_state=seed)

    dat = pd.DataFrame({"y":y,
                        "x0":x0,
                        "x1":x1,
                        "x2":x2,
                        "x3":x3,
                        "x4":[f"f_{fl}" for fl in x4]})
    return dat


def sim10(n,c=1,family=GAUMLSS([Identity(),LOG()]),seed=None):
    """
    Like ``sim9``, except that c is used to scale effect of f0.

    I.e. the model of the mean is: c*f0 + f1 + f4 For the sd: f2 + f3.

    References:

     - Gu, C. & Whaba, G., (1991). Minimizing GCV/GML scores with multiple smoothing parameters via the Newton method.
     - Wood, S. N., Pya, N., Saefken, B., (2016). Smoothing Parameter and Model Selection for General Smooth Models
     - mgcv source code: gam.sim.r

    :param scale: Standard deviation for `family='Gaussian'` else scale parameter
    :type scale: float
    :param c: Effect strength for random effect - 0 = No effect (sd=0), 1 = Maximal effect (sd=1)
    :type c: float
    :param family: Distribution for response variable, must be: `GAUMLSS()` or `GAMMALS()`. Defaults to `GAUMLSS([Identity(),LOG()])`
    :type family: Family, optional
    """
    np_gen = np.random.default_rng(seed)

    x0 = np_gen.random(n)
    x1 = np_gen.random(n)
    x2 = np_gen.random(n)
    x3 = np_gen.random(n)
    x4 = np_gen.integers(low=0,high=40,size=n)

    rind = scp.stats.norm.rvs(size=40,scale=1,random_state=seed)

    f0 = 2* np.sin(np.pi*x0)
    f1 = np.exp(2*x1)
    f2 = 0.2*np.power(x2,11)*np.power(10*(1-x2),6)+10*np.power(10*x2,3)*np.power(1-x2,10)
    f3 = np.zeros_like(x3)
    f4 = rind[x4]

    mu_mean = c*f0 + f1 + f4
    mu_sd = 5 + f2 + f3

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

        alpha = 1/mus[1]
        beta = alpha/mus[0]  
        y = scp.stats.gamma.rvs(a=alpha,scale=(1/beta),size=n,random_state=seed)

    dat = pd.DataFrame({"y":y,
                        "x0":x0,
                        "x1":x1,
                        "x2":x2,
                        "x3":x3,
                        "x4":[f"f_{fl}" for fl in x4]})
    return dat


def sim11(n,scale,c=1,binom_offset=0,n_ranef=40,family=Gaussian(),prop_q=0.95,seed=None):
    """
    Like ``sim4``, except that a random smooth of variable `x0` is added - extension of the second simulation performed by Wood et al., (2016).

    c is used here to scale the contribution of the random smooth. Setting it to 0 means the ground truth is maximally wiggly. Setting it to 1
    means the random smooth is actually a random intercept.

    References:

     - Gu, C. & Whaba, G., (1991). Minimizing GCV/GML scores with multiple smoothing parameters via the Newton method.
     - Wood, S. N., Pya, N., Saefken, B., (2016). Smoothing Parameter and Model Selection for General Smooth Models
     - mgcv source code: gam.sim.r

    :param scale: Standard deviation for `family='Gaussian'` else scale parameter
    :type scale: float
    :param c: Effect strength for random smooth - 0 = Maximally wiggly, 1 = ground truth is random intercept
    :type c: float
    :param binom_offset: Additive adjustment to log-predictor for Binomial model (5 in mgcv) and baseline hazard parameter for Propoprtional Hazard model. Defaults to 0.
    :type binom_offset: float
    :param prop_q: Simulated times exceeding ``q(prop_q)'' where ``q`` is the quantile function of a Weibull model parameterized with ``b=np.min(max(binom_offset,0.01)*np.exp(eta))`` are treated as censored for Propoprtional Hazard model. Defaults to 0.
    :type prop_q: float
    :param n_ranef: Number of levels for the random smooth term. Defaults to 40.
    :type n_ranef: int
    :param family: Distribution for response variable, must be: `Gaussian()`, `Gamma()`, or `Binomial()`. Defaults to `Gaussian()`
    :type family: Family, optional
    """
    np_gen = np.random.default_rng(seed)

    x0 = np_gen.random(n)
    x1 = np_gen.random(n)
    x2 = np_gen.random(n)
    x3 = np_gen.random(n)
    x4 = np_gen.integers(low=0,high=n_ranef-1,size=n)

    f0 = 2* np.sin(np.pi*x0)
    f1 = np.exp(2*x1)
    f2 = 0.2*np.power(x2,11)*np.power(10*(1-x2),6)+10*np.power(10*x2,3)*np.power(1-x2,10)
    f3 = np.zeros_like(x3)
    f4 = np.zeros_like(x0)

    c = max(1e-7,c)
    
    # Set up random smooth sampler for smooth of x0
    # Based on prior assumption discussed by Wood (2017)
    fs_dat = pd.DataFrame({"x0":x0,
                        "y":np_gen.random(n)})

    fs_formula = Formula(lhs=lhs("y"),
                            terms=[f(["x0"],identifiable=False,nk=10,penalty=[PenType.DIFFERENCE],pen_kwargs=[{"m":1}])],
                            data=fs_dat)

    fs_model = GAMM(fs_formula,Gaussian())
    fs_model.formula.build_penalties()

    mmat = fs_model.get_mmat()
    cov = fs_formula.cov_flat
    S = fs_formula.penalties[0].S_J

    C, Srp, Drp, IRrp, rms1, rms2, rp_rank = reparam(mmat,S,cov,identity=True,scale=False,QR=True)
    
    mmat_RP = mmat @ C
    mmat_RP[:,[-1]] = 1
    
    V = (Srp/(c*1e7)).toarray()
    
    V[-1,-1] = 1
    #print(V)

    for l4 in np.unique(x4):
        if not seed is None:
            sample = scp.stats.multivariate_normal.rvs(mean=[0 for _ in range(Srp.shape[1])],cov=V,size=1,random_state=seed+l4)
        else:
            sample = scp.stats.multivariate_normal.rvs(mean=[0 for _ in range(Srp.shape[1])],cov=V,size=1,random_state=None)
        #print(sample)
        if c >= 1: # Random smooth is random intercept
            sample[:-1] = 0

        fl4 = mmat_RP@sample
        f4[x4 == l4] = fl4[x4 == l4]


    eta = f0 + f1 + f2 + f3 + f4 # eta in truth for non-Gaussian

    if isinstance(family,Gaussian):
        y = scp.stats.norm.rvs(loc=eta,scale=scale,size=n,random_state=seed)
    
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
        mu = family.link.fi(eta)
        alpha = 1/scale
        beta = alpha/mu  
        y = scp.stats.gamma.rvs(a=alpha,scale=(1/beta),size=n,random_state=seed)
    
    elif isinstance(family,Binomial):
        eta += binom_offset
        mu = family.link.fi(eta*scale)
        y = scp.stats.binom.rvs(1, mu, size=n,random_state=seed)

    elif isinstance(family,PropHaz):
        # Based on example code for mgcv's coxph family available here:
        # https://github.com/cran/mgcv/blob/aff4560d187dfd7d98c7bd367f5a0076faf129b7/man/coxph.Rd
        # Assumes a Weibull proportional Hazard model so baseline hazard function is
        # simply the Weibull hazard function as defined on Wikipedia, see:
        # https://en.wikipedia.org/wiki/Weibull_distribution#Cumulative_distribution_function
        # parameterization below assumes alternative re-parameterization on Wikipedia with
        # b = \lambda^{-k} and k here is the scale

        # First center eta
        eta -= np.mean(eta)

        # Now compute b parameter of Weibull
        b = max(binom_offset,0.01)*np.exp(eta)

        # And sample from Weibull quantile function as also done by numpy, see:
        # https://numpy.org/doc/2.0/reference/random/generated/numpy.random.weibull.html
        U = np_gen.random(n)

        y = np.power((-np.log(U)/b),scale)

        # Determine censoring cut-off
        b_limit = np.min(max(binom_offset,0.01)*np.exp(eta))
        q_limit = np.power((-np.log(prop_q)/b_limit),scale) 

        # Apply censoring
        delta = y <= q_limit
        y[delta == False] = 0
        delta = 1*delta
    

    dat = pd.DataFrame({"y":y,
                        "x0":x0,
                        "x1":x1,
                        "x2":x2,
                        "x3":x3,
                        "x4":[f"f_{fl}" for fl in x4],
                        "eta":eta})
    
    if isinstance(family,PropHaz):
        dat["delta"] = delta

    return dat

def sim12(n,c=1,n_ranef=40,family=GAUMLSS([Identity(),LOG]),seed=None):
    """
    Like ``sim11``, except for GAMLSS models. ``x0``, ``x1``, and ``x4`` impact the mean - the remaining variables impact the scale parameter.

    c is used here to scale the contribution of the random smooth (which is included in the model fo the mean). Setting it to 0 means the ground truth is maximally wiggly. Setting it to 1
    means the random smooth is actually a random intercept.

    References:

     - Gu, C. & Whaba, G., (1991). Minimizing GCV/GML scores with multiple smoothing parameters via the Newton method.
     - Wood, S. N., Pya, N., Saefken, B., (2016). Smoothing Parameter and Model Selection for General Smooth Models
     - mgcv source code: gam.sim.r

    :param c: Effect strength for random smooth - 0 = Maximally wiggly, 1 = ground truth is random intercept
    :type c: float
    :param n_ranef: Number of levels for the random smooth term. Defaults to 40.
    :type n_ranef: int
    :param family: Distribution for response variable, must be: `GAUMLSS()` or `GAMMALS()`. Defaults to `GAUMLSS([Identity(),LOGb()])`
    :type family: Family, optional
    """
    np_gen = np.random.default_rng(seed)

    x0 = np_gen.random(n)
    x1 = np_gen.random(n)
    x2 = np_gen.random(n)
    x3 = np_gen.random(n)
    x4 = np_gen.integers(low=0,high=n_ranef-1,size=n)

    f0 = 2* np.sin(np.pi*x0)
    f1 = np.exp(2*x1)
    f2 = 0.2*np.power(x2,11)*np.power(10*(1-x2),6)+10*np.power(10*x2,3)*np.power(1-x2,10)
    f3 = np.zeros_like(x3)
    f4 = np.zeros_like(x0)

    c = max(1e-7,c)
    
    # Set up random smooth sampler for smooth of x0
    # Based on prior assumption discussed by Wood (2017)
    fs_dat = pd.DataFrame({"x0":x0,
                        "y":np_gen.random(n)})

    fs_formula = Formula(lhs=lhs("y"),
                            terms=[f(["x0"],identifiable=False,nk=10,penalty=[PenType.DIFFERENCE],pen_kwargs=[{"m":1}])],
                            data=fs_dat)

    fs_model = GAMM(fs_formula,Gaussian())
    fs_model.formula.build_penalties()

    mmat = fs_model.get_mmat()
    cov = fs_formula.cov_flat
    S = fs_formula.penalties[0].S_J

    C, Srp, Drp, IRrp, rms1, rms2, rp_rank = reparam(mmat,S,cov,identity=True,scale=False,QR=True)
    
    mmat_RP = mmat @ C
    mmat_RP[:,[-1]] = 1
    
    V = (Srp/(c*1e7)).toarray()
    
    V[-1,-1] = 1
    #print(V)

    for l4 in np.unique(x4):
        if not seed is None:
            sample = scp.stats.multivariate_normal.rvs(mean=[0 for _ in range(Srp.shape[1])],cov=V,size=1,random_state=seed+l4)
        else:
            sample = scp.stats.multivariate_normal.rvs(mean=[0 for _ in range(Srp.shape[1])],cov=V,size=1,random_state=None)
        #print(sample)
        if c >= 1: # Random smooth is random intercept
            sample[:-1] = 0

        fl4 = mmat_RP@sample
        f4[x4 == l4] = fl4[x4 == l4]

    eta_mean = f0 + f1 + f4
    eta_sd = f2 + f3
    
    eta_sd *= 0.1

    if isinstance(family.links[0],LOG):
        eta_mean *= 0.5

    mus = [family.links[0].fi(eta_mean),family.links[1].fi(eta_sd)]
    #print(mus[1])

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
        
        alpha = 1/mus[1]
        beta = alpha/mus[0]  
        y = scp.stats.gamma.rvs(a=alpha,scale=(1/beta),size=n,random_state=seed)

    dat = pd.DataFrame({"y":y,
                        "x0":x0,
                        "x1":x1,
                        "x2":x2,
                        "x3":x3,
                        "x4":[f"f_{fl}" for fl in x4],
                        "eta_mean":eta_mean,
                        "eta_scale":eta_sd})
    return dat