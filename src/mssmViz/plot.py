
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import copy
import math
from .sim import np,pd,scp
from mssm.models import *
import warnings

################################## Contains functions to visualize and validate GAMM & GAMMLSS models ##################################


def __get_data_limit_counts(formula,pred_dat,cvars,by,by_cont,lim_dist=0.1):
    """Checks for every row in the data used for prediction, whether continuous variables are within training data limits.
     
    Also finds how often each combination of continuous variables exists in trainings data.

    :param formula: A GAMM Formula, model must have been fit.
    :type formula: Formula
    :param pred_dat: pandas DataFrame holding prediction data.
    :type pred_dat: pandas.Dataframe
    :param cvars: A list of the continuous variables to take into account.
    :type cvars: [str]
    :param by: A list of categorical variables associated with a smooth term, i.e., if a smooth has a different shape for different levels of a factor or a prediction.
    :type by: [str]
    :param by_cont: Optional name of a continuous by variable for a smooth term. Data points for which this covariate is exactly zero are treated as outside of limits.
    :type by_cont: str or None
    :param lim_dist: The floating point distance (on normalized scale, i.e., values have to be in ``[0,1]``) at which a point is considered too far away from training data. Setting this to 0 means we visualize only points for which there is trainings data, setting this to 1 means visualizing everything. Defaults to 0.1 
    :type lim_dist: float, optional
    :return: Three vectors + list. First contains bool indicating whether all continuous variables in prediction data row had values within training limits. Second contains all
    unique combinations of continuous variable values in training set. Third contains count for each unique combination in training set. Final list holds names of continuous variables in the order of the columns of the second vector.
    :rtype: tuple
    """
    
    _,pred_cov,_,_,_,_,_ = formula.encode_data(pred_dat,prediction=True)

    # Find continuous predictors and categorical ones
    pred_cols = pred_dat.columns
    cont_idx = [formula.get_var_map()[var] for var in pred_cols if formula.get_var_types()[var] == VarType.NUMERIC and var in cvars]
    cont_vars = [var for var in pred_cols if formula.get_var_types()[var] == VarType.NUMERIC and var in cvars]
    factor_idx = []

    if not by is None:
        factor_idx = [formula.get_var_map()[var] for var in pred_cols if formula.get_var_types()[var] == VarType.FACTOR and var in by]

    # Get sorted encoded cov structure for prediction and training data containing continuous variables
    sort_pred = pred_cov[:,cont_idx]
    sort_train = formula.cov_flat[:,cont_idx]

    by_cont_var = np.ones(sort_train.shape[0])

    if by_cont is not None:
        by_cont_idx = formula.get_var_map()[by_cont]
        by_cont_var = formula.cov_flat[:,by_cont_idx]

    if len(factor_idx) > 0 and not by is None:
        # Now get all columns corresponding to factor variables so that we can check
        # which rows in the trainings data belong to conditions present in the pred data.
        sort_cond_pred = pred_cov[:,factor_idx]
        sort_cond_train = formula.cov_flat[:,factor_idx]

        # Now get unique encoded rows - only considering factor variables
        pred_cond_unique = np.unique(sort_cond_pred,axis=0)

        train_cond_unq,train_cond_inv = np.unique(sort_cond_train,axis=0,return_inverse=True)

        # Check which training conditions are present in prediction
        train_cond_unq_exists = np.array([(train == pred_cond_unique).all(axis=1).any() for train in train_cond_unq])

        # Now get part of training cov matching conditions in prediction data
        sort_train = sort_train[train_cond_unq_exists[train_cond_inv] & (np.abs(by_cont_var) > 0),:]

    elif by_cont is not None:
        sort_train = sort_train[np.abs(by_cont_var) > 0,:]

    # Check for each combination in continuous prediction columns whether the values are within
    # min and max of the respective trainings columns
    pred_in_limits = np.ones(len(sort_pred),dtype=bool)
    sort_pred_scaled = copy.deepcopy(sort_pred)
    sort_train_scaled = copy.deepcopy(sort_train)
    
    # Based on exclude.too.far function in mgcv - scale new & train data to be in unit square
    # see: https://rdrr.io/cran/mgcv/src/R/plots.r#sym-exclude.too.far
    for cont_i in range(sort_pred.shape[1]):
        
        min_pred = min(sort_pred_scaled[:,cont_i])
        sort_pred_scaled[:,cont_i] -= min_pred
        sort_train_scaled[:,cont_i] -= min_pred
        max_pred = max(sort_pred_scaled[:,cont_i])
        sort_pred_scaled[:,cont_i] /= max_pred
        sort_train_scaled[:,cont_i] /= max_pred

    # Then find minimum distance to any data point for each prediction value
    dist = np.array([min(np.linalg.norm(sort_train_scaled - sort_pred_scaled[predi,:],axis=1)) for predi in range(sort_pred.shape[0])])
    pred_in_limits[dist > lim_dist] = False

    # Now find the counts in the training data for each combination of continuous variables
    train_unq,train_unq_counts = np.unique(sort_train,axis=0,return_counts=True)
    
    return pred_in_limits,train_unq,train_unq_counts.astype(float),cont_vars


def __pred_plot(pred,b,tvars,pred_in_limits,x1,x2,x1_exp,ci,n_vals,ax,_cmp,col,ylim,link,legend_label):
    """Internal function to visualize a univariate smooth of covariate `x1` or a tensor smooth of covariate `x1` and `x2`.

    Called by :func:`plot`, :func:`plot_fitted`, and :func:`plot_diff`.

    :param pred: Vector holding model prediction
    :type pred: [float]
    :param b: Vector holding standard error that needs to be added to/subtracted from `pred` to obtain ci boundaries.
    :type b: [float]
    :param tvars: List of variables to be visualized - contains one string for univariate smooths, two for tensor smooths.
    :type tvars: [str]
    :param pred_in_limits: bolean vector indicating which prediction was obtained for a covariate combination within data limits.
    :type pred_in_limits: [bool]
    :param x1: Unique values of covariate x1
    :type x1: [float]
    :param x2: Unique values of covariate x2
    :type x2: [float]
    :param x1_exp: For univariate smooth like `x1`, for tensor smooth this holds `x1` for each value of `x2`
    :type x1_exp: [float]
    :param ci: Same as `x1_exp` for `x2`
    :type ci: [float]
    :param n_vals: Number of values use to create each marginal covariate
    :type n_vals: int
    :param ax: matplotlib.axis to plot on
    :type ax: matplotlib.axis
    :param _cmp: matplotlib colormap
    :type _cmp: matplotlib.colormap
    :param col: color to use for univariate plot, float in [0,1]
    :type col: float
    :param ylim: limits for y-axis/z-axis
    :type ylim: (float,float)
    :param link: Link function of model.
    :type link: Link
    :param legend_label: Legend label to pass to univariate smooths.
    :type legend_label: str
    """
    
    # Handle tensor smooth case
    if len(tvars) == 2:
        T_pred = pred.reshape(n_vals,n_vals)

        if not link is None:
            T_pred = link.fi(T_pred)

        # Mask anything out of data limits.
        if not pred_in_limits is None:
            T_pred = np.ma.array(T_pred, mask = (pred_in_limits == False))

        T_pred = T_pred.T

        vmin = ylim[0] if ylim is not None else np.min(T_pred)
        vmax = ylim[1] if ylim is not None else np.max(T_pred)
        levels = np.linspace(vmin,vmax,n_vals)

        if ci:
            # Mask anything where CI contains zero
            f1 = ax.contourf(x1,x2,T_pred,levels=levels,cmap=_cmp,alpha=0.4)
            T_pred = np.ma.array(T_pred, mask= ((pred + b) > 0) & ((pred - b) < 0))
        
        # Plot everything (outside ci)
        ff = ax.contourf(x1,x2,T_pred,levels=levels,cmap=_cmp)
        ll = ax.contour(x1,x2,T_pred,colors="grey")

    elif len(tvars) == 1:

        # Form prediciton + CIs
        x = x1_exp
        y = pred
        if ci:
            cu = pred + b
            cl = pred - b

        # transformation applied to ci boundaries - NOT b!
        if not link is None:
            y = link.fi(y)
            if ci:
                cu = link.fi(cu)
                cl = link.fi(cl)

        # Hide everything outside data limits
        if not pred_in_limits is None:
            x = x[pred_in_limits]
            y = y[pred_in_limits]
            if ci:
                cu = cu[pred_in_limits]
                cl = cl[pred_in_limits]

        if ci:
            ax.fill([*x,*np.flip(x)],
                    [*(cu),*np.flip(cl)],
                    color=_cmp(col),alpha=0.5)
            
        ax.plot(x,y,color=_cmp(col),label=legend_label)

        if not ylim is None:
            ax.set_ylim(ylim)

def plot(model:GAMM or GAMMLSS,which:[int] or None = None, dist_par=0, n_vals:int = 30,ci=None,
         ci_alpha=0.05,use_inter=False,whole_interval=False,n_ps=10000,seed=None,cmp:str or None = None,
         plot_exist=True,plot_exist_style='both',response_scale=False,axs=None,
         fig_size=(6/2.54,6/2.54),math_font_size = 9,math_font = 'cm',
         ylim=None,prov_cols=None,lim_dist=0.1):
    """Helper function to plot all smooth functions estimated by a `GAMM` or `GAMMLSS` model.

    Smooth functions are automatically evaluated over a range of ``n_values`` spaced equally to cover their entire covariate.
    For tensor smooths a ``n_values*n_values`` grid is created. Visualizations can be obtained on the scale of the linear predictor (the default), but also
    on what is often referred to as the 'response'-scale - corresponding to the estimated mean of the RVs modeled by the model. Note, that CIs will then only be approximate.

    To simply obtain visualizations of all smooth terms estimated, it is sufficient to call::

        plot(model) # or plot(model,dist_par=0) in case of a GAMMLSS model
    
    This will visualize all smooth terms estimated by the model and automatically determine whether confidence intervals should be drawn or not (by default CIs
    are only visualized for fixed effects). Note that, for tensor smooths, areas of the smooth for which the CI contains zero will be visualized with low opacity
    if the CI is to be visualized.

    References:

    - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
    - Simpson, G. (2016). Simultaneous intervals for smooths revisited.

    :param model: The estimated GAMM or GAMMLSS model for which the visualizations are to be obtained
    :type model: GAMM or GAMMLSS
    :param which: The indices corresponding to the smooth that should be visualized or ``None`` in which case all smooth terms will be visualized, defaults to None
    :type which: [int] or None, optional
    :param dist_par: The index corresponding to the parameter for which to make the prediction (e.g., 0 = mean) - only necessary if a GAMMLSS model is provided, defaults to 0
    :type dist_par: int, optional
    :param n_vals: Number of covariate values over which to evaluate the function. Will result in ``n_vals**2`` eval points for tensor smooths, defaults to 30
    :type n_vals: int, optional
    :param ci: Whether the standard error ``se`` for credible interval (CI; see  Wood, 2017) calculation should be computed and used to visualize CIs. The CI is then [``pred`` - ``se``, ``pred`` + ``se``], defaults to None in which case the CI will be visualized for fixed effects but not for random smooths
    :type ci: bool or None, optional
    :param ci_alpha: The alpha level to use for the standard error calculation. Specifically, 1 - (``alpha``/2) will be used to determine the critical cut-off value according to a N(0,1), defaults to 0.05
    :type ci_alpha: float, optional
    :param use_inter: Whether or not the standard error for CIs should be computed based on just the smooth or based on the smooth + the model intercept - the latter results in better coverage for strongly penalized functions (see Wood, 2017), defaults to False
    :type use_inter: bool, optional
    :param whole_interval: Whether or not to adjuste the point-wise CI to behave like whole-function (based on Wood, 2017; section 6.10.2 and Simpson, 2016), defaults to False
    :type whole_interval: bool, optional
    :param n_ps: How many samples to draw from the posterior in case the point-wise CI is adjusted to behave like a whole-function CI, defaults to 10000
    :type n_ps: int, optional
    :param seed: Can be used to provide a seed for the posterior sampling step in case the point-wise CI is adjusted to behave like a whole-function CI, defaults to None
    :type seed: int, optional
    :param cmp: string corresponding to name for a matplotlib colormap, defaults to None in which case it will be set to 'RdYlBu_r'.
    :type cmp: str or None, optional
    :param plot_exist: Whether or not an indication of the data distribution should be provided. For univariate smooths setting this to True will add a rug-plot to the bottom, indicating for which covariate values samples existed in the training data. For tensor smooths setting this to true will result in a 2d scatter rug plot being added and/or values outside of data limits being hidden, defaults to True
    :type plot_exist: bool, optional
    :param plot_exist_style: Determines the style of the data distribution indication for smooths. Must be 'rug', 'hide',or 'both'. 'both' will both add the rug-plot and hide values out of data limits, defaults to 'both'
    :type plot_exist_style: str, optional
    :param response_scale: Whether or not predictions and CIs should be shown on the scale of the model predictions (linear scale) or on the 'response-scale' i.e., the scale of the mean, defaults to False
    :type response_scale: bool, optional
    :param axs: A list of matplotlib.axis on which Figures should be drawn, defaults to None in which case axis will be created by the function and plot.show() will be called at the end
    :type axs: [matplotlib.axis], optional
    :param fig_size: Tuple holding figure size, which will be used to determine the size of the figures created if `axs=None`, defaults to (6/2.54,6/2.54)
    :type fig_size: tuple, optional
    :param math_font_size: Font size for math notation, defaults to 9
    :type math_font_size: int, optional
    :param math_font: Math font to use, defaults to 'cm'
    :type math_font: str, optional
    :param ylim: Tuple holding y-limits (z-limits for 2d plots), defaults to None in which case y_limits will be inferred from the predictions made
    :type ylim: (float,float), optional
    :param prov_cols: A float or a list (in case of a smooth with a `by` argument) of floats in [0,1]. Used to get a color for unicariate smooth terms, defaults to None in which case colors will be selected automatically depending on whether the smooth has a `by` keyword or not
    :type prov_cols: float or [float], optional
    :param lim_dist: The floating point distance (on normalized scale, i.e., values have to be in ``[0,1]``) at which a point is considered too far away from training data. Setting this to 0 means we visualize only points for which there is trainings data, setting this to 1 means visualizing everything. Defaults to 0.1 
    :type lim_dist: float, optional
    :raises ValueError: If fewer matplotlib axis are provided than the number of figures that would be created
    """

    if isinstance(model,GAMMLSS):
        # Set up everything for that we can plot all smooth terms for distribution parameter `dist_par`.
        model.formula = model.formulas[dist_par]
    
    # Get all necessary information from the model formula
    terms = model.formula.get_terms()
    stidx = model.formula.get_smooth_term_idx()

    varmap = model.formula.get_var_map()
    vartypes = model.formula.get_var_types()
    varmins = model.formula.get_var_mins()
    varmaxs = model.formula.get_var_maxs()
    code_factors = model.formula.get_coding_factors()
    factor_codes = model.formula.get_factor_codings()

    # Default colormap
    if cmp is None:
        cmp = 'RdYlBu_r'
        
    _cmp = matplotlib.colormaps[cmp]

    if not which is None:
        stidx = which

    # Check number of figures matches axis
    n_figures = 0
    for sti in stidx:
        if isinstance(terms[sti],fs):
            n_figures +=1
        else:
            if not terms[sti].by is None:
                n_figures += len(code_factors[terms[sti].by])

            else:
                n_figures += 1
    
    if not axs is None and len(axs) != n_figures:
        raise ValueError(f"{n_figures} plots would be created, but only {len(axs)} axes were provided!")

    # if nothing is provided, create figures + axis
    figs = None
    if axs is None:
        figs = [plt.figure(figsize=fig_size,layout='constrained') for _ in range(n_figures)]
        axs = [fig.add_subplot(1,1,1) for fig in figs]
    
    axi = 0

    for sti in stidx:

        # Start by generating prediction data for the current smooth term.
        tvars = terms[sti].variables
        pred_dat = {}
        x1_exp = []
        if len(tvars) == 2:
            # Set up a grid of n_vals*n_vals
            x1 = np.linspace(varmins[tvars[0]],varmaxs[tvars[0]],n_vals)
            x2 = np.linspace(varmins[tvars[1]],varmaxs[tvars[1]],n_vals)

            x2_exp = []

            for x1v in x1:
                x1_exp.extend([x1v for _ in range(n_vals)])
                x2_exp.extend(x2)
            
            pred_dat[tvars[0]] = x1_exp
            pred_dat[tvars[1]] = x2_exp
        
        elif len(tvars) == 1:
            x1 = None
            x2 = None
            # Simply set up x1_exp directly.
            x1_exp = np.linspace(varmins[tvars[0]],varmaxs[tvars[0]],n_vals)
            pred_dat[tvars[0]] = x1_exp
        else:
            continue
        
        # Now fill the data used for prediction with placeholders for all other variables
        # included in the model. These will be ignored for the prediction.
        if terms[sti].by is None and terms[sti].binary is None:
            for vari in varmap.keys():
                if vari in terms[sti].variables:
                    continue
                else:
                    if vartypes[vari] == VarType.FACTOR:
                        if vari in model.formula.get_subgroup_variables():
                            pred_dat[vari.split(":")[0]] = [code_factors[vari][0] for _ in range(len(x1_exp))]
                        else:
                            pred_dat[vari] = [code_factors[vari][0] for _ in range(len(x1_exp))]
                    else:
                        if terms[sti].by_cont is not None and vari == terms[sti].by_cont:
                            pred_dat[vari] = [1 for _ in range(len(x1_exp))]
                        else:
                            pred_dat[vari] = [0 for _ in range(len(x1_exp))]
            
            pred_dat_pd = pd.DataFrame(pred_dat)
            
            use_ci = ci
            if use_ci is None:
                use_ci = True

            # Add intercept for prediction - remember to subtract it later
            use = [sti]
            if use_inter:
                use = [0,sti]

            if isinstance(model,GAMMLSS):
                # Reset formula to prevent any problems with the call to predict, since the GAMMLSS class might
                # change this attribute itself.
                model.formula = None
                
                pred,_,b= model.predict(dist_par,use,pred_dat_pd,ci=use_ci,alpha=ci_alpha,whole_interval=whole_interval,n_ps=n_ps,seed=seed)

                # Set formula again.
                model.formula = model.formulas[dist_par]
            else:
                pred,_,b= model.predict(use,pred_dat_pd,ci=use_ci,alpha=ci_alpha,whole_interval=whole_interval,n_ps=n_ps,seed=seed)

            # Subtract intercept from prediction - it was just used to adjust se
            if use_inter:
                _cf,_ = model.get_pars()
                pred -= _cf[0]

            # Compute data limits and anything needed for rug plot
            plot_in_limits = None
            if plot_exist:
                pred_in_limits,train_unq,train_unq_counts,cont_vars = __get_data_limit_counts(model.formula,pred_dat_pd,tvars,None,terms[sti].by_cont,lim_dist)

            if plot_exist and (plot_exist_style == "both" or plot_exist_style == "hide"):
                plot_in_limits = pred_in_limits
            
            # Prepare link to transform prediction + ci to response-scale
            link = None
            if response_scale:
                if isinstance(model,GAMMLSS):
                    link = model.family.links[dist_par]
                else:
                    link = model.family.link

            # Now plot
            __pred_plot(pred,b,tvars,plot_in_limits,x1,x2,x1_exp,use_ci,n_vals,axs[axi],_cmp,0.7 if prov_cols is None else prov_cols,ylim,link,None)

            # Specify labels and add rug plots if requested
            if plot_in_limits is None:
                vmin = ylim[0] if ylim is not None else np.min(pred)
                vmax = ylim[1] if ylim is not None else np.max(pred)
            else:
                vmin = ylim[0] if ylim is not None else np.min(pred[plot_in_limits])
                vmax = ylim[1] if ylim is not None else np.max(pred[plot_in_limits])

            ticks = np.linspace(vmin,vmax,5)

            if len(tvars) == 1:
                y_lab = "$f_{" + str(terms[sti].by_cont) + "}" if terms[sti].by_cont is not None else '$f'
                axs[axi].set_ylabel(y_lab + '(' + tvars[0] + ')$',math_fontfamily=math_font,size=math_font_size,fontweight='bold')
                axs[axi].set_xlabel(tvars[0],fontweight='bold')
                axs[axi].spines['top'].set_visible(False)
                axs[axi].spines['right'].set_visible(False)
                axs[axi].set_yticks(ticks)
                axs[axi].set_yticklabels([f"{tick: .2f}" for tick in ticks])

                if plot_exist:
                    
                    #train_unq_counts[train_unq_counts > 0] = 1 
                    #pred_range = np.abs(np.max(pred) - np.min(pred))*0.025
                    x_counts = np.ndarray.flatten(train_unq[:,[cvar ==tvars[0] for cvar in cont_vars]])
                    #x_range = np.abs(np.max(x_counts) - np.min(x_counts))
                    
                    axs[axi].scatter(x_counts,[axs[axi].get_ylim()[0]]*len(x_counts),marker='|',color='black',linewidths=0.25)
            
            elif len(tvars) == 2:
                axs[axi].set_ylabel(tvars[1],fontweight='bold')
                axs[axi].set_xlabel(tvars[0],fontweight='bold')
                axs[axi].set_box_aspect(1)
                
                if plot_exist and (plot_exist_style == "both" or plot_exist_style == 'rug'):
                    train_unq_counts[train_unq_counts > 0] = 0.1
                    x_counts = np.ndarray.flatten(train_unq[:,[cvar ==tvars[0] for cvar in cont_vars]])
                    y_counts = np.ndarray.flatten(train_unq[:,[cvar ==tvars[1] for cvar in cont_vars]])
                    tot_range = np.abs(max(np.max(x_counts),np.max(y_counts)) - min(np.min(x_counts),np.min(y_counts)))
                    axs[axi].scatter(x_counts,y_counts,alpha=train_unq_counts,color='black',s=tot_range/(len(x_counts)))

                # Credit to Lasse: https://stackoverflow.com/questions/63118710/
                # This made sure that the colorbar height always matches those of the contour plots.
                axins = inset_axes(axs[axi], width = "5%", height = "100%", loc = 'lower left',
                        bbox_to_anchor = (1.02, 0., 1, 1), bbox_transform = axs[axi].transAxes,
                        borderpad = 0)
                
                if use_ci:
                    cbar = plt.colorbar(axs[axi].collections[1],cax=axins)
                else:
                    cbar = plt.colorbar(axs[axi].collections[0],cax=axins)

                cbar.set_ticks(ticks)

                cbar.ax.set_yticklabels([f"{tick: .2f}" for tick in ticks])

                cbar_label_pre = "$f_{" + str(terms[sti].by_cont) + "}" if terms[sti].by_cont is not None else '$f'
                cbar_label = cbar_label_pre + '(' + tvars[0] + ',' + tvars[1] + ')$'

                cbar.ax.set_ylabel(cbar_label,math_fontfamily=math_font,size=math_font_size)

            axi += 1

        # Now handle by terms - essentially we need to perform the above separately for every level
        # of the by/binary factor.
        elif not terms[sti].by is None or not terms[sti].binary is None:
            
            if not terms[sti].by is None:
                sti_by = terms[sti].by
            else:
                sti_by = terms[sti].binary[0]

            levels = list(code_factors[sti_by].keys())

            if not terms[sti].binary is None:
                levels = [factor_codes[sti_by][terms[sti].binary[1]]]

            # Select a small set of levels for random smooths
            if isinstance(terms[sti],fs):
                ymin = np.finfo(float).max
                ymax = np.finfo(float).min
                
                if len(levels) > 25:
                    levels = np.random.choice(levels,replace=False,size=25)

            if prov_cols is None:
                level_cols = np.linspace(0.1,0.9,len(levels))
            else:
                level_cols = prov_cols

            for level_col,leveli in zip(level_cols,levels):
                pred_level_dat = copy.deepcopy(pred_dat)

                for vari in varmap.keys():
                    if vari in terms[sti].variables:
                        continue
                    else:
                        # Note, placeholder selection must exlcude by/binary variable for which we need to provide the
                        # current level!
                        if vartypes[vari] == VarType.FACTOR and vari == sti_by:
                            if vari in model.formula.get_subgroup_variables():
                                pred_level_dat[vari.split(":")[0]] = [code_factors[vari][leveli] for _ in range(len(x1_exp))]
                            else:
                                pred_level_dat[vari] = [code_factors[vari][leveli] for _ in range(len(x1_exp))]
                        elif vartypes[vari] == VarType.FACTOR:
                            if vari in model.formula.get_subgroup_variables():
                                if sti_by in model.formula.get_subgroup_variables() and sti_by.split(":")[0] == vari.split(":")[0]:
                                    continue
                                
                                pred_level_dat[vari.split(":")[0]] = [code_factors[vari][0] for _ in range(len(x1_exp))]
                            else:
                                pred_level_dat[vari] = [code_factors[vari][0] for _ in range(len(x1_exp))]
                        else:
                            if terms[sti].by_cont is not None and vari == terms[sti].by_cont:
                                pred_level_dat[vari] = [1 for _ in range(len(x1_exp))]
                            else:
                                pred_level_dat[vari] = [0 for _ in range(len(x1_exp))]
                
                pred_dat_pd = pd.DataFrame(pred_level_dat)

                # CI-decision - exclude factor smooths if not requested explicitly.
                use_ci = ci
                if use_ci is None:
                    if not isinstance(terms[sti],fs):
                        use_ci = True
                    else:
                        use_ci = False
                
                # Again, add intercept
                use = [sti]
                if use_inter:
                    use = [0,sti]

                if isinstance(model,GAMMLSS):
                    # Reset formula to prevent any problems with the call to predict, since the GAMMLSS class might
                    # change this attribute itself.
                    model.formula = None
                    
                    pred,_,b= model.predict(dist_par,use,pred_dat_pd,ci=use_ci,alpha=ci_alpha,whole_interval=whole_interval,n_ps=n_ps,seed=seed)

                    # Set formula again.
                    model.formula = model.formulas[dist_par]
                else:
                    pred,_,b= model.predict(use,pred_dat_pd,ci=use_ci,alpha=ci_alpha,whole_interval=whole_interval,n_ps=n_ps,seed=seed)

                # Subtract intercept
                if use_inter:
                    _cf,_ = model.get_pars()
                    pred -= _cf[0]

                # Compute data-limits and prepare rug plots
                plot_in_limits = None
                if plot_exist:
                    pred_in_limits,train_unq,train_unq_counts,cont_vars = __get_data_limit_counts(model.formula,pred_dat_pd,tvars,[sti_by],terms[sti].by_cont,lim_dist)

                if plot_exist and (plot_exist_style == "both" or plot_exist_style == "hide"):
                    plot_in_limits = pred_in_limits

                # Get correct link to transform to response scale
                link = None
                if response_scale:
                    if isinstance(model,GAMMLSS):
                        link = model.family.links[dist_par]
                    else:
                        link = model.family.link

                __pred_plot(pred,b,tvars,plot_in_limits,x1,x2,x1_exp,use_ci,n_vals,axs[axi],_cmp,level_col,ylim,link,None)
                
                # And set up labels again + rug plots if requested
                if isinstance(terms[sti],fs):
                    ymin = min(ymin,np.min(pred))
                    ymax = max(ymax,np.max(pred))

                else:
                    if plot_in_limits is None:
                        vmin = ylim[0] if ylim is not None else np.min(pred)
                        vmax = ylim[1] if ylim is not None else np.max(pred)
                    else:
                        vmin = ylim[0] if ylim is not None else np.min(pred[plot_in_limits])
                        vmax = ylim[1] if ylim is not None else np.max(pred[plot_in_limits])
                    ticks = np.linspace(vmin,vmax,5)

                    if len(tvars) == 1:
                        ax_label = '$f_{' + str(code_factors[sti_by][leveli]) + '}' + '(' + tvars[0] + ')$'
                        axs[axi].set_ylabel(ax_label,math_fontfamily=math_font,size=math_font_size,fontweight='bold')
                        axs[axi].set_xlabel(tvars[0],fontweight='bold')
                        axs[axi].spines['top'].set_visible(False)
                        axs[axi].spines['right'].set_visible(False)
                        axs[axi].set_yticks(ticks)
                        axs[axi].set_yticklabels([f"{tick: .2f}" for tick in ticks])
                        
                        if plot_exist and (plot_exist_style == "both" or plot_exist_style == 'rug'):
                    
                            #train_unq_counts /= np.max(train_unq_counts)
                            #train_unq_counts[train_unq_counts > 0] = 1 
                            #pred_range = np.abs(np.max(pred) - np.min(pred))*0.01
                            x_counts = np.ndarray.flatten(train_unq[:,[cvar ==tvars[0] for cvar in cont_vars]])
                            #x_range = np.abs(np.max(x_counts) - np.min(x_counts))

                            #axs[axi].bar(x=x_counts,bottom=axs[axi].get_ylim()[0],height=pred_range*train_unq_counts,color='black',width=max(0.05,x_range/(2*len(x_counts))))
                            axs[axi].scatter(x_counts,[axs[axi].get_ylim()[0]]*len(x_counts),marker='|',color='black',linewidths=0.25)


                    elif len(tvars) == 2:
                        axs[axi].set_ylabel(tvars[1],fontweight='bold')
                        axs[axi].set_xlabel(tvars[0],fontweight='bold')
                        axs[axi].set_box_aspect(1)

                        if plot_exist and (plot_exist_style == "both" or plot_exist_style == 'rug'):

                            train_unq_counts[train_unq_counts > 0] = 0.1
                            x_counts = np.ndarray.flatten(train_unq[:,[cvar ==tvars[0] for cvar in cont_vars]])
                            y_counts = np.ndarray.flatten(train_unq[:,[cvar ==tvars[1] for cvar in cont_vars]])
                            tot_range = np.abs(max(np.max(x_counts),np.max(y_counts)) - min(np.min(x_counts),np.min(y_counts)))
                            axs[axi].scatter(x_counts,y_counts,alpha=train_unq_counts,color='black',s=tot_range/(len(x_counts)))

                        # Credit to Lasse: https://stackoverflow.com/questions/63118710/
                        # This made sure that the colorbar height always matches those of the contour plots.
                        axins = inset_axes(axs[axi], width = "5%", height = "100%", loc = 'lower left',
                                bbox_to_anchor = (1.02, 0., 1, 1), bbox_transform = axs[axi].transAxes,
                                borderpad = 0)
                        
                        with warnings.catch_warnings(): # Overflow
                            warnings.simplefilter("ignore")
                            if use_ci:
                                cbar = plt.colorbar(axs[axi].collections[1],cax=axins)
                            else:
                                cbar = plt.colorbar(axs[axi].collections[0],cax=axins)

                        cbar.set_ticks(ticks)
                        
                        cbar.ax.set_yticklabels([f"{tick: .2f}" for tick in ticks])

                        cbar_label = '(' + tvars[0] + ',' + tvars[1] + ')$'

                        cbar_label = '$f_{' + str(code_factors[sti_by][leveli]) + '}' + cbar_label

                        cbar.ax.set_ylabel(cbar_label,math_fontfamily=math_font,size=math_font_size)
                    axi += 1

            # Random smooths are all plotted to single figure, so handle labels here. No reason to plot rug
            if isinstance(terms[sti],fs):
                vmin = ylim[0] if ylim is not None else ymin
                vmax = ylim[1] if ylim is not None else ymax
                ticks = np.linspace(vmin,vmax,5)
                
                axs[axi].set_ylabel('$f_{' + str(sti_by) + '}(' + tvars[0] + ')$',math_fontfamily=math_font,size=math_font_size,fontweight='bold')
                axs[axi].set_xlabel(tvars[0],fontweight='bold')
                axs[axi].spines['top'].set_visible(False)
                axs[axi].spines['right'].set_visible(False)
                axs[axi].set_yticks(ticks)
                axs[axi].set_yticklabels([f"{tick: .2f}" for tick in ticks])
                axi += 1
    
    if isinstance(model,GAMMLSS):
        # Clean up
        model.formula = None

    if figs is not None:
        plt.show()


def plot_fitted(pred_dat,tvars,model:GAMM or GAMMLSS,use:[int] or None = None,pred_factors: [str] or None =None,dist_par=0,
                ci=True,ci_alpha=0.05,whole_interval=False,n_ps=10000,seed=None,
                cmp:str or None = None,plot_exist=True,plot_exist_style='both',
                response_scale=True,ax=None,fig_size=(6/2.54,6/2.54),ylim=None,col=0.7,
                label=None,legend_label=False,title=None,lim_dist=0.1):
    """Plots the model prediction based on (a subset of) the terms included in the model for new data `pred_dat`.

    In contrast to `plot`, the predictions are by default transformed to the scale of the mean (i.e., response-scale). If `use=None`, the model
    will simply use all parametric and regular smooth terms (but no random effects) for the prediction (i.e., only the "fixed" effects in the model).

    For a GAMM, a simple example of this function would be::

        # Fit model
        model = GAMM(Formula(lhs("y"),[i(),f(["time"])],data=dat),Gaussian())
        model.fit()

        # Create prediction data
        pred_dat = pd.DataFrame({"time":np.linspace(0,np.max(dat["time"]),30)})

        # Plot predicted mean = \alpha + f(time)
        plot_fitted(pred_dat,["time"],model)

        # This is in contrast to `plot`, which would just visualize pred = f(time)
        plot(model)

    Note that, for predictions visualized as a function of two variables, areas of the prediction for which the CI contains zero will again be visualized with low opacity
    if the CI is to be visualized.
    
    References:

    - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
    - Simpson, G. (2016). Simultaneous intervals for smooths revisited.

    :param pred_dat: A pandas DataFrame containing new data for which to make the prediction. Importantly, all variables present in the data used to fit the model also need to be present in this DataFrame. Additionally, factor variables must only include levels also present in the data used to fit the model. If you want to exclude a specific factor from the prediction (for example the factor subject) don't include the terms that involve it in the ``use`` argument.
    :type pred_dat: pandas.DataFrame
    :param tvars: List of variables to be visualized - must contain one string for predictions visualized as a function of a single variable, two for predictions visualized as a function of two variables
    :type tvars: [str]
    :param model: The estimated GAMM or GAMMLSS model for which the visualizations are to be obtained
    :type model: GAMM or GAMMLSS
    :param use: The indices corresponding to the terms that should be used to obtain the prediction or ``None`` in which case all fixed effects will be used, defaults to None
    :type use: [int] or None, optional
    :param pred_factors: List of factor variables to consider for data limit/availability computations - by default, all factor variables in the model are considered.
    :type pred_factors: [str]
    :param dist_par: The index corresponding to the parameter for which to make the prediction (e.g., 0 = mean) - only necessary if a GAMMLSS model is provided, defaults to 0
    :type dist_par: int, optional
    :param ci: Whether the standard error ``se`` for credible interval (CI; see  Wood, 2017) calculation should be computed and used to visualize CIs. The CI is then [``pred`` - ``se``, ``pred`` + ``se``], defaults to None in which case the CI will be visualized for fixed effects but not for random smooths
    :type ci: bool or None, optional
    :param ci_alpha: The alpha level to use for the standard error calculation. Specifically, 1 - (``alpha``/2) will be used to determine the critical cut-off value according to a N(0,1), defaults to 0.05
    :type ci_alpha: float, optional
    :param whole_interval: Whether or not to adjuste the point-wise CI to behave like whole-function (based on Wood, 2017; section 6.10.2 and Simpson, 2016), defaults to False
    :type whole_interval: bool, optional
    :param n_ps: How many samples to draw from the posterior in case the point-wise CI is adjusted to behave like a whole-function CI, defaults to 10000
    :type n_ps: int, optional
    :param seed: Can be used to provide a seed for the posterior sampling step in case the point-wise CI is adjusted to behave like a whole-function CI, defaults to None
    :type seed: int, optional
    :param cmp: string corresponding to name for a matplotlib colormap, defaults to None in which case it will be set to 'RdYlBu_r'.
    :type cmp: str or None, optional
    :param plot_exist: Whether or not an indication of the data distribution should be provided. For predictions visualized as a function of a single variable setting this to True will add a rug-plot to the bottom, indicating for which covariate values samples existed in the training data. For predictions visualized as a function of a two variables setting this to true will result in a 2d scatter rug plot being added and/or values outside of data limits being hidden, defaults to True
    :type plot_exist: bool, optional
    :param plot_exist_style: Determines the style of the data distribution indication for smooths. Must be 'rug', 'hide',or 'both'. 'both' will both add the rug-plot and hide values out of data limits, defaults to 'both'
    :type plot_exist_style: str, optional
    :param response_scale: Whether or not predictions and CIs should be shown on the scale of the model predictions (linear scale) or on the 'response-scale' i.e., the scale of the mean, defaults to True
    :type response_scale: bool, optional
    :param ax: A matplotlib.axis on which the Figure should be drawn, defaults to None in which case an axis will be created by the function and plot.show() will be called at the end
    :type ax: matplotlib.axis, optional
    :param fig_size: Tuple holding figure size, which will be used to determine the size of the figures created if `ax=None`, defaults to (6/2.54,6/2.54)
    :type fig_size: tuple, optional
    :param ylim: Tuple holding y-limits (z-limits for 2d plots), defaults to None in which case y_limits will be inferred from the predictions made
    :type ylim: (float,float), optional
    :param col: A float in [0,1]. Used to get a color for univariate predictions from the chosen colormap, defaults to 0.7
    :type col: float, optional
    :param label: A label to add to the y axis for univariate predictions (or to a legend, if `legend_label`=True) or to the color-bar for tensor predictions, defaults to None
    :type label: str, optional
    :param legend_label: Whether or not any `label` should be added to a legend (don't forget to call :func:`plt.legend()`) or to the y-axis for univariate predicitions, defaults to False (the latter)
    :type legend_label: bool, optional
    :param title: A list of titles to add to each plot, defaults to None
    :type title: [str], optional
    :param lim_dist: The floating point distance (on normalized scale, i.e., values have to be in ``[0,1]``) at which a point is considered too far away from training data. Setting this to 0 means we visualize only points for which there is trainings data, setting this to 1 means visualizing everything. Defaults to 0.1 
    :type lim_dist: float, optional
    :raises ValueError: If a visualization is requested for more than 2 variables
    """
    
    if isinstance(model,GAMMLSS):
        # Set up everything for that we can plot all smooth terms for distribution parameter `dist_par`.
        model.formula = model.formulas[dist_par]
    
    # Select only fixed effects if nothing is provided
    if use is None:
        use = model.formula.get_linear_term_idx()

        terms = model.formula.get_terms()
        for sti in model.formula.get_smooth_term_idx():
            if not isinstance(terms[sti],fs):
                use.append(sti)
    
    # Create figure if necessary
    fig = None
    if ax is None:
        fig = plt.figure(figsize=fig_size,layout='constrained')
        ax = fig.add_subplot(1,1,1)

    # Sort data if len(tvars) == 2, before getting variables in next step
    # to prevent ambiguity.
    if len(tvars) == 2:
        pred_dat = pred_dat.sort_values([tvars[0], tvars[1]], ascending=[True, True])
    
    # Set up predictor variables as done in `plot`
    x1_exp = np.array(pred_dat[tvars[0]])
    x1 = np.unique(x1_exp)
    x2 = None
    if len(tvars) == 2:
        x2 = np.unique(pred_dat[tvars[1]])

    elif len(tvars) > 2:
        raise ValueError("Can only visualize fitted effects over one or two continuous variables.")
    
    if cmp is None:
        cmp = 'RdYlBu_r'
        
    _cmp = matplotlib.colormaps[cmp] 

    if isinstance(model,GAMMLSS):
        # Reset formula to prevent any problems with the call to predict, since the GAMMLSS class might
        # change this attribute itself.
        model.formula = None
        
        pred,_,b= model.predict(dist_par,use,pred_dat,ci=ci,alpha=ci_alpha,whole_interval=whole_interval,n_ps=n_ps,seed=seed)

        # Set formula again.
        model.formula = model.formulas[dist_par]
    else:
        pred,_,b= model.predict(use,pred_dat,ci=ci,alpha=ci_alpha,whole_interval=whole_interval,n_ps=n_ps,seed=seed)

    # Optionally get data limits
    plot_in_limits = None
    if plot_exist:
        if pred_factors is None:
            pred_factors = [var for var in pred_dat.columns if model.formula.get_var_types()[var] == VarType.FACTOR]
        if (not pred_factors is None) and len(pred_factors) == 0:
            pred_factors = None

        pred_in_limits,train_unq,train_unq_counts,cont_vars = __get_data_limit_counts(model.formula,pred_dat,tvars,pred_factors,None,lim_dist)

    if plot_exist and (plot_exist_style == "both" or plot_exist_style == "hide"):
        plot_in_limits = pred_in_limits

    # By default transform predictions to scale of mean
    link = None
    if response_scale:
        if isinstance(model,GAMMLSS):
            link = model.family.links[dist_par]
        else:
            link = model.family.link
    
    plot_label = None
    if legend_label and len(tvars) == 1:
        plot_label = label

    __pred_plot(pred,b,tvars,plot_in_limits,x1,x2,x1_exp,ci,len(x1),ax,_cmp,col,ylim,link,plot_label)

    # Label axes + visualize rug plots if requested
    if plot_in_limits is None:
        vmin = ylim[0] if ylim is not None else np.min(pred)
        vmax = ylim[1] if ylim is not None else np.max(pred)
    else:
        vmin = ylim[0] if ylim is not None else np.min(pred[plot_in_limits])
        vmax = ylim[1] if ylim is not None else np.max(pred[plot_in_limits])
    ticks = np.linspace(vmin,vmax,5)

    if len(tvars) == 2:
       
        if plot_exist and (plot_exist_style == "both" or plot_exist_style == 'rug'):

            train_unq_counts[train_unq_counts > 0] = 0.1
            x_counts = np.ndarray.flatten(train_unq[:,[cvar ==tvars[0] for cvar in cont_vars]])
            y_counts = np.ndarray.flatten(train_unq[:,[cvar ==tvars[1] for cvar in cont_vars]])
            tot_range = np.abs(max(np.max(x_counts),np.max(y_counts)) - min(np.min(x_counts),np.min(y_counts)))
            ax.scatter(x_counts,y_counts,alpha=train_unq_counts,color='black',s=tot_range/(len(x_counts)))

        # Credit to Lasse: https://stackoverflow.com/questions/63118710/
        # This made sure that the colorbar height always matches those of the contour plots.
        axins = inset_axes(ax, width = "5%", height = "100%", loc = 'lower left',
                bbox_to_anchor = (1.02, 0., 1, 1), bbox_transform = ax.transAxes,
                borderpad = 0)
        
        if ci:
            cbar = plt.colorbar(ax.collections[1],cax=axins)
        else:
            cbar = plt.colorbar(ax.collections[0],cax=axins)

        cbar.set_ticks(ticks)

        cbar.ax.set_yticklabels([f"{tick: .2f}" for tick in ticks])

        if not label is None:
            cbar.set_label(label,fontweight='bold')
        else:
            cbar.set_label("Predicted",fontweight='bold')

        ax.set_xlabel(tvars[0],fontweight='bold')
        ax.set_ylabel(tvars[1],fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    else:
        if not label is None and (legend_label==False):
            ax.set_ylabel(label,fontweight='bold')
        else:
            ax.set_ylabel("Predicted",fontweight='bold')
        ax.set_xlabel(tvars[0],fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{tick: .2f}" for tick in ticks])

        if plot_exist and (plot_exist_style == "both" or plot_exist_style == 'rug'):
                    
            #train_unq_counts[train_unq_counts > 0] = 1 
            #pred_range = np.abs(np.max(pred) - np.min(pred))*0.025
            x_counts = np.ndarray.flatten(train_unq[:,[cvar ==tvars[0] for cvar in cont_vars]])
            #x_range = np.abs(np.max(x_counts) - np.min(x_counts))

            ax.scatter(x_counts,[ax.get_ylim()[0]]*len(x_counts),marker='|',color='black',linewidths=0.25)
    
    if not title is None:
        ax.set_title(title,fontweight='bold')

    if isinstance(model,GAMMLSS):
        # Clean up
        model.formula = None

    if fig is not None:
        plt.show()

def plot_diff(pred_dat1,pred_dat2,tvars,model: GAMM or GAMMLSS,use:[int] or None = None,dist_par=0,
              ci_alpha=0.05,whole_interval=False,n_ps=10000,seed=None,cmp:str or None = None,
              plot_exist=True,response_scale=True,ax=None,fig_size=(6/2.54,6/2.54),
              ylim=None,col=0.7,label=None,title=None,lim_dist=0.1):
    """Plots the expected difference (and CI around this expected difference) between two sets of predictions, evaluated for `pred_dat1` and `pred_dat2`.

    This function is primarily designed to visualize the expected difference between two levels of a categorical/factor variable. For example, consider the following
    model below, including a separate smooth of "time" per level of the factor "cond". It is often of interest to visualize *when* in time the two levels of "cond" differ
    from each other in their dependent variable. For this, the difference curve over "time", essentially the smooth of "time" for the first level subtracted from the
    smooth of "time" for the second level of factor "cond" (offset terms can also be accounted for, check the `use` argument), can be visualized together with a CI (Wood, 2017).
    This CI can provide insights into whether and *when* the two levels can be expected to be different from each other. To visualize this difference curve as well as the
    difference CI, this function can be used as follows::

        # Define & estimate model
        model = GAMM(Formula(lhs("y"),[i(), l(["cond"]), f(["time"],by="cond")],data=dat),Gaussian())
        model.fit()

        # Create prediction data, differing only in the level of factor cond
        time_pred = np.linspace(0,np.max(dat["time"]),30)
        new_dat1 = pd.DataFrame({"cond":["a" for _ in range(len(time_pred))],
                                "time":time_pred})

        new_dat2 = pd.DataFrame({"cond":["b" for _ in range(len(time_pred))],
                                "time":time_pred})

        # Now visualize diff = (\alpha_a + f_a(time)) - (\alpha_b + f_b(time)) and the CI around diff
        plot_diff(pred_dat1,pred_dat2,["time"],model)

    This is only the most basic example to illustrate the usefulness of this function. Many other options are possible. Consider for example the model below, which allows for
    the expected time-course to vary smoothly as a function of additional covariate "x" - achieved by inclusion of the tensor smooth term of "time" and "x". In addition, this
    model allows for the shape of the tensor smooth to differ between the levels of factor "cond"::

        model = GAMM(Formula(lhs("y"),[i(), l(["cond"]), f(["time","x"],by="cond",te=True)],data=dat),Gaussian())

    For such a model, multiple predicted differences might be of interest. One option would be to look only at a single level of "cond" and to visualize the predicted difference
    in the time-course for two different values of "x" (perhaps two quantiles). In that case, `pred_dat1` and `pred_dat2` would have to be set up to differ only in the value of
    "x" - they should be equivalent in terms of "time" and "cond" values.

    Alternatively, it might be of interest to look at the predicted difference between the tensor smooth surfaces for two levels of factor "cond". Rather than being interested in
    a difference curve, this would mean we are interested in a difference *surface*. To achieve this, `pred_dat1` and `pred_dat2` would again have to be set up to differ only in the
    value of "cond" - they should be equivalent in terms of "time" and "x" values. In addition, it would be necessary to specify `tvars=["time","x"]`. Note that, for such difference surfaces,
    areas of the difference prediction for which the CI contains zero will again be visualized with low opacity if the CI is to be visualized.
    
    References:

    - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
    - Simpson, G. (2016). Simultaneous intervals for smooths revisited.

    :param pred_dat1: A pandas DataFrame containing new data for which the prediction is to be compared to the prediction obtained for `pred_dat2`. Importantly, all variables present in the data used to fit the model also need to be present in this DataFrame. Additionally, factor variables must only include levels also present in the data used to fit the model. If you want to exclude a specific factor from the difference prediction (for example the factor subject) don't include the terms that involve it in the ``use`` argument.
    :type pred_dat1: pandas.DataFrame
    :param pred_dat2: Like `pred_dat1` - ideally differing only in the level of a single factor variable or the value of a single continuous variable.
    :type pred_dat2: pandas.DataFrame
    :param tvars: List of variables to be visualized - must contain one string for difference predictions visualized as a function of a single variable, two for difference predictions visualized as a function of two variables
    :type tvars: [str]
    :param model: The estimated GAMM or GAMMLSS model for which the visualizations are to be obtained
    :type model: GAMM or GAMMLSS
    :param use: The indices corresponding to the terms that should be used to obtain the prediction or ``None`` in which case all fixed effects will be used, defaults to None
    :type use: [int] or None, optional
    :param dist_par: The index corresponding to the parameter for which to make the prediction (e.g., 0 = mean) - only necessary if a GAMMLSS model is provided, defaults to 0
    :type dist_par: int, optional
    :param ci_alpha: The alpha level to use for the standard error calculation. Specifically, 1 - (``alpha``/2) will be used to determine the critical cut-off value according to a N(0,1), defaults to 0.05
    :type ci_alpha: float, optional
    :param whole_interval: Whether or not to adjuste the point-wise CI to behave like whole-function (based on Wood, 2017; section 6.10.2 and Simpson, 2016), defaults to False
    :type whole_interval: bool, optional
    :param n_ps: How many samples to draw from the posterior in case the point-wise CI is adjusted to behave like a whole-function CI, defaults to 10000
    :type n_ps: int, optional
    :param seed: Can be used to provide a seed for the posterior sampling step in case the point-wise CI is adjusted to behave like a whole-function CI, defaults to None
    :type seed: int, optional
    :param cmp: string corresponding to name for a matplotlib colormap, defaults to None in which case it will be set to 'RdYlBu_r'.
    :type cmp: str or None, optional
    :param plot_exist: Whether or not an indication of the data distribution should be provided. For difference predictions visualized as a function of a single variable this will simply hide predictions outside of the data-limits. For difference predictions visualized as a function of a two variables setting this to true will result in values outside of data limits being hidden, defaults to True
    :type plot_exist: bool, optional
    :param response_scale: Whether or not predictions and CIs should be shown on the scale of the model predictions (linear scale) or on the 'response-scale' i.e., the scale of the mean, defaults to True
    :type response_scale: bool, optional
    :param ax: A matplotlib.axis on which the Figure should be drawn, defaults to None in which case an axis will be created by the function and plot.show() will be called at the end
    :type ax: matplotlib.axis, optional
    :param fig_size: Tuple holding figure size, which will be used to determine the size of the figures created if `ax=None`, defaults to (6/2.54,6/2.54)
    :type fig_size: tuple, optional
    :param ylim: Tuple holding y-limits (z-limits for 2d plots), defaults to None in which case y_limits will be inferred from the predictions made
    :type ylim: (float,float), optional
    :param col: A float in [0,1]. Used to get a color for univariate predictions from the chosen colormap, defaults to 0.7
    :type col: float, optional
    :param label: A list of label to add to the y axis for univariate predictions or to the color-bar for tensor predictions, defaults to None
    :type label: [str], optional
    :param title: A list of titles to add to each plot, defaults to None
    :type title: [str], optional
    :param lim_dist: The floating point distance (on normalized scale, i.e., values have to be in ``[0,1]``) at which a point is considered too far away from training data. Setting this to 0 means we visualize only points for which there is trainings data, setting this to 1 means visualizing everything. Defaults to 0.1 
    :type lim_dist: float, optional
    :raises ValueError: If a visualization is requested for more than 2 variables
    """
    
    if isinstance(model,GAMMLSS):
        # Set up everything for that we can plot all smooth terms for distribution parameter `dist_par`.
        model.formula = model.formulas[dist_par]

    if use is None:
        use = model.formula.get_linear_term_idx()

        terms = model.formula.get_terms()
        for sti in model.formula.get_smooth_term_idx():
            if not isinstance(terms[sti],fs):
                use.append(sti)

    fig = None
    if ax is None:
        fig = plt.figure(figsize=fig_size,layout='constrained')
        ax = fig.add_subplot(1,1,1)
    
    x1_exp = np.array(pred_dat1[tvars[0]])
    x1 = np.unique(x1_exp)
    x2 = None
    if len(tvars) == 2:
        x2 = np.unique(pred_dat1[tvars[1]])

    elif len(tvars) > 2:
        raise ValueError("Can only visualize fitted effects over one or two continuous variables.")
    
    if cmp is None:
        cmp = 'RdYlBu_r'
        
    _cmp = matplotlib.colormaps[cmp] 


    if isinstance(model,GAMMLSS):
        # Reset formula to prevent any problems with the call to predict, since the GAMMLSS class might
        # change this attribute itself.
        model.formula = None
        
        pred,_,b= model.predict_diff(pred_dat1,pred_dat2,dist_par,use,alpha=ci_alpha,whole_interval=whole_interval,n_ps=n_ps,seed=seed)

        # Set formula again.
        model.formula = model.formulas[dist_par]
    else:
        pred,b= model.predict_diff(pred_dat1,pred_dat2,use,alpha=ci_alpha,whole_interval=whole_interval,n_ps=n_ps,seed=seed)

    in_limits = None
    if plot_exist:
        pred_factors1 = [var for var in pred_dat1.columns if model.formula.get_var_types()[var] == VarType.FACTOR]
        if len(pred_factors1) == 0:
            pred_factors1 = None
        pred_in_limits1,train_unq1,train_unq_counts1,cont_vars1 = __get_data_limit_counts(model.formula,pred_dat1,tvars,pred_factors1,None,lim_dist)

        pred_factors2 = [var for var in pred_dat2.columns if model.formula.get_var_types()[var] == VarType.FACTOR]
        if len(pred_factors2) == 0:
            pred_factors2 = None
        pred_in_limits2,train_unq2,train_unq_counts2,cont_vars2 = __get_data_limit_counts(model.formula,pred_dat2,tvars,pred_factors2,None,lim_dist)

        in_limits = pred_in_limits1 & pred_in_limits2
    
    link = None
    if response_scale:
        if isinstance(model,GAMMLSS):
            link = model.family.links[dist_par]
        else:
            link = model.family.link

    __pred_plot(pred,b,tvars,in_limits,x1,x2,x1_exp,True,len(x1),ax,_cmp,col,ylim,link,None)

    if in_limits is None:
        vmin = ylim[0] if ylim is not None else np.min(pred)
        vmax = ylim[1] if ylim is not None else np.max(pred)
    else:
        vmin = ylim[0] if ylim is not None else np.min(pred[in_limits])
        vmax = ylim[1] if ylim is not None else np.max(pred[in_limits])
    ticks = np.linspace(vmin,vmax,5)

    if len(tvars) == 2:
        # Credit to Lasse: https://stackoverflow.com/questions/63118710/
        # This made sure that the colorbar height always matches those of the contour plots.
        axins = inset_axes(ax, width = "5%", height = "100%", loc = 'lower left',
                bbox_to_anchor = (1.02, 0., 1, 1), bbox_transform = ax.transAxes,
                borderpad = 0)
        
        cbar = plt.colorbar(ax.collections[1],cax=axins)

        cbar.set_ticks(ticks)

        cbar.ax.set_yticklabels([f"{tick: .2f}" for tick in ticks])

        if not label is None:
            cbar.set_label(label,fontweight='bold')
        else:
            cbar.set_label("Predicted Difference",fontweight='bold')
        
        ax.set_xlabel(tvars[0],fontweight='bold')
        ax.set_ylabel(tvars[1],fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    else:
        if not label is None:
            ax.set_ylabel(label,fontweight='bold')
        else:
            ax.set_ylabel("Predicted Difference",fontweight='bold')
        ax.set_xlabel(tvars[0],fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{tick: .2f}" for tick in ticks])

        if plot_exist:
            ax.set_xlim(min(x1),max(x1))
    
    if not title is None:
        ax.set_title(title,fontweight='bold')
    
    if isinstance(model,GAMMLSS):
        # Clean up
        model.formula = None

    if fig is not None:
        plt.show()


def plot_val(model:GAMM or GAMMLSS,pred_viz:[str] or None = None,resid_type="deviance",
             ar_lag=100,response_scale=False,qq=True,axs=None,fig_size=(6/2.54,6/2.54)):
    """Plots residual plots useful for validating whether the `model` meets the regression assumptions.

    At least four plots will be generated:

    - A scatter-plot: Model predictions (always on response/mean scale) vs. Observations
    - A scatter-plot: Model predictions (optionally on response/mean scale) vs. Residuals
    - A Histogram/QQ-plot: Residuals (with density overlay of expected distribution)/Quantile-quantile plot for residuals against theoretical quantiles.
    - An ACF plot: Showing the auto-correlation in the residuals at each of `ar_lag` lags
    
    For each additional predictor included in `pred_viz`, an additional scatter-plot will be generated plotting the predictor values against the residuals.

    Which residuals will be visualized depends on the choice of `model` and `resid_type`. If `model` is a `GAMM` model, `resid_type` will determine whether
    "Pearson" or "Deviance" (default) residuals are to be plotted (Wood, 2017). Except, for a Gaussian `GAMM`, in which case the function will always plot
    the classical residuals, i.e., the difference between model predictions (\mu_i) and observed values (y_i). This ensures that by default and for any `GAMM`
    we can expect the residuals to look like N independent samples from N(0,sqrt(phi)) - where \phi is the scale parameter of the `GAMM` (\sigma^2 for Gaussian).
    Hence, we can interpret all the plots in the same way. Note, that residuals for Binomial models will generally not look pretty or like N(0,sqrt(phi)) - but they
    should still be reasonably independent.

    If `model` is a `GAMMLSS` model, `resid_type` will be ignored. Instead, the function will always plot standardized residuals that behave a lot like deviance residuals, except for
    the fact that we also cancel for \phi, so that we can expect the residuals to look like N independent samples from N(0,1). In many cases, the computation of these residuals will
    thus follow the computation for GAMMs (for a Gaussian GAMMLSS model we can for example simply scale the residual vector by the sigma parameter estimated for each observation to achieve
    the desired distribution result) while for others more complicated standardization might be necessary (see Rigby & Stasinopoulos, 2005) - this will be noted in the docstring of
    the :func:`resid()` method implemented by each `GAMLSSFamily` family. Again, we have to make an exeception for the Multinomial model (`MULNOMLSS`), which is currently not supported by this function.

    References:

    - Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for Location, Scale and Shape.
    - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

    :param model: Estimated GAMM or GAMMLSS model, for which the reisdual plots should be generated.
    :type model: GAMM or GAMMLSS
    :param pred_viz: A list of additional predictor variables included in the model. For each one provided an additional plot will be created with the predictor on the x-axis and the residuals on the y-axis, defaults to None
    :type pred_viz: [str] or None, optional
    :param resid_type: Type of residual to visualize. For a `model` that is a GAMM this can be "Pearson" or "Deviance" - for a Gaussian GAMM the function will alwasy plot default residuals (y_i - \mu_i) independent of what is provided. For a `model` that is a GAMMLSS, the function will always plot standardized residuals that should approximately behave like deviance ones - except that they can be expected to look like N(0,1) if the model is specified correctly, defaults to "deviance"
    :type resid_type: str, optional
    :param ar_lag: Up to which lag the auto-correlation function in the residuals should be computed and visualized, defaults to 100
    :type ar_lag: int, optional
    :param response_scale: Whether or not predictions should be visualized on the scale of the mean or not, defaults to False - i.e., predictions are visualized on the scale of the model predictions/linear scale
    :type response_scale: bool, optional
    :param qq: Whether or not a qq-plot should be drawn instead of a Histogram, defaults to True
    :type qq: bool, optional
    :param axs: A list of matplotlib.axis on which Figures should be drawn, defaults to None in which case axis will be created by the function and plot.show() will be called at the end
    :type axs: [matplotlib.axis], optional
    :param fig_size: Tuple holding figure size, which will be used to determine the size of the figures created if `axs=None`, defaults to (6/2.54,6/2.54)
    :type fig_size: tuple, optional
    :raises ValueError: If fewer matplotlib axis are provided than the number of figures that would be created
    :raises TypeError: If the function is called with a `model` of the `MULNOMLSS` family, which is currently not supported
    """

    if isinstance(model.family,MULNOMLSS):
        raise TypeError("Function does not currently support `Multinomial` models.")

    if isinstance(model,GAMMLSS):
        # Set up everything for that we can plot all smooth terms for distribution parameter `dist_par`.
        model.formula = model.formulas[0]

    varmap = model.formula.get_var_map()
    n_figures = 4

    if pred_viz is not None:
        for pr in pred_viz:
            n_figures +=1
    
    if not axs is None and len(axs) != n_figures:
        raise ValueError(f"{n_figures} plots would be created, but only {len(axs)} axes were provided!")

    figs = None
    if axs is None:
        figs = [plt.figure(figsize=fig_size,layout='constrained') for _ in range(n_figures)]
        axs = [fig.add_subplot(1,1,1) for fig in figs]
    
    if isinstance(model,GAMMLSS) == False:
        _, sigma = model.get_pars() # sigma = **variance** of residuals!
        pred = model.pred # The model prediction for the entire data
    else:
        sigma = 1 # Standardized residuals should look like N(0,1)
        pred = model.overall_preds[0]

    if response_scale:
        if isinstance(model,GAMMLSS):
            pred = model.family.links[0].fi(pred)
        else:
            pred = model.family.link.fi(pred)

    if isinstance(model,GAMMLSS) == False:
        res = model.get_resid(type=resid_type)
    else:
        res = model.get_resid() # resid are alwasy standardized for GAMMLSS models

    y = model.formula.y_flat[model.formula.NOT_NA_flat] # The dependent variable after NAs were removed

    # obs vs. pred plot should always be on response scale
    if isinstance(model,GAMMLSS) == False and (response_scale == False):
        axs[0].scatter(model.family.link.fi(pred),y,color="black",facecolor='none')
    elif isinstance(model,GAMMLSS) and (response_scale == False):
        axs[0].scatter(model.family.links[0].fi(pred),y,color="black",facecolor='none')
    else:
        axs[0].scatter(pred,y,color="black",facecolor='none')

    axs[0].set_xlabel("Predicted (Mean scale)",fontweight='bold')
    axs[0].set_ylabel("Observed",fontweight='bold')
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)

    axs[1].scatter(pred,res,color="black",facecolor='none')
    if response_scale == False:
        axs[1].set_xlabel("Predicted",fontweight='bold')
    else:
        axs[1].set_xlabel("Predicted (Mean scale)",fontweight='bold')
    axs[1].set_ylabel("Residuals",fontweight='bold')
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)

    axi = 2

    if pred_viz is not None:
        for pr in pred_viz:
            pr_val =  model.formula.cov_flat[model.formula.NOT_NA_flat,varmap[pr]]
            axs[axi].scatter(pr_val,res,color="black",facecolor='none')
            axs[axi].set_xlabel(pr,fontweight='bold')
            axs[axi].set_ylabel("Residuals",fontweight='bold')
            axs[axi].spines['top'].set_visible(False)
            axs[axi].spines['right'].set_visible(False)
            axi += 1

    # Histogram for normality
    if qq==False:
        axs[axi].hist(res,bins=100,density=True,color="black")
        x = np.linspace(scp.stats.norm.ppf(0.0001,scale=math.sqrt(sigma)),
                        scp.stats.norm.ppf(0.9999,scale=math.sqrt(sigma)), 100)

        axs[axi].plot(x, scp.stats.norm.pdf(x,scale=math.sqrt(sigma)),
                'r-', lw=3, alpha=0.6)

        axs[axi].set_xlabel("Residuals",fontweight='bold')
        axs[axi].set_ylabel("Density",fontweight='bold')
        axs[axi].spines['top'].set_visible(False)
        axs[axi].spines['right'].set_visible(False)
    else:
        # Get theoretical quantiles
        tq = scp.stats.norm.ppf(np.linspace(0,1,len(res)),scale=math.sqrt(sigma))
        # Get empirical quantiles
        eq = np.quantile(res,np.linspace(0,1,len(res)))
        axs[axi].scatter(tq,eq,color="black",facecolor='none')
        # Add theoretical truth line
        axs[axi].plot(tq,tq,color="black")

        axs[axi].set_xlabel("Theoretical Quantiles",fontweight='bold')
        axs[axi].set_ylabel("Empirical Residual Quantiles",fontweight='bold')
        axs[axi].spines['top'].set_visible(False)
        axs[axi].spines['right'].set_visible(False)

    axi += 1

    # Auto-correlation check
    cc = np.vstack([res[:-ar_lag,0],*[res[l:-(ar_lag-l),0] for l in range(1,ar_lag)]]).T
    acf = [np.corrcoef(cc[:,0],cc[:,l])[0,1] for l in range(ar_lag)]

    for lg in range(ar_lag):
        axs[axi].plot([lg,lg],[0,acf[lg]],color="black",linewidth=0.5)

    axs[axi].axhline(0,color="red")
    axs[axi].set_xlabel("Lag",fontweight='bold')
    axs[axi].set_ylabel("ACF",fontweight='bold')
    axs[axi].spines['top'].set_visible(False)
    axs[axi].spines['right'].set_visible(False)

    if isinstance(model,GAMMLSS):
        # Clean up
        model.formula = None

    if figs is not None:
        plt.show()