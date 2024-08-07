
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import copy
from .sim import np,pd,scp
from mssm.models import *


def __get_data_limit_counts(formula,pred_dat,cvars,by):
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
        sort_train = sort_train[train_cond_unq_exists[train_cond_inv],:]

    # Check for each combination in continuous prediction columns whether the values are within
    # min and max of the respective trainings columns
    pred_in_limits = np.ones(len(sort_pred),dtype=bool)

    for cont_i in range(sort_pred.shape[1]):
        pred_in_limits = pred_in_limits & ((sort_pred[:,cont_i] <= np.max(sort_train[:,cont_i])) & (sort_pred[:,cont_i] >= np.min(sort_train[:,cont_i])))

    # Now find the counts in the training data for each combination of continuous variables
    train_unq,train_unq_counts = np.unique(sort_train,axis=0,return_counts=True)
    
    return pred_in_limits,train_unq,train_unq_counts.astype(float),cont_vars


def __pred_plot(pred,b,tvars,pred_in_limits,x1,x2,x1_exp,ci,n_vals,ax,_cmp,col,ylim,link):
    
    if len(tvars) == 2:
        T_pred = pred.reshape(n_vals,n_vals)

        if not link is None:
            T_pred = link.fi(T_pred)

        if not pred_in_limits is None:
            T_pred = np.ma.array(T_pred, mask = (pred_in_limits == False))

        T_pred = T_pred.T

        halfrange = None
        if not ylim is None:
            halfrange = np.max(np.abs(ylim))
        if ci:
            f1 = ax.contourf(x1,x2,T_pred,levels=n_vals,cmap=_cmp,norm=colors.CenteredNorm(halfrange=halfrange),alpha=0.4)
            T_pred = np.ma.array(T_pred, mask= ((pred + b) > 0) & ((pred - b) < 0))
            
        ff = ax.contourf(x1,x2,T_pred,levels=n_vals,cmap=_cmp,norm=colors.CenteredNorm(halfrange=halfrange))
        ll = ax.contour(x1,x2,T_pred,colors="grey")

    elif len(tvars) == 1:

        x = x1_exp
        y = pred
        if ci:
            cu = pred + b
            cl = pred - b

        if not link is None:
            y = link.fi(y)
            if ci:
                cu = link.fi(cu)
                cl = link.fi(cl)

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
            
        ax.plot(x,y,color=_cmp(col))

def plot(model:GAMM,which:[int] or None = None,cmp:str or None = None, n_vals:int = 30,ci=None,
         ci_alpha=0.05,whole_interval=False,n_ps=10000,seed=None,plot_exist=False,te_exist_style='both',response_scale=False,axs=None,fig_size=(6/2.54,6/2.54),
         math_font_size = 9,math_font = 'cm',tp_use_inter=False,ylim=None,prov_level_cols=None):

    
    terms = model.formula.get_terms()
    stidx = model.formula.get_smooth_term_idx()

    varmap = model.formula.get_var_map()
    vartypes = model.formula.get_var_types()
    varmins = model.formula.get_var_mins()
    varmaxs = model.formula.get_var_maxs()
    code_factors = model.formula.get_coding_factors()
    factor_codes = model.formula.get_factor_codings()

    if cmp is None:
        cmp = 'RdYlBu_r'
        
    _cmp = matplotlib.colormaps[cmp]

    if not which is None:
        stidx = which

    n_figures = 0
    for sti in stidx:
        if isinstance(terms[sti],fs):
            n_figures +=1
        else:
            if not terms[sti].by is None:
                n_figures += len(code_factors[terms[sti].by])

            else:
                n_figures += 1
    
    if not axs is None and sum(axs.shape) != n_figures:
        raise ValueError(f"{n_figures} plots would be created, but only {sum(axs.shape)} axes were provided!")

    figs = None
    if axs is None:
        figs = [plt.figure(figsize=fig_size,layout='constrained') for _ in range(n_figures)]
        axs = [fig.add_subplot(1,1,1) for fig in figs]
    
    axi = 0

    for sti in stidx:

        tvars = terms[sti].variables
        #print(tvars)
        pred_dat = {}
        x1_exp = []
        if len(tvars) == 2:


            x1 = np.linspace(varmins[tvars[0]],varmaxs[tvars[0]],n_vals)
            x2 = np.linspace(varmins[tvars[1]],varmaxs[tvars[1]],n_vals)

            
            x2_exp = []

            for x1v in x1:
                for x2v in x2:
                    x1_exp.append(x1v)
                    x2_exp.append(x2v)
            
            pred_dat[tvars[0]] = x1_exp
            pred_dat[tvars[1]] = x2_exp
        
        elif len(tvars) == 1:
            x1 = None
            x2 = None
            x1_exp = np.linspace(varmins[tvars[0]],varmaxs[tvars[0]],n_vals)
            pred_dat[tvars[0]] = x1_exp
        else:
            continue
        
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
                        pred_dat[vari] = [0 for _ in range(len(x1_exp))]
            
            pred_dat_pd = pd.DataFrame(pred_dat)
            
            use_ci = ci
            if use_ci is None:
                use_ci = True

            use = [sti]
            if tp_use_inter and len(tvars) == 2:
                use = [0,sti]

            pred,_,b= model.predict(use,pred_dat_pd,ci=use_ci,alpha=ci_alpha,whole_interval=whole_interval,n_ps=n_ps,seed=seed)

            if tp_use_inter and len(tvars) == 2:
                _cf,_ = model.get_pars()
                pred -= _cf[0]

            te_in_limits = None
            if plot_exist:
                pred_in_limits,train_unq,train_unq_counts,cont_vars = __get_data_limit_counts(model.formula,pred_dat_pd,tvars,None)

            if len(tvars) == 2 and (te_exist_style == "both" or te_exist_style == "hide"):
                te_in_limits = pred_in_limits
            
            link = None
            if response_scale:
                link = model.family.link

            __pred_plot(pred,b,tvars,te_in_limits,x1,x2,x1_exp,use_ci,n_vals,axs[axi],_cmp,0.7 if prov_level_cols is None else prov_level_cols,ylim,link)

            if len(tvars) == 1:
                axs[axi].set_ylabel('$f(' + tvars[0] + ')$',math_fontfamily=math_font,size=math_font_size,fontweight='bold')
                axs[axi].set_xlabel(tvars[0],fontweight='bold')
                axs[axi].spines['top'].set_visible(False)
                axs[axi].spines['right'].set_visible(False)

                if plot_exist:
                    
                    train_unq_counts[train_unq_counts > 0] = 1 
                    pred_range = np.abs(np.max(pred) - np.min(pred))*0.025
                    x_counts = np.ndarray.flatten(train_unq[:,[cvar ==tvars[0] for cvar in cont_vars]])
                    x_range = np.abs(np.max(x_counts) - np.min(x_counts))
                    
                    axs[axi].bar(x=x_counts,bottom=axs[axi].get_ylim()[0],height=pred_range*train_unq_counts,color='black',width=max(0.05,x_range/(2*len(x_counts))))
            
            elif len(tvars) == 2:
                axs[axi].set_ylabel(tvars[1],fontweight='bold')
                axs[axi].set_xlabel(tvars[0],fontweight='bold')
                axs[axi].set_box_aspect(1)
                
                if plot_exist and (te_exist_style == "both" or te_exist_style == 'rug'):
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

                cbar_label = '(' + tvars[0] + ',' + tvars[1] + ')$'

                cbar_label = '$f' + cbar_label

                cbar.ax.set_ylabel(cbar_label,math_fontfamily=math_font,size=math_font_size)

            axi += 1
            #plt.tight_layout()
            
        
        elif not terms[sti].by is None or not terms[sti].binary is None:
            
            if not terms[sti].by is None:
                sti_by = terms[sti].by
            else:
                sti_by = terms[sti].binary[0]

            levels = list(code_factors[sti_by].keys())

            if not terms[sti].binary is None:
                levels = [factor_codes[sti_by][terms[sti].binary[1]]]
        
            if isinstance(terms[sti],fs) and len(levels) > 25:
                levels = np.random.choice(levels,replace=False,size=25)

            if prov_level_cols is None:
                level_cols = np.linspace(0.1,0.9,len(levels))
            else:
                level_cols = prov_level_cols

            for level_col,leveli in zip(level_cols,levels):
                pred_level_dat = copy.deepcopy(pred_dat)

                for vari in varmap.keys():
                    #print(vari)
                    if vari in terms[sti].variables:
                        continue
                    else:
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
                            pred_level_dat[vari] = [0 for _ in range(len(x1_exp))]
                
                pred_dat_pd = pd.DataFrame(pred_level_dat)

                use_ci = ci
                if use_ci is None:
                    if not isinstance(terms[sti],fs):
                        use_ci = True
                    else:
                        use_ci = False
                
                use = [sti]
                if tp_use_inter and len(tvars) == 2:
                    use = [0,sti]
                
                pred,_,b= model.predict(use,pred_dat_pd,ci=use_ci,alpha=ci_alpha,whole_interval=whole_interval,n_ps=n_ps,seed=seed)

                if tp_use_inter and len(tvars) == 2:
                    _cf,_ = model.get_pars()
                    pred -= _cf[0]

                te_in_limits = None
                if plot_exist:
                    pred_in_limits,train_unq,train_unq_counts,cont_vars = __get_data_limit_counts(model.formula,pred_dat_pd,tvars,[sti_by])
                
                if len(tvars) == 2 and (te_exist_style == "both" or te_exist_style == "hide"):
                    te_in_limits = pred_in_limits

                link = None
                if response_scale:
                    link = model.family.link

                __pred_plot(pred,b,tvars,te_in_limits,x1,x2,x1_exp,use_ci,n_vals,axs[axi],_cmp,level_col,ylim,link)

                if not isinstance(terms[sti],fs):

                    if len(tvars) == 1:
                        ax_label = '$f_{' + str(code_factors[sti_by][leveli]) + '}' + '(' + tvars[0] + ')$'
                        axs[axi].set_ylabel(ax_label,math_fontfamily=math_font,size=math_font_size,fontweight='bold')
                        axs[axi].set_xlabel(tvars[0],fontweight='bold')
                        axs[axi].spines['top'].set_visible(False)
                        axs[axi].spines['right'].set_visible(False)
                        
                        if plot_exist:
                    
                            #train_unq_counts /= np.max(train_unq_counts)
                            train_unq_counts[train_unq_counts > 0] = 1 
                            pred_range = np.abs(np.max(pred) - np.min(pred))*0.025
                            x_counts = np.ndarray.flatten(train_unq[:,[cvar ==tvars[0] for cvar in cont_vars]])
                            x_range = np.abs(np.max(x_counts) - np.min(x_counts))

                            axs[axi].bar(x=x_counts,bottom=axs[axi].get_ylim()[0],height=pred_range*train_unq_counts,color='black',width=max(0.05,x_range/(2*len(x_counts))))

                    elif len(tvars) == 2:
                        axs[axi].set_ylabel(tvars[1],fontweight='bold')
                        axs[axi].set_xlabel(tvars[0],fontweight='bold')
                        axs[axi].set_box_aspect(1)

                        if plot_exist and (te_exist_style == "both" or te_exist_style == 'rug'):

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

                        cbar_label = '(' + tvars[0] + ',' + tvars[1] + ')$'

                        cbar_label = '$f_{' + str(code_factors[sti_by][leveli]) + '}' + cbar_label

                        cbar.ax.set_ylabel(cbar_label,math_fontfamily=math_font,size=math_font_size)
                    axi += 1

            if isinstance(terms[sti],fs):
                axs[axi].set_ylabel('$f_{' + str(sti_by) + '}(' + tvars[0] + ')$',math_fontfamily=math_font,size=math_font_size,fontweight='bold')
                axs[axi].set_xlabel(tvars[0],fontweight='bold')
                axs[axi].spines['top'].set_visible(False)
                axs[axi].spines['right'].set_visible(False)
                axi += 1
    
    if figs is not None:
        plt.show()


def plot_fitted(pred_dat,tvars,model,use:[int] or None = None,ci=True,ci_alpha=0.05,whole_interval=False,n_ps=10000,seed=None,plot_exist=False,te_exist_style='both',response_scale=True,ax=None,cmp:str or None = None,
                fig_size=(6/2.54,6/2.54),math_font_size = 9,math_font = 'cm',ylim=None,col=0.7,label=None,title=None):
    
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

    pred,_,b= model.predict(use,pred_dat,ci=ci,alpha=ci_alpha,whole_interval=whole_interval,n_ps=n_ps,seed=seed)

    te_in_limits = None
    if plot_exist:
        pred_factors = [var for var in pred_dat.columns if model.formula.get_var_types()[var] == VarType.FACTOR]
        if len(pred_factors) == 0:
            pred_factors = None

        pred_in_limits,train_unq,train_unq_counts,cont_vars = __get_data_limit_counts(model.formula,pred_dat,tvars,pred_factors)

    if len(tvars) == 2 and (te_exist_style == "both" or te_exist_style == "hide"):
        te_in_limits = pred_in_limits

    link = None
    if response_scale:
        link = model.family.link
    
    __pred_plot(pred,b,tvars,te_in_limits,x1,x2,x1_exp,ci,len(x1),ax,_cmp,col,ylim,link)

    if len(tvars) == 2:
        # Credit to Lasse: https://stackoverflow.com/questions/63118710/
        # This made sure that the colorbar height always matches those of the contour plots.

        if plot_exist and (te_exist_style == "both" or te_exist_style == 'rug'):

            train_unq_counts[train_unq_counts > 0] = 0.1
            x_counts = np.ndarray.flatten(train_unq[:,[cvar ==tvars[0] for cvar in cont_vars]])
            y_counts = np.ndarray.flatten(train_unq[:,[cvar ==tvars[1] for cvar in cont_vars]])
            tot_range = np.abs(max(np.max(x_counts),np.max(y_counts)) - min(np.min(x_counts),np.min(y_counts)))
            ax.scatter(x_counts,y_counts,alpha=train_unq_counts,color='black',s=tot_range/(len(x_counts)))

        axins = inset_axes(ax, width = "5%", height = "100%", loc = 'lower left',
                bbox_to_anchor = (1.02, 0., 1, 1), bbox_transform = ax.transAxes,
                borderpad = 0)
        
        if ci:
            cbar = plt.colorbar(ax.collections[1],cax=axins)
        else:
            cbar = plt.colorbar(ax.collections[0],cax=axins)

        if not label is None:
            cbar.set_label(label,fontweight='bold')
        else:
            cbar.set_label("Predicted",fontweight='bold')
    else:
        if not label is None:
            ax.set_ylabel(label,fontweight='bold')
        else:
            ax.set_ylabel("Predicted",fontweight='bold')
        ax.set_xlabel(tvars[0],fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if plot_exist:
                    
            train_unq_counts[train_unq_counts > 0] = 1 
            pred_range = np.abs(np.max(pred) - np.min(pred))*0.025
            x_counts = np.ndarray.flatten(train_unq[:,[cvar ==tvars[0] for cvar in cont_vars]])
            x_range = np.abs(np.max(x_counts) - np.min(x_counts))
            
            ax.bar(x=x_counts,bottom=ax.get_ylim()[0],height=pred_range*train_unq_counts,color='black',width=max(0.05,x_range/(2*len(x_counts))))
    
    if not title is None:
        ax.set_title(title,fontweight='bold')

    if fig is not None:
        plt.show()

def plot_diff(pred_dat1,pred_dat2,tvars,model,use:[int] or None = None,ci_alpha=0.05,whole_interval=False,n_ps=10000,seed=None,plot_exist=True,response_scale=True,ax=None,n_vals=30,cmp:str or None = None,
              fig_size=(6/2.54,6/2.54),math_font_size = 9,math_font = 'cm',ylim=None,col=0.7,label=None,title=None):

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

    pred,b= model.predict_diff(pred_dat1,pred_dat2,use,alpha=ci_alpha,whole_interval=whole_interval,n_ps=n_ps,seed=seed)

    in_limits = None
    if plot_exist:
        pred_factors1 = [var for var in pred_dat1.columns if model.formula.get_var_types()[var] == VarType.FACTOR]
        if len(pred_factors1) == 0:
            pred_factors1 = None
        pred_in_limits1,train_unq1,train_unq_counts1,cont_vars1 = __get_data_limit_counts(model.formula,pred_dat1,tvars,pred_factors1)

        pred_factors2 = [var for var in pred_dat2.columns if model.formula.get_var_types()[var] == VarType.FACTOR]
        if len(pred_factors2) == 0:
            pred_factors2 = None
        pred_in_limits2,train_unq2,train_unq_counts2,cont_vars2 = __get_data_limit_counts(model.formula,pred_dat2,tvars,pred_factors2)

        in_limits = pred_in_limits1 & pred_in_limits2
    
    link = None
    if response_scale:
        link = model.family.link

    __pred_plot(pred,b,tvars,in_limits,x1,x2,x1_exp,True,n_vals,ax,_cmp,col,ylim,link)

    if len(tvars) == 2:
        # Credit to Lasse: https://stackoverflow.com/questions/63118710/
        # This made sure that the colorbar height always matches those of the contour plots.
        axins = inset_axes(ax, width = "5%", height = "100%", loc = 'lower left',
                bbox_to_anchor = (1.02, 0., 1, 1), bbox_transform = ax.transAxes,
                borderpad = 0)
        
        cbar = plt.colorbar(ax.collections[1],cax=axins)

        if not label is None:
            cbar.set_label(label,fontweight='bold')
        else:
            cbar.set_label("Predicted Difference",fontweight='bold')
    else:
        if not label is None:
            ax.set_ylabel(label,fontweight='bold')
        else:
            ax.set_ylabel("Predicted Difference",fontweight='bold')
        ax.set_xlabel(tvars[0],fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if plot_exist:
            ax.set_xlim(min(x1),max(x1))
    
    if not title is None:
        ax.set_title(title,fontweight='bold')

    if fig is not None:
        plt.show()


def plot_val(model:GAMM or GAMMLSS,pred_viz:[str] or None = None,resid_type="deviance",ar_lag=100,response_scale=False,axs=None,fig_size=(6/2.54,6/2.54),cmp:str or None = None):

    varmap = model.formula.get_var_map()
    n_figures = 4

    if pred_viz is not None:
        for pr in pred_viz:
            n_figures +=1
    
    if not axs is None and sum(axs.shape) != n_figures:
        raise ValueError(f"{n_figures} plots would be created, but only {sum(axs.shape)} axes were provided!")

    figs = None
    if axs is None:
        figs = [plt.figure(figsize=fig_size,layout='constrained') for _ in range(n_figures)]
        axs = [fig.add_subplot(1,1,1) for fig in figs]
    
    _, sigma = model.get_pars() # sigma = **variance** of residuals!
    pred = model.pred # The model prediction for the entire data

    if response_scale:
        pred = self.family.link.fi(pred)

    res = model.get_resid(type=resid_type)
    y = model.formula.y_flat[model.formula.NOT_NA_flat] # The dependent variable after NAs were removed

    axs[0].scatter(pred,y,color="black",facecolor='none')
    axs[0].set_xlabel("Predicted",fontweight='bold')
    axs[0].set_ylabel("Observed",fontweight='bold')
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)

    axs[1].scatter(pred,res,color="black",facecolor='none')
    axs[1].set_xlabel("Predicted",fontweight='bold')
    axs[1].set_ylabel("Residuals",fontweight='bold')
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)

    axi = 2

    if pred_viz is not None:
        for pr in pred_viz:
            pr_val =  model.formula.cov_flat[model.formula.NOT_NA_flat,varmap[pr]]
            axs[axi].scatter(pred,res,color="black",facecolor='none')
            axs[axi].set_xlabel(pr,fontweight='bold')
            axs[axi].set_ylabel("Residuals",fontweight='bold')
            axs[axi].spines['top'].set_visible(False)
            axs[axi].spines['right'].set_visible(False)
            axi += 1

    # Histogram for normality
    axs[axi].hist(res,bins=100,density=True,color="black")
    x = np.linspace(scp.stats.norm.ppf(0.0001,scale=math.sqrt(sigma)),
                    scp.stats.norm.ppf(0.9999,scale=math.sqrt(sigma)), 100)

    axs[axi].plot(x, scp.stats.norm.pdf(x,scale=math.sqrt(sigma)),
            'r-', lw=3, alpha=0.6)

    axs[axi].set_xlabel("Residuals",fontweight='bold')
    axs[axi].set_ylabel("Density",fontweight='bold')
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

    if figs is not None:
        plt.show()