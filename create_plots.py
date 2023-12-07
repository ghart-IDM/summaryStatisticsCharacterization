'''
Generate figures based on the results rendered by the run_experiments.py script.
'''
import os
import re
import numpy 
import pandas

import statsmodels.api as sm
from statsmodels.formula.api import ols

import optuna

import seaborn
import matplotlib
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

#-------------------------------------------------------------------------------
# Parameters
#-------------------------------------------------------------------------------
# Model
model = 'SIR'

# Optimization
name     = model + "-tree-summary-stats-rev0"
storage  = f'sqlite:///{name}.db'
dir_figure = os.path.join('figures', model)
csv_fileName = model + 'TrialData.csv'

seaborn.set(font_scale=1.8)
#-------------------------------------------------------------------------------



def main():

    if os.path.isfile(csv_fileName):
        data = pandas.read_csv(csv_fileName)
        if model == 'SIR':
            data['params_sample_fraction'] = data['params_sample_fraction'].round(2)
    else:
        # Load study and get dataframe
        study = optuna.load_study(storage=storage, study_name=name)
        data = study.trials_dataframe()
        
        # # Remove failed sims
        # data = data.loc[data.state=='COMPLETE', :]
        # data = data.loc[~numpy.isinf(data.value), :]
        
        columns = data.columns.str.startswith("params_") | data.columns.str.startswith("user_attrs_")
        data = data.loc[:, columns]
        data = data.drop( columns = ["user_attrs_time_tree_generation"] )
        data.columns = data.columns.str.replace('_topology', '_unweighted')
        data.to_csv(csv_fileName, index=False)
    
    time_dependent_params = data.iloc[0:2,:].filter(regex='^user_attrs_params_').columns.str.replace("user_attrs_params_","").to_list()
    time_dependent_params = [re.sub('R_eff_1([0-9]+)', r'R_eff_\1', x) for x in time_dependent_params]
    data.rename(columns=lambda x: re.sub('R_eff_1([0-9]+)', r'R_eff_\1', x), inplace=True)
    params = data.iloc[0:2,:].filter(regex='^params_').columns.str.replace("params_","").to_list()
    data.columns = data.columns.str.replace("user_attrs_params_", "")
    features = data.iloc[0:2,:].filter(regex='^user_attrs_').columns.str.replace("user_attrs_", "").to_list()
    time_dependent_features = [x for x in features if re.search('_[0-9]+$', x)]
    features = [x for x in features if not re.search('_[0-9]+$', x)]
    data.columns = data.columns.str.replace("params_", "")
    data.columns = data.columns.str.replace("user_attrs_", "")

    build_report( data, params, features )

    if model == 'SIR':
        data = data.loc[:, params + time_dependent_params + time_dependent_features]
        data.set_index(params, append=True, inplace=True)
        data.columns = data.columns.str.rsplit("_", n=1, expand=True)
        data = data.stack(level=1).rename_axis(['Index_0'] + params + ['Week']).reset_index()
        data['Week'] = data['Week'].astype('int64')
        time_dependent_features = numpy.unique([x.rpartition('_')[0] for x in time_dependent_features]).tolist()
        time_dependent_params = numpy.unique([x.rpartition('_')[0] for x in time_dependent_params]).tolist()
    
        # Individual Statistics
        n_params = len(time_dependent_params)
        for stat_name in time_dependent_features:
            print( '... processing (time dependent): ', stat_name )
            stat_df = data.loc[ :, ['Week'] + time_dependent_params + [stat_name ]]
            if stat_df is not None:
                single_stat_analysis_time_dependent( stat_name, stat_df.fillna(0).clip(-1e6, 1e6),
                                     time_dependent_params)

    
    return
    
    

def build_report( data, params, features ):

    scores = multi_stat_analysis( data, params, features )
    

    # Individual Statistics
    print('scores.index = ', scores.index)
    n_params = len(params)
    for stat_name in features:
        print( '... processing: ', stat_name )
        stat_df = data.loc[ :, params + [stat_name ]]
        if stat_df is not None:
            single_stat_analysis( stat_name, stat_df.fillna(0).clip(-1e6, 1e6),
                                 params)

    return


def multi_stat_analysis( data, params, features ):

    # Initialize
    n_stats = len(features) 
    n_params = len(params)
    scores = pandas.DataFrame( columns=["corr_" + x for x in params] + ["R"],
                              index=features, dtype=numpy.float64)
    scores['name'] = features
    scores.set_index("name", inplace=True)
    for param in params:
        scores['corr_' + param] = numpy.empty(n_stats)


    # Compute scores
    for stat_name in features:
        stat_df = data.loc[ :, params + [stat_name ]]
        if stat_df is not None:

            for param in params:
                scores.loc[stat_name, "corr_" + param] = stat_df[stat_name].corr( stat_df[param] )
                

            model = ols(stat_name + " ~ " + " + ".join(params), data=stat_df).fit()
            scores.loc[stat_name, "R"] = model.rsquared**.5

    scores.dropna(inplace=True)
    # scores.index = scores.index.str.replace('_topology', '_unweighted')
    scores["R_abs"] = scores["R"].abs()
    scores = scores.sort_values( "R_abs", ascending=False )    #.replace([numpy.inf, -numpy.inf], numpy.nan, inplace=True) \
                   
    print('scores = \n', scores)


    # Plot correlations
    fig_corr, ax_corr = plt.subplots( n_params+1, 1, figsize=(18,20), sharex=True )
    scores.plot.bar( y="R", ax=ax_corr[0] )
    for i, param in enumerate(params):
        scores.plot.bar( y="corr_" + param, ax=ax_corr[i + 1] )
    
    fig_corr.tight_layout()
    plt.savefig(os.path.join(dir_figure, 'correlations.png'))
    plt.close(fig_corr)

    # Fixing the annot to have all values may require downgrading MATPLOTLIB to 3.7.3
    fig_table, ax_table = plt.subplots( 1, 1, figsize=(17,20) )
    seaborn.heatmap( scores.drop(columns="R_abs"),
                     vmin       = -1,
                     vmax       = 1,
                     cmap       = "BrBG",
                     annot      = True, 
                     linewidths = .5, 
                     ax         = ax_table
                    )

    fig_table.tight_layout()
    plt.savefig(os.path.join(dir_figure, 'scoreRanking.png'))
    plt.close(fig_table)


    fig_pc, ax_pc = plt.subplots( 1, 1, figsize=(24,24) )
    corr = data[features].corr()
    corr.dropna(inplace=True, how='all')
    corr.dropna(inplace=True, how='all', axis=1)
    cax = ax_pc.matshow( corr, cmap="BrBG", vmin=-1, vmax=1 )
    fig_pc.colorbar(cax)
    ticks = numpy.arange( 0, len(corr.columns), 1 )
    ax_pc.set_xticks( ticks )
    plt.xticks( rotation=90 )
    ax_pc.set_yticks( ticks )
    ax_pc.set_xticklabels( corr.columns )
    ax_pc.set_yticklabels( corr.columns )
    
    fig_pc.tight_layout()
    plt.savefig(os.path.join(dir_figure, 'correlations_features.png'))
    plt.close(fig_pc)


    return scores
    
def single_stat_analysis( stat_name, data, params):

    n_params = len(params)
    # Draw scatter plot
    fig_sp, ax_sp = plt.subplots( n_params, 1, figsize=(16,16) )
    for i, param in enumerate(params):
        data.plot.scatter( param, stat_name, ax=ax_sp[i] )
        # seaborn.swarmplot( data, x=param, y=stat_name, ax=ax_sp[i] )
    ax_sp[0].set_title(stat_name)
    fig_sp.tight_layout()
    fig_sp.savefig( os.path.join(dir_figure, 'scatter-plot--' + stat_name + '.png' ))
    plt.close(fig_sp)
    
    # Draw box plot
    fig_sp, ax_sp = plt.subplots( n_params, 1, figsize=(16,16) )
    for i, param in enumerate(params):
        my_plot = seaborn.boxplot( data, x=param, y=stat_name, ax=ax_sp[i] )
        if model == 'birthDeath':
            my_plot.set_xticklabels(my_plot.get_xticklabels(), rotation=90)
        else:
            my_plot.set_ylabel(None)
    ax_sp[0].set_title(stat_name)
    fig_sp.supylabel(stat_name)
    fig_sp.tight_layout()
    fig_sp.savefig( os.path.join(dir_figure, 'box-plot--' + stat_name + '.png' ))
    plt.close(fig_sp)

    # Draw contour plot
    fig_cp, ax_cp = plt.subplots( n_params, n_params, figsize=(16,16) )
    fig_cp.set_figheight(15)
    fig_cp.set_figwidth(18)
    for i in range(n_params):
        # ax_cp[0,i].set_title(stat_name)
        for j in range(n_params): 
            if i != j:
                cntr = ax_cp[i,j].tricontourf( data[params[i]].values,
                                          data[params[j]].values,
                                          data[stat_name].values,
                                          levels = 10
                                         )
                ax_cp[i,j].set_xlabel(params[i])
                ax_cp[i,j].set_ylabel(params[j])
    fig_cp.suptitle(stat_name)#, fontsize=18)
    fig_cp.tight_layout()
    fig_cp.colorbar( cntr, ax=ax_cp )
    fig_cp.savefig( os.path.join(dir_figure, 'contour-plot--' + stat_name + '.png' ))
    plt.close(fig_cp)
    
    dataMean = pandas.DataFrame(data.groupby(params)[stat_name].agg('mean'))
    dataMean['std'] = data.groupby(params)[stat_name].agg('std')
        
    return    
    
def single_stat_analysis_time_dependent( stat_name, data, params):

    n_params = len(params)
    # Draw scatter plot
    fig_sp, ax_sp = plt.subplots( n_params, 1, figsize=(16,16) )
    for i, param in enumerate(params):
        # data.plot.scatter( param, stat_name, c='Week', cmap='viridis', ax=ax_sp[i] )
        seaborn.scatterplot(data[data['Week']<10], x=param, y=stat_name, hue='Week', cmap='viridis', ax=ax_sp[i] )
    ax_sp[0].set_title(stat_name)
    fig_sp.tight_layout()
    fig_sp.savefig( os.path.join(dir_figure, 'scatter-plot--time--' + stat_name + '.png' ))
    plt.close(fig_sp)
    
    # # Draw box plot
    # fig_sp, ax_sp = plt.subplots( n_params, 1, figsize=(16,16) )
    # for i, param in enumerate(params):
    #     my_plot = seaborn.boxplot( data, x='', y=stat_name, hue=param, ax=ax_sp[i] )
    #     ax_sp[i].legend([], [], frameon=False)
    #     ax_sp[i].set_title(param)
    #     if model == 'birthDeath':
    #         my_plot.set_xticklabels(my_plot.get_xticklabels(), rotation=90)
    #     else:
    #         my_plot.set_ylabel(None)
    # ax_sp[0].set_title(stat_name)
    # fig_sp.supylabel(stat_name)
    # fig_sp.tight_layout()
    # fig_sp.savefig( os.path.join(dir_figure, 'box-plot--time--' + stat_name + '.png' ))
    # plt.close(fig_sp)

    # # Draw contour plot
    # fig_cp, ax_cp = plt.subplots( n_params, 1, figsize=(20,16) )
    # fig_cp.set_figheight(15)
    # fig_cp.set_figwidth(18)
    # for i in range(n_params):
    #     # ax_cp[0,i].set_title(stat_name)
    #     idx = ~numpy.isnan(data[stat_name])
    #     cntr = ax_cp[i].tricontourf( data['Week'].values[idx],
    #                               data[params[i]].values[idx],
    #                               data[stat_name].values[idx],
    #                               levels = 10
    #                              )
    #     ax_cp[i].set_ylabel(params[i])
    # fig_cp.supxlabel('Week')
    # fig_cp.suptitle(stat_name)#, fontsize=18)
    # fig_cp.tight_layout()
    # fig_cp.colorbar( cntr, ax=ax_cp )
    # fig_cp.savefig( os.path.join(dir_figure, 'contour-plot--time--' + stat_name + '.png' ))
    # plt.close(fig_cp)
        
    return    
    
if __name__ == "__main__":
    main()
