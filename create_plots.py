'''
Generate figures based on the results rendered by the run_experiments.py script.
'''
import os
import numpy 
import pandas

import statsmodels.api as sm
from statsmodels.formula.api import ols

import optuna

import seaborn
import matplotlib
import matplotlib.pyplot as plt



#-------------------------------------------------------------------------------
# Parameters
#-------------------------------------------------------------------------------

# Optimization
#name     = "20210929-tree-summary-stats-test"
name     = "birthDeath-tree-summary-stats-rev0"
storage  = f'sqlite:///{name}.db'
dir_figure = 'figures/birthDeath'

seaborn.set(font_scale=1.8)
#-------------------------------------------------------------------------------



def main():

    # Load study and get dataframe
    study = optuna.load_study(storage=storage, study_name=name)
    study_df = study.trials_dataframe()
    
    # # Remove failed sims
    # study_df = study_df.loc[study_df.state=='COMPLETE', :]
    # study_df = study_df.loc[~numpy.isinf(study_df.value), :]
    
    columns = study_df.columns.str.startswith("params_") | study_df.columns.str.startswith("user_attrs_")
    data = study_df.loc[:, columns].copy()
    data = data.drop( columns = ["user_attrs_time_tree_generation"] )
    data.columns = data.columns.str.replace('_topology', '_unweighted')
    data.columns = data.columns.str.replace("^user_attrs_params_", "params_")
    params = data.iloc[0:2,:].filter(regex='^params_').columns.str.replace("params_","").to_list()
    features = data.iloc[0:2,:].filter(regex='^user_attrs_').columns.str.replace("user_attrs_", "").to_list()
    data.columns = data.columns.str.replace("params_", "")
    data.columns = data.columns.str.replace("user_attrs_", "")


    # print(data)    

    # print(data.columns)

    build_report( data, params, features )



    return
    
    

def build_report( data, params, features ):

    scores = multi_stat_analysis( data, params, features )
    

    # Individual Statistics
    print('scores.index = ', scores.index)
    n_params = len(params)
    for stat_name in scores.index[n_params:]:
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
    for stat_name in data.columns[n_params:]:
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
    corr = data.corr()
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
    ax_sp[0].set_title(stat_name)
    fig_sp.tight_layout()
    fig_sp.savefig( os.path.join(dir_figure, 'scatter-plot--' + stat_name + '.png' ))
    plt.close(fig_sp)
    
    # Draw box plot
    fig_sp, ax_sp = plt.subplots( n_params, 1, figsize=(16,16) )
    for i, param in enumerate(params):
        my_plot = seaborn.boxplot( data, x=param, y=stat_name, ax=ax_sp[i] )
        my_plot.set_xticklabels(my_plot.get_xticklabels(), rotation=90)
    ax_sp[0].set_title(stat_name)
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
    
    
    
if __name__ == "__main__":
    main()
