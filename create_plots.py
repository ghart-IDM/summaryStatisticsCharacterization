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
if model == 'birthDeath':
    title_name = 'Birth-Death'
elif model == 'SIR':
    title_name = 'SIR-like'
else:
    title_name = ''

# Optimization
name     = model + "-tree-summary-stats-rev0"
storage  = f'sqlite:///{name}.db'
dir_figure = os.path.join('figures', model)
csv_fileName = model + 'TrialData.csv'

seaborn.set(font_scale=1.8)
corr_method = 'spearman'
max_lag = 2
#-------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Statistic renaming
#------------------------------------------------------------------------------
stats_mapping = {'max_branch_length':              'Max branch length',                     	# Branch Length
                 'max_ex_branch_length':           'Max ext. branch length',                 	# Branch Length
                 'max_in_branch_length':           'Max int. branch length',                 	# Branch Length
                 'max_ratio_branch_length':        'Max ratio of branchs',                   	# Branch Length
                 'mean_branch_length':             'Mean branch length',                     	# Branch Length
                 'mean_ex_branch_length':          'Mean ext. branch length',               	# Branch Length
                 'mean_in_branch_length':          'Mean int. branch length',               	# Branch Length
                 'mean_ratio_branch_length':       'Mean ratio of branch lengths',              # Branch Length
                 'median_branch_length':           'Median branch length',        	            # Branch Length
                 'median_ex_branch_length':        'Median ext. branch length',               	# Branch Length
                 'median_in_branch_length':        'Median int. branch length',               	# Branch Length
                 'median_ratio_branch_length':     'Median ratio of branch lengths',            # Branch Length
                 'min_branch_length':              'Min branch length',                       	# Branch Length
                 'min_ex_branch_length':           'Min ext. branch length',                	# Branch Length
                 'min_in_branch_length':           'Min int. branch length',                	# Branch Length
                 'min_ratio_branch_length':        'Min ratio of branch lengths',             	# Branch Length
                 'std_branch_length':              'Std. dev. branch length',                	# Branch Length
                 'std_ex_branch_length':           'Std. dev. ext. branch length',            	# Branch Length
                 'std_in_branch_length':           'Std. dev. int. branch length',              # Branch Length
                 'std_ratio_branch_length':        'Std. dev. rotio of branch lengths',     	# Branch Length
                 'B1':                             'B1',                                       	# Depth
                 'B2':                             'B2',                                       	# Depth
                 'max_height':                     'Maximum height',                           	# Depth
                 'mean_height':                    'Mean height',                              	# Depth
                 'min_height':                     'Minimum height',                           	# Depth
                 'sackin':                         'Sackin index',                             	# Depth
                 'sackin_var':                     'Sackin variance',                          	# Depth
                 'cherries':                       'Cherries',                                 	# Local
                 'double_cherries':                'Double cherries',                          	# Local
                 'fourprong':                      'Fourprongs',                            	# Local
                 'frac_basal':                     'Fraction basal',                        	# Local
                 'mean_LBI':                       'Mean LBI',                                	# Local
                 'mean_NN_dist':                   'Mean nearest neighbor dist. weighted',   	# Local
                 'mean_NN_dist_unweighted':        'Mean nearest neighbor dist. unweighted',	# Local
                 'pitchforks':                     'Pitchforks',                            	# Local
                 'max_lineages':                   'Maximum lineages',                      	# LTT
                 'mean_b_time':                    'Mean branching time',                   	# LTT
                 'mean_s_time':                    'Mean sampling time',                    	# LTT
                 'slope_1':                        'Slope 1',                               	# LTT
                 'slope_2':                        'Slope 2',                               	# LTT
                 'slope_ratio':                    'Slope ratio',                           	# LTT
                 't_max_lineages':                 'Time of maximum lineages',              	# LTT
                 'betweenness_max':                'Max betweenness',                       	# Network Science
                 'closeness_max':                  'Max closeness weighted',                 	# Network Science
                 'closeness_max_unweighted':       'Max closeness unweighted',              	# Network Science
                 'diameter':                       'Diameter weighted',                     	# Network Science
                 'diameter_unweighted':            'Diameter unweighted',                   	# Network Science
                 'eigenvector_max':                'Eigenvector centrality weighted',       	# Network Science
                 'eigenvector_max_unweighted':     'Eigenvector centrality unweighted',       	# Network Science
                 'mean_path':                      'Mean path weighted',                    	# Network Science
                 'mean_path_unweighted':           'Mean path unweighted',                  	# Network Science
                 'eigen_gap':                      'Eigen gap weighted',                    	# Spectral
                 'eigen_gap_unweighted':           'Eigen gap unweighted',                  	# Spectral
                 'eigenvalue_max_adj':             'Max adj. eigenvalue weighted',          	# Spectral
                 'eigenvalue_max_adj_unweighted':  'Max adj. eigenvalue unweighted',         	# Spectral
                 'eigenvalue_max_dLap':            'Max dist. lap. eigenvalue weighted',    	# Spectral
                 'eigenvalue_max_dLap_unweighted': 'Max dist. lap. eigenvalue unweighted',     	# Spectral
                 'eigenvalue_max_lap':             'Max lap. eigenvalue weighted',          	# Spectral
                 'eigenvalue_max_lap_unweighted':  'Max lap. eigenvalue unweighted',        	# Spectral
                 'eigenvalue_min_adj':             'Min adj. eigenvalue weighted',          	# Spectral
                 'eigenvalue_min_adj_unweighted':  'Min adj. eigenvalue unweighted',        	# Spectral
                 'eigenvalue_min_lap':             'Min lap. eigenvalue weighted',          	# Spectral
                 'eigenvalue_min_lap_unweighted':  'Min lap. eigenvalue unweighted',        	# Spectral
                 'kurtosis':                       'Kurtosis weighted',                     	# Spectral
                 'kurtosis_unweighted':            'Kurtosis unweighted',                   	# Spectral
                 'skewness':                       'Skewness weighted',                     	# Spectral
                 'skewness_unweighted':            'Skewness unweighted',                   	# Spectral
                 'WD_ratio':                       'Maximum width-to-depth ratio',          	# Topology
                 'colless':                        'Colless index',                         	# Topology
                 'frac_imbalance':                 'Fration of imbalance',                  	# Topology
                 'frac_ladder':                    'Fraction of ladders',                   	# Topology
                 'max_dW':                         'Maximum midth change',                  	# Topology
                 'max_ladder':                     'Maximum ladder size',                   	# Topology
                 'mean_imbalance_ratio':           'Mean imbalance ratio'                   	# Topology
} 

stat_grouping = {'max_branch_length':              'Branch Length',
                 'max_ex_branch_length':           'Branch Length',
                 'max_in_branch_length':           'Branch Length',
                 'max_ratio_branch_length':        'Branch Length',
                 'mean_branch_length':             'Branch Length',
                 'mean_ex_branch_length':          'Branch Length',
                 'mean_in_branch_length':          'Branch Length',
                 'mean_ratio_branch_length':       'Branch Length',
                 'median_branch_length':           'Branch Length',
                 'median_ex_branch_length':        'Branch Length',
                 'median_in_branch_length':        'Branch Length',
                 'median_ratio_branch_length':     'Branch Length',
                 'min_branch_length':              'Branch Length',
                 'min_ex_branch_length':           'Branch Length',
                 'min_in_branch_length':           'Branch Length',
                 'min_ratio_branch_length':        'Branch Length',
                 'std_branch_length':              'Branch Length',
                 'std_ex_branch_length':           'Branch Length',
                 'std_in_branch_length':           'Branch Length',
                 'std_ratio_branch_length':        'Branch Length',
                 'B1':                             'Depth',
                 'B2':                             'Depth',
                 'max_height':                     'Depth',
                 'mean_height':                    'Depth',
                 'min_height':                     'Depth',
                 'sackin':                         'Depth',
                 'sackin_var':                     'Depth',
                 'cherries':                       'Local',
                 'double_cherries':                'Local',
                 'fourprong':                      'Local',
                 'frac_basal':                     'Local',
                 'mean_LBI':                       'Local',
                 'mean_NN_dist':                   'Local',
                 'mean_NN_dist_unweighted':        'Local',
                 'pitchforks':                     'Local',
                 'max_lineages':                   'LTT',
                 'mean_b_time':                    'LTT',
                 'mean_s_time':                    'LTT',
                 'slope_1':                        'LTT',
                 'slope_2':                        'LTT',
                 'slope_ratio':                    'LTT',
                 't_max_lineages':                 'LTT',
                 'betweenness_max':                'Network Science',
                 'closeness_max':                  'Network Science',
                 'closeness_max_unweighted':       'Network Science',
                 'diameter':                       'Network Science',
                 'diameter_unweighted':            'Network Science',
                 'eigenvector_max':                'Network Science',
                 'eigenvector_max_unweighted':     'Network Science',
                 'mean_path':                      'Network Science',
                 'mean_path_unweighted':           'Network Science',
                 'eigen_gap':                      'Spectral',
                 'eigen_gap_unweighted':           'Spectral',
                 'eigenvalue_max_adj':             'Spectral',
                 'eigenvalue_max_adj_unweighted':  'Spectral',
                 'eigenvalue_max_dLap':            'Spectral',
                 'eigenvalue_max_dLap_unweighted': 'Spectral',
                 'eigenvalue_max_lap':             'Spectral',
                 'eigenvalue_max_lap_unweighted':  'Spectral',
                 'eigenvalue_min_adj':             'Spectral',
                 'eigenvalue_min_adj_unweighted':  'Spectral',
                 'eigenvalue_min_lap':             'Spectral',
                 'eigenvalue_min_lap_unweighted':  'Spectral',
                 'kurtosis':                       'Spectral',
                 'kurtosis_unweighted':            'Spectral',
                 'skewness':                       'Spectral',
                 'skewness_unweighted':            'Spectral',
                 'WD_ratio':                       'Topology',
                 'colless':                        'Topology',
                 'frac_imbalance':                 'Topology',
                 'frac_ladder':                    'Topology',
                 'max_dW':                         'Topology',
                 'max_ladder':                     'Topology',
                 'mean_imbalance_ratio':           'Topology'
                 }

group_colors = {'Branch Length': 'b',
                'Depth': 'g',
                'Local': 'r',
                'LTT': 'c',
                'Network Science': 'm',
                'Spectral': 'y',
                'Topology': 'b'
                }

param_mapping = {'birth_rate': 'Birth rate',
                 'death_rate': 'Death rate',
                 'n_leaves':   'Num. leaves',
                 'R0': '$R_0$',
                 'contacts': 'Number\nof\ncontacts',
                 'infection_duration': 'Infectious\nduration',
                 'population_size': 'Total\npop.\nsize',
                 'sample_fraction': 'Sample\ncoverage',
                 'sample_time': 'Sample\ndelay',
                 'R_eff': '$R_e$',
                 'cum_incidence': 'Cumulative\nincidence',
                 'incidence': 'Incidence',
                 'prevalence': 'Prevalence'
                 }

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
    # time_dependent_params = [re.sub('R_eff_1([0-9]+)', r'R_eff_\1', x) for x in time_dependent_params]
    # data.rename(columns=lambda x: re.sub('R_eff_1([0-9]+)', r'R_eff_\1', x), inplace=True)
    params = data.iloc[0:2,:].filter(regex='^params_').columns.str.replace("params_","").to_list()
    data.columns = data.columns.str.replace("user_attrs_params_", "")
    features = data.iloc[0:2,:].filter(regex='^user_attrs_').columns.str.replace("user_attrs_", "").to_list()
    slopes = []
    if 'slope_1' in features:
        slopes.append('slope_1')
    if 'slope_2' in features:
        slopes.append('slope_2')
    time_dependent_features = [x for x in features if re.search('_[0-9]+$', x)]
    if len(slopes) > 0:
        for slope in slopes:
            time_dependent_features.remove(slope)
    features = [x for x in features if not re.search('_[0-9]+$', x)]
    if len(slopes) > 0:
        features = features + slopes
    data.columns = data.columns.str.replace("params_", "")
    data.columns = data.columns.str.replace("user_attrs_", "")
    data['sim'] = numpy.arange(data.shape[0])
    

    # build_report( data, params, features )

    if model == 'SIR':
        data = data.loc[:, ['sim'] + params + time_dependent_params + time_dependent_features]
        data.set_index(['sim'] + params, append=True, inplace=True)
        data.columns = data.columns.str.rsplit("_", n=1, expand=True)
        data = data.stack(level=1).rename_axis(['Index_0'] + ['sim'] + params + ['Week']).reset_index()
        data['Week'] = data['Week'].astype('int64')
        data['sim'] = data['sim'].astype('int64')
        data['cum_incidence'] = data['cum_incidence']/data['population_size']
        data['incidence'] = data['incidence']/data['population_size']
        data['prevalence'] = data['prevalence']/data['population_size']
        time_dependent_features = numpy.unique([x.rpartition('_')[0] for x in time_dependent_features]).tolist()
        time_dependent_params = numpy.unique([x.rpartition('_')[0] for x in time_dependent_params]).tolist()
    
        # Multi-stat analysis
        multi_stat_analysis_time_dependent( data, time_dependent_params, time_dependent_features )
        
        # # Individual Statistics
        # n_params = len(time_dependent_params)
        # for stat_name in time_dependent_features:
        #     print( '... processing (time dependent): ', stat_name )
        #     stat_df = data.loc[ :, ['sim', 'Week'] + time_dependent_params + [stat_name ]]
        #     if stat_df is not None:
        #         single_stat_analysis_time_dependent( stat_name, stat_df.fillna(0).clip(-1e6, 1e6),
        #                               time_dependent_params)

    
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
    gof_name = "Multiple\ncorrelation"
    rename_feature = []
    for feature in features:
        rename_feature.append(stats_mapping[feature])
    scores = pandas.DataFrame( columns=["group"] + [param_mapping[x] for x in params] + [gof_name],
                              index=rename_feature, dtype=numpy.float64)
    scores['name'] = rename_feature
    scores.set_index("name", inplace=True)
    for param in params:
        scores[param_mapping[param]] = numpy.empty(n_stats)

    # Compute scores
    for stat_name in features:
        stat_df = data.loc[ :, params + [stat_name ]]
        if stat_df is not None:
            scores.loc[stats_mapping[stat_name], "group"] = stat_grouping[stat_name]

            for param in params:
                scores.loc[stats_mapping[stat_name], param_mapping[param]] = stat_df[stat_name].corr( stat_df[param], method=corr_method )
                

            model = ols(stat_name + " ~ " + " + ".join(params), data=stat_df).fit()
            scores.loc[stats_mapping[stat_name], gof_name] = model.rsquared**.5

    scores.dropna(inplace=True)
    # scores.index = scores.index.str.replace('_topology', '_unweighted')
    scores["R_abs"] = scores[gof_name].abs()
    scores = scores.groupby(['group'], as_index=True).apply(lambda x: x.sort_values( "R_abs", ascending=False )).droplevel('group', axis='index')    #.replace([numpy.inf, -numpy.inf], numpy.nan, inplace=True) \
    counts = scores.groupby('group', as_index=False)['R_abs'].count().rename(columns={'R_abs': 'count'})
                   
    print('scores = \n', scores)


    # Plot correlations
    fig_corr, ax_corr = plt.subplots( n_params+1, 1, figsize=(18,20), sharex=True )
    scores.plot.bar( y=gof_name, ax=ax_corr[0] )
    for i, param in enumerate(params):
        scores.plot.bar( y=param_mapping[param], ax=ax_corr[i + 1] )
    ax_corr[-1].set_xlabel('')
    # Set text color by group
    xticklabels = ax_corr[-1].get_xticklabels()
    for label in xticklabels:
        label.set_color(group_colors[scores.loc[label.get_text(), 'group']])
    # Create grouping axis
    ax2 = ax_corr[-1].twiny()
    # ax2.tick_params(axis='x', which='minor', rotation=90)
    ax2.spines['bottom'].set_position(('axes', -2.3))
    ax2.spines['bottom'].set_color('black')
    ax2.tick_params('x', length=0, width=0, which='minor')
    ax2.tick_params('x', direction='in', which='major')
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.set_xticks([0] + counts['count'].cumsum().tolist())
    ax2.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    ax2.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator((counts['count'].cumsum() - counts['count'][::-1]/2)))
    ax2.xaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter(counts['group'].tolist()))
    for label in ax2.get_xticklabels(minor=True):
        label.set_horizontalalignment('center')
    fig_corr.tight_layout()
    plt.savefig(os.path.join(dir_figure, 'correlations.png'))
    plt.close(fig_corr)

    # Fixing the annot to have all values may require downgrading MATPLOTLIB to 3.7.3
    fig_table, ax_table = plt.subplots( 1, 1, figsize=(22,35) )
    seaborn.heatmap( scores.drop(columns=["R_abs", "group"]),
                     vmin       = -1,
                     vmax       = 1,
                     cmap       = "RdBu",
                     annot      = True, 
                     linewidths = .5, 
                     ax         = ax_table
                    )
    ticks = numpy.arange(0.5, len(scores.index), 1)
    ax_table.set_title('Summary Statistics vs. ' + title_name + ' Model Parameters: Correlation Analysis\n')
    ax_table.xaxis.tick_top()
    ax_table.xaxis.set_label_position('top')
    # xticklabels = ax_table.get_xticklabels()
    # for label in xticklabels:
    #     label.set_text(param_mapping[label.get_text()])
    ax_table.set_ylabel('')
    ax_table.set_yticks( ticks )
    ax_table.set_yticklabels( scores.index )
    # Set text color by group
    yticklabels = ax_table.get_yticklabels()
    for label in yticklabels:
        label.set_color(group_colors[scores.loc[label.get_text(), 'group']])
    # Create grouping axis
    ax2 = ax_table.twinx()
    ax2.tick_params(axis='y', which='minor', rotation=90)
    ax2.spines['left'].set_position(('axes', -0.5))
    ax2.spines['left'].set_color('black')
    ax2.tick_params('y', length=0, width=0, which='minor')
    ax2.tick_params('y', direction='in', which='major')
    ax2.yaxis.set_ticks_position("left")
    ax2.yaxis.set_label_position("left")
    ax2.set_yticks([0] + counts.loc[::-1, 'count'].cumsum().tolist())
    ax2.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    ax2.yaxis.set_minor_locator(matplotlib.ticker.FixedLocator((counts.loc[::-1, 'count'].cumsum() - counts['count'][::-1]/2)))
    ax2.yaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter(counts.loc[::-1, 'group'].tolist()))
    for label in ax2.get_yticklabels(minor=True):
        label.set_verticalalignment('center')
    fig_table.tight_layout()
    plt.savefig(os.path.join(dir_figure, 'scoreRanking.png'))
    plt.close(fig_table)


    # fig_pc, ax_pc = plt.subplots( 1, 1, figsize=(24,24) )
    corr = data[features].rename(columns=stats_mapping).corr(method=corr_method)
    corr.dropna(inplace=True, how='all')
    corr.dropna(inplace=True, how='all', axis=1)
    cg = seaborn.clustermap(corr, cmap="RdBu", vmin=-1, vmax=1, figsize=(24,24) )
    cg.ax_row_dendrogram.set_visible(False)
    cg.ax_col_dendrogram.set_visible(False)
    # cax = ax_pc.matshow( corr, cmap="BrBG", vmin=-1, vmax=1 )
    # fig_pc.colorbar(cax)
    ticks = numpy.arange( 0, len(corr.columns), 1 )
    cg.ax_heatmap.set_xticks( ticks )
    # plt.xticks( rotation=90 )
    cg.ax_heatmap.set_yticks( ticks )
    cg.ax_heatmap.set_xticklabels( corr.columns )
    cg.ax_heatmap.set_yticklabels( corr.columns )
    yticklabels = cg.ax_heatmap.get_yticklabels()
    for label in yticklabels:
        label.set_color(group_colors[scores.loc[label.get_text(), 'group']])
    xticklabels = cg.ax_heatmap.get_xticklabels()
    for label in xticklabels:
        label.set_color(group_colors[scores.loc[label.get_text(), 'group']])
    
    # cg.tight_layout()
    plt.savefig(os.path.join(dir_figure, 'correlations_features.png'))
    plt.close('all')


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
    ax_sp[0].set_title(stats_mapping[stat_name])
    fig_sp.supylabel(stats_mapping[stat_name])
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
    fig_cp.suptitle(stats_mapping[stat_name])#, fontsize=18)
    fig_cp.tight_layout()
    fig_cp.colorbar( cntr, ax=ax_cp )
    fig_cp.savefig( os.path.join(dir_figure, 'contour-plot--' + stat_name + '.png' ))
    plt.close(fig_cp)
    
    dataMean = pandas.DataFrame(data.groupby(params)[stat_name].agg('mean'))
    dataMean['std'] = data.groupby(params)[stat_name].agg('std')
        
    return    
    
def multi_stat_analysis_time_dependent( data, params, features ):

    # Initialize
    n_stats = len(features) 
    n_params = len(params)
    gof_name = "Multiple\ncorrelation"
    rename_feature = []
    for feature in features:
        rename_feature.append(stats_mapping[feature])
    scores = pandas.DataFrame( columns=["group"] + [param_mapping[x] for x in params] + [gof_name],
                              index=rename_feature, dtype=numpy.float64)
    
    scores['name'] = rename_feature
    scores.set_index("name", inplace=True)
    lags = pandas.DataFrame( columns=["group"] + [param_mapping[x] for x in params],
                              index=rename_feature, dtype=numpy.int64)
    lags['name'] = rename_feature
    lags.set_index("name", inplace=True)
    for param in params:
        scores[param_mapping[param]] = numpy.empty(n_stats)
        lags[param_mapping[param]] = numpy.empty(n_stats)


    # Compute scores
    for stat_name in features:
        stat_df = data.loc[ :, ['sim'] + params + [stat_name ]]
        if stat_df is not None:
            scores.loc[stats_mapping[stat_name], "group"] = stat_grouping[stat_name]
            lags.loc[stats_mapping[stat_name], "group"] = stat_grouping[stat_name]

            for param in params:
                lag = 0
                corr = 0
                temp = stat_df.groupby('sim')[[stat_name, param]].corr(method=corr_method).reset_index()
                temp = temp.loc[temp['level_1']==param,['sim', stat_name]].set_index('sim')
                temp['size'] = stat_df.groupby('sim')[param].agg('count')
                corr = numpy.nansum(temp[stat_name]*temp['size'])/numpy.nansum((~numpy.isnan(temp[stat_name]))*temp['size'])
                for lag_i in numpy.arange(1, max_lag+1):
                    temp = stat_df[['sim', param, stat_name]]
                    temp[param] = temp.groupby('sim', as_index=False)[param].apply(lambda x: x.shift(1, fill_value=x.iloc[0])).values
                    temp = temp.groupby('sim').corr(method=corr_method).reset_index()
                    temp = temp.loc[temp['level_1']==param,['sim', stat_name]].set_index('sim')
                    temp['size'] = stat_df.groupby('sim')[param].agg('count')
                    corr_temp = numpy.nansum(temp[stat_name]*temp['size'])/numpy.nansum((~numpy.isnan(temp[stat_name]))*temp['size'])
                    if abs(corr_temp) > abs(corr):
                        corr = corr_temp
                        lag = lag_i
                for lag_i in numpy.arange(-1, -1*(max_lag+1), -1):
                    temp = stat_df[['sim', param, stat_name]]
                    temp[param] = temp.groupby('sim', as_index=False)[param].apply(lambda x: x.shift(1, fill_value=x.iloc[-1])).values
                    temp = temp.groupby('sim').corr(method=corr_method).reset_index()
                    temp = temp.loc[temp['level_1']==param,['sim', stat_name]].set_index('sim')
                    temp['size'] = stat_df.groupby('sim')[param].agg('count')
                    corr_temp = numpy.nansum(temp[stat_name]*temp['size'])/numpy.nansum((~numpy.isnan(temp[stat_name]))*temp['size'])
                    if abs(corr_temp) > abs(corr):
                        corr = corr_temp
                        lag = lag_i
                
                scores.loc[stats_mapping[stat_name], param_mapping[param]] = corr
                lags.loc[stats_mapping[stat_name], param_mapping[param]] = lag
            model = ols(stat_name + " ~ " + " + ".join(params), data=stat_df).fit()
            scores.loc[stats_mapping[stat_name], gof_name] = model.rsquared**.5
                
    # # Compute scores
    # for stat_name in features:
    #     stat_df = data.loc[ :, params + [stat_name ]]
    #     if stat_df is not None:
    #         scores.loc[stats_mapping[stat_name], "group"] = stat_grouping[stat_name]

    #         for param in params:
    #             scores.loc[stats_mapping[stat_name], param_mapping[param]] = stat_df[stat_name].corr( stat_df[param], method=corr_method )    

    #         model = ols(stat_name + " ~ " + " + ".join(params), data=stat_df).fit()
    #         scores.loc[stat_name, gof_name] = model.rsquared**.5
    
    # scores = pandas.read_csv('scores_time_dependent.csv')
    # lags = pandas.read_csv('lags_time_dependent.csv')
    scores.dropna(inplace=True)
    # scores.index = scores.index.str.replace('_topology', '_unweighted')
    scores["R_abs"] = scores[gof_name].abs()
    lags['R_abs'] = scores[gof_name].abs()
    # scores = scores.sort_values( "R_abs", ascending=False )    #.replace([numpy.inf, -numpy.inf], numpy.nan, inplace=True) \
    scores = scores.groupby(['group'], as_index=True).apply(lambda x: x.sort_values( "R_abs", ascending=False )).droplevel('group', axis='index')
    counts = scores.groupby('group', as_index=False)['R_abs'].count().rename(columns={'R_abs': 'count'})
    # lags = lags.groupby(['group'], as_index=True).apply(lambda x: x.sort_values( "R_abs", ascending=False )).droplevel('group', axis='index')

    scores.to_csv('scores_time_dependent.csv')
    lags.to_csv('lags_time_dependent.csv')

    print('scores = \n', scores)


    # Plot correlations
    fig_corr, ax_corr = plt.subplots( n_params+1, 1, figsize=(18,20), sharex=True )
    scores.plot.bar( y=gof_name, ax=ax_corr[0] )
    for i, param in enumerate(params):
        scores.plot.bar( y=param_mapping[param], ax=ax_corr[i + 1] )
    ax_corr[-1].set_xlabel('')
    xticklabels = ax_corr[-1].get_xticklabels()
    for label in xticklabels:
        label.set_color(group_colors[scores.loc[label.get_text(), 'group']])
    # Create grouping axis
    ax2 = ax_corr[-1].twiny()
    # ax2.tick_params(axis='x', which='minor', rotation=90)
    ax2.spines['bottom'].set_position(('axes', -2.3))
    ax2.spines['bottom'].set_color('black')
    ax2.tick_params('x', length=0, width=0, which='minor')
    ax2.tick_params('x', direction='in', which='major')
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.set_xticks([0] + counts['count'].cumsum().tolist())
    ax2.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    ax2.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator((counts['count'].cumsum() - counts['count'][::-1]/2)))
    ax2.xaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter(counts['group'].tolist()))
    for label in ax2.get_xticklabels(minor=True):
        label.set_horizontalalignment('center')
    fig_corr.tight_layout()
    plt.savefig(os.path.join(dir_figure, 'correlations_time_dependent.png'))
    plt.close(fig_corr)
    

    # Fixing the annot to have all values may require downgrading MATPLOTLIB to 3.7.3
    fig_table, ax_table = plt.subplots( 1, 1, figsize=(17,20) )
    seaborn.heatmap( scores.drop(columns=["R_abs", "group", gof_name]),
                     vmin       = -1,
                     vmax       = 1,
                     cmap       = "RdBu",
                     annot      = True, 
                     linewidths = .5, 
                     ax         = ax_table
                    )
    ticks = numpy.arange(0.5, len(scores.index), 1)
    ax_table.set_title('Summary Statistics vs. ' + title_name + ' Model Parameters: Time Dependent Correlation Analysis\n')
    ax_table.xaxis.tick_top()
    ax_table.xaxis.set_label_position('top')
    ax_table.set_ylabel('')
    ax_table.set_yticks( ticks )
    ax_table.set_yticklabels( scores.index )
    # Set text color by group
    yticklabels = ax_table.get_yticklabels()
    for label in yticklabels:
        label.set_color(group_colors[scores.loc[label.get_text(), 'group']])
    # Create grouping axis
    ax2 = ax_table.twinx()
    ax2.tick_params(axis='y', which='minor', rotation=90)
    ax2.spines['left'].set_position(('axes', -0.5))
    ax2.spines['left'].set_color('black')
    ax2.tick_params('y', length=0, width=0, which='minor')
    ax2.tick_params('y', direction='in', which='major')
    ax2.yaxis.set_ticks_position("left")
    ax2.yaxis.set_label_position("left")
    ax2.set_yticks([0] + counts.loc[::-1, 'count'].cumsum().tolist())
    ax2.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    ax2.yaxis.set_minor_locator(matplotlib.ticker.FixedLocator((counts.loc[::-1, 'count'].cumsum() - counts['count'][::-1]/2)))
    ax2.yaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter(counts.loc[::-1, 'group'].tolist()))
    for label in ax2.get_yticklabels(minor=True):
        label.set_verticalalignment('center')
    fig_table.tight_layout()
    plt.savefig(os.path.join(dir_figure, 'scoreRanking_time_dependent.png'))
    plt.close(fig_table)
    # scores.to_csv('scores_time_dependent.csv')
    
    # Fixing the annot to have all values may require downgrading MATPLOTLIB to 3.7.3
    fig_table, ax_table = plt.subplots( 1, 1, figsize=(17,20) )
    seaborn.heatmap( lags.drop(columns='group'),
                     vmin       = -1*max_lag,
                     vmax       = max_lag,
                     cmap       = "RdBu",
                     annot      = True, 
                     linewidths = .5, 
                     ax         = ax_table
                    )
    ticks = numpy.arange(0.5, len(lags.index), 1)
    ax_table.set_title('Summary Statistics vs. ' + title_name + ' Model Parameters: Best Lag for Correlation\n')
    ax_table.xaxis.tick_top()
    ax_table.xaxis.set_label_position('top')
    # xticklabels = ax_table.get_xticklabels()
    # for label in xticklabels:
    #     label.set_text(param_mapping[label.get_text()])
    ax_table.set_ylabel('')
    ax_table.set_yticks( ticks )
    ax_table.set_yticklabels( lags.index )
    # Set text color by group
    yticklabels = ax_table.get_yticklabels()
    for label in yticklabels:
        label.set_color(group_colors[lags.loc[label.get_text(), 'group']])
    # Create grouping axis
    ax2 = ax_table.twinx()
    ax2.tick_params(axis='y', which='minor', rotation=90)
    ax2.spines['left'].set_position(('axes', -0.5))
    ax2.spines['left'].set_color('black')
    ax2.tick_params('y', length=0, width=0, which='minor')
    ax2.tick_params('y', direction='in', which='major')
    ax2.yaxis.set_ticks_position("left")
    ax2.yaxis.set_label_position("left")
    ax2.set_yticks([0] + counts.loc[::-1, 'count'].cumsum().tolist())
    ax2.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    ax2.yaxis.set_minor_locator(matplotlib.ticker.FixedLocator((counts.loc[::-1, 'count'].cumsum() - counts['count'][::-1]/2)))
    ax2.yaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter(counts.loc[::-1, 'group'].tolist()))
    for label in ax2.get_yticklabels(minor=True):
        label.set_verticalalignment('center')
    fig_table.tight_layout()
    plt.savefig(os.path.join(dir_figure, 'best_lags.png'))
    plt.close(fig_table)
    # lags.to_csv('lags_time_dependent.csv')


    fig_table, ax_table = plt.subplots( 1, 1, figsize=(22,35) )
    seaborn.heatmap( scores.drop(columns=["R_abs", "group"]),
                     vmin       = -1,
                     vmax       = 1,
                     cmap       = "RdBu",
                     annot      = True, 
                     linewidths = .5, 
                     ax         = ax_table
                    )
    ticks = numpy.arange(0.5, len(scores.index), 1)
    ax_table.set_title('Summary Statistics vs. ' + title_name + ' Model Parameters: Correlation Analysis\n')
    ax_table.xaxis.tick_top()
    ax_table.xaxis.set_label_position('top')
    # xticklabels = ax_table.get_xticklabels()
    # for label in xticklabels:
    #     label.set_text(param_mapping[label.get_text()])
    ax_table.set_ylabel('')
    ax_table.set_yticks( ticks )
    ax_table.set_yticklabels( scores.index )
    # Set text color by group
    yticklabels = ax_table.get_yticklabels()
    for label in yticklabels:
        label.set_color(group_colors[scores.loc[label.get_text(), 'group']])
    # Create grouping axis
    ax2 = ax_table.twinx()
    ax2.tick_params(axis='y', which='minor', rotation=90)
    ax2.spines['left'].set_position(('axes', -0.5))
    ax2.spines['left'].set_color('black')
    ax2.tick_params('y', length=0, width=0, which='minor')
    ax2.tick_params('y', direction='in', which='major')
    ax2.yaxis.set_ticks_position("left")
    ax2.yaxis.set_label_position("left")
    ax2.set_yticks([0] + counts.loc[::-1, 'count'].cumsum().tolist())
    ax2.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    ax2.yaxis.set_minor_locator(matplotlib.ticker.FixedLocator((counts.loc[::-1, 'count'].cumsum() - counts['count'][::-1]/2)))
    ax2.yaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter(counts.loc[::-1, 'group'].tolist()))
    for label in ax2.get_yticklabels(minor=True):
        label.set_verticalalignment('center')
    fig_table.tight_layout()
    plt.savefig(os.path.join(dir_figure, 'scoreRanking.png'))
    plt.close(fig_table)


    return scores

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
