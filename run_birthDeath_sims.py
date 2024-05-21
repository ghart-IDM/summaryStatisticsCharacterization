'''
Grid sweep for computing summary statistics of simulated phylogenetic trees. The
trees are generated using a birth-death process.
'''
import os
import time
import numpy 
import pandas

import ngesh
import optuna

from getTreeStats import getTreeStats





#-------------------------------------------------------------------------------
# Parameters
#-------------------------------------------------------------------------------

# Tree generation
birth_rate_bounds = [ 0.25, 10 ]
death_rate_bounds = [ 0.25, 10 ]
step_size = 0.25
n_leaves_bounds = [250, 250]
step_size_leaves = 50   

# Summary statistics
path_to_summary_statistics = '/home/ghart/Codes/phyloModels/phylomodels/features/trees'


# Optimization
n_trials = 1345#1*1600 # Number of runs
n_jobs   = 1  # Number of cores for parallel runs; use -1 for all cores
name     = "birthDeath-tree-summary-stats-rev0"
storage  = f'sqlite:///{name}.db'
#-------------------------------------------------------------------------------




#-------------------------------------------------------------------------------
# Experiment definition
#-------------------------------------------------------------------------------
def run_experiment( params=None ):
    ''' Definition of a single experiment'''

    # Set parameters
    if params == None:
        params = {}
    min_leaves = params.get( 'n_leaves'  , 50  )
    birth      = params.get( 'birth_rate', 1   )
    death      = params.get( 'death_rate', 0.5 )
    seed       = params.get( 'rand_seed' , 0   )

    # Generate tree
    print( "... generating tree with birth = ", birth, ", death = ", death,
            ", and n_leaves = ", min_leaves, ' Trial: ', seed )
    tic = time.time()
    try:
        tree = ngesh.random_tree.gen_tree( min_leaves = min_leaves,
                                           birth      = birth,
                                           death      = death,
                                           seed       = seed
                                          )
    except RuntimeError as e:
        print( "... ERROR: generating tree with birth = ", birth, 
               " and death = ", death, " --- ", e )
        return -1, None, None
    toc = time.time() - tic
    
    # Compute summary statistics
    print( "... computing summary statistics for tree with birth = ", birth, 
            ", death = ", death, ", and n_leaves = ", min_leaves, ' Trial: ', seed )
       
    tree_summary_stats = getTreeStats(tree)    
    # Finalize and return    
    status = 0
    return status, tree_summary_stats, toc




def objective(trial):
    ''' Define the objective for Optuna '''
    params = {}

    params['birth_rate'] = trial.suggest_float( 'birth_rate', 
                                                birth_rate_bounds[0], 
                                                birth_rate_bounds[1],
                                                step = step_size
                                               )
    params['death_rate'] = trial.suggest_float( 'death_rate', 
                                                death_rate_bounds[0], 
                                                death_rate_bounds[1],
                                                step = step_size
                                               )
    params['n_leaves']   = trial.suggest_int(    'n_leaves', 
                                                 n_leaves_bounds[0], 
                                                 n_leaves_bounds[1],
                                                 step = step_size_leaves
                                               )
    params['rand_seed' ] = trial.number
    
    status, out, toc_tree = run_experiment( params )
    
    if status < 0:
        cost = numpy.inf
    else:
        cost = 0
        trial.set_user_attr( 'time_tree_generation', toc_tree )
        for key, value in out.iloc[0].to_dict().items():
            trial.set_user_attr( key, value )
    

    
    return cost




def make_study():
    ''' Make a study '''

    
    birth_num = 1+int( (birth_rate_bounds[1] - birth_rate_bounds[0])/step_size )
    death_num = 1+int( (death_rate_bounds[1] - death_rate_bounds[0])/step_size )
    leaf_num = 1+int( (n_leaves_bounds[1] - n_leaves_bounds[0])/step_size_leaves )
    
    search_space = { 'birth_rate': numpy.linspace( birth_rate_bounds[0], 
                                                   birth_rate_bounds[1],
                                                   birth_num
                                                  ),
                     'death_rate': numpy.linspace( death_rate_bounds[0], 
                                                   death_rate_bounds[1],
                                                   death_num
                                                  ),
                     'n_leaves': numpy.linspace(   n_leaves_bounds[0], 
                                                   n_leaves_bounds[1],
                                                   leaf_num
                                                  ),
                    }
    
    sampler = optuna.samplers.GridSampler( search_space )
    output = optuna.create_study( sampler        = sampler,
                                  storage        = storage, 
                                  study_name     = name, 
                                  load_if_exists = True
                                 )
    
    return output




if __name__ == '__main__':
    
    make_study()
    study = optuna.load_study(storage=storage, study_name=name)
    output = study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    best_pars = study.best_params
    print('best_pars = ', best_pars)
