'''
Grid sweep for computing summary statistics of simulated phylogenetic trees. The
trees are generated using a birth-death process.
'''
import os
import time
import numpy 
import pandas

import optuna

from getTreeStats import getTreeStats
from phylomodels.trees.generate_treeFromFile import read_treeFromLineList
from phylomodels.trees.transform_joinTrees import transform_joinTrees
from phylomodels.trees.transform_transToPhyloTree import transform_transToPhyloTree




#-------------------------------------------------------------------------------
# Parameters
#-------------------------------------------------------------------------------

# Tree generation
param_names = ['contacts',
               'population_size',
               'infection_duration',
               'R0',
               'sample_time',
               'sample_fraction']
param_bounds = {'contacts':           [4, 8],
                'population_size':    [100, 300],
                'infection_duration': [3, 11],
                'R0':                 [2, 8],
                'sample_time':        [0, 14],
                'sample_fraction':    [.5, 1]}
param_step_size = {'contacts':           4,
                   'population_size':    100,
                   'infection_duration': 4,
                   'R0':                 4,
                   'sample_time':        7,
                   'sample_fraction':    0.25}


# Optimization
n_trials = 25  # Number of runs
n_jobs   = 2  # Number of cores for parallel runs; use -1 for all cores
name     = "SIR-tree-summary-stats-rev0"
storage  = f'sqlite:///{name}.db'
#-------------------------------------------------------------------------------




#-------------------------------------------------------------------------------
# Experiment definition
#-------------------------------------------------------------------------------
def run_experiment( params=None ):
    ''' Definition of a single experiment'''

    # Set up commend to run simulation with the given parameters
    cmd = "./flu --quiet"
    if "population_size" in params:
        cmd += " --pop:" + str(int(params["population_size"]))
    if "contacts" in params:
       cmd += " --c:" + str(params["contacts"])
    if "infection_duration" in params:
        cmd += " --d:" + str(params["infection_duration"])
    cmd += " --t:150"
    if "R0" in params:
        cmd += " --r0:" + str(params["R0"])
    if "seed" in params:
        seed = params["seed"]
        cmd += " --seed:" + str(seed)      
    else:
        seed = 0
    cmd += " --output:transmissions" + str(seed) + ".bin"
    outputFile = 'transmissions' + str(seed) + '.bin'

    # Run sim
    tic = time.time()

    print( "... running sim with ", params)
    os.system(cmd)

    # Read in sim output and prepare for tree
    lineList = numpy.fromfile(outputFile, dtype=numpy.uint32)
    rows = lineList.shape[0] // 3
    lineList = lineList.reshape((rows,3))
    lineList = pandas.DataFrame(data=lineList, index=None, columns=['infectedById', 'id', 'timeInfected'])
    os.system('rm transmissions' + str(seed) + '.bin')

    # Check if an outbreak actually happened
    if lineList.shape[0] < 15:
        print( "... ERROR: running sim with birth ", params )
        return -1, None, None
    toc = time.time() - tic
    
    print( "... building tree with ", params)
    lineList['id'] = lineList['id'].astype(str)
    lineList['infectedById'] = lineList['infectedById'].astype(str)
    # Set sample time and sample
    lineList['sampledTime'] = lineList['timeInfected'] + params['sample_time']
    nodes_to_keep = lineList['id'].sample(n=int(numpy.floor(lineList.shape[0]*params['sample_fraction'])), replace=False)
    lineList.loc[~lineList['id'].isin(nodes_to_keep), 'sampledTime'] = numpy.nan
    # Build transmission trees (there should only be one)
    trees = read_treeFromLineList(lineList, sampleTime='sampledTime')
    for i in range(len(trees)):
        trees[i] = transform_transToPhyloTree(trees[i])
    # combined transmission tree(s)
    tree = transform_joinTrees(trees, method='constant_coalescent')

    # Compute summary statistics
    print( "... computing summary statistics for tree with ", params )
    tree_summary_stats = getTreeStats(tree)

    # Finalize and return    
    status = 0
    return status, tree_summary_stats, toc




def objective(trial):
    ''' Define the objective for Optuna '''
    params = {}

    for param in param_names:
        params[param] = trial.suggest_float(param,
                                            param_bounds[param][0],
                                            param_bounds[param][1],
                                            step=param_step_size[param])
    params['seed' ] = trial.number
    
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

    search_space = {}
    for param in param_names:
        param_num = 1 + int( (param_bounds[param][1] - param_bounds[param][0])/param_step_size[param] )
        search_space[param] = numpy.linspace( param_bounds[param][0], 
                                              param_bounds[param][1],
                                              param_num
                                            )
    
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
