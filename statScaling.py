#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 21 May 2024

@author: ghart
"""


import os
import numpy as np
import pandas as pd
from time import perf_counter
from ete3 import Tree
from staticRenaming import stats_mapping
import phylomodels.features.trees
os.environ['OPENBLAS_NUM_THREADS'] = '8'


fileName = 'timings.csv'
if os.path.exists(fileName):
    timings = pd.read_csv(fileName)
else:
    timings = pd.DataFrame(columns=['statName', 'group', 'n', 'time'])

timing_list = [timings]

num_iters = 50
sizes = [50]

for stat_name, properties in stats_mapping.items():
    for size in sizes:
        for i in np.arange(num_iters):
            tree = Tree()
            tree.populate(size=size, random_branches=True, branch_range=(0.1, 1.0))
            for i, node in enumerate(tree.traverse('levelorder')):
                node.name = str(i)
            moduleName = getattr( phylomodels.features.trees,     properties['functionName'] )
            subroutine = getattr(moduleName, properties['functionName'])
            tic = perf_counter()
            stat = subroutine(tree)
            toc = perf_counter()
            timing_list.append(pd.DataFrame({'statName': [stat_name], 'group': [properties['group']], 'n': [size], 'time': [toc - tic]}))

timings = pd.concat([timings] + timing_list)
timings.to_csv('timings.csv', index=False)