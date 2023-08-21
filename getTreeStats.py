#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 13:18:53 2023

@author: ghart
"""
import numpy as np
import pandas as pd
from ete3 import Tree
from time import perf_counter
from phylomodels.features.trees import *
from phylomodels.features.trees.helper.unique_node_attr import unique_node_attr
from phylomodels.features.trees.helper.get_LTT import get_LTT
from phylomodels.features.trees.helper.get_eigenvalues_dist_lap import get_eigenvalues_dist_lap
from phylomodels.features.trees.helper.get_eigenvalues_lap import get_eigenvalues_lap
from phylomodels.features.trees.helper.get_eigenvalues_adj import get_eigenvalues_adj
from phylomodels.features.trees.helper.get_distance_mat import get_distance_mat
from phylomodels.features.trees.helper.get_adjacency_mat import get_adjacency_mat

def getTreeStats(tree):
    tree_id = 'tree'
    treeData = pd.DataFrame( {"tree_id": [tree_id]} ).set_index("tree_id")
    
    kwargs = {}
    #kwargs = {'attr': 'population'}
    #kwargs.update(unique_node_attr(tree, 'population'))
    
    print('Running branch length summary statistics.')
    # All branch lengths
    stat = BL_calculate_min.BL_calculate_min(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = BL_calculate_max.BL_calculate_max(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = BL_calculate_mean.BL_calculate_mean(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = BL_calculate_median.BL_calculate_median(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = BL_calculate_std.BL_calculate_std(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    # External branch lengths
    stat = BL_calculate_external_min.BL_calculate_external_min(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = BL_calculate_external_max.BL_calculate_external_max(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = BL_calculate_external_mean.BL_calculate_external_mean(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = BL_calculate_external_median.BL_calculate_external_median(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = BL_calculate_external_std.BL_calculate_external_std(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    # Internal branch lengths
    stat = BL_calculate_internal_min.BL_calculate_internal_min(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = BL_calculate_internal_max.BL_calculate_internal_max(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = BL_calculate_internal_mean.BL_calculate_internal_mean(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = BL_calculate_internal_median.BL_calculate_internal_median(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = BL_calculate_internal_std.BL_calculate_internal_std(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    # ratio branch lengths
    stat = BL_calculate_ratio_min.BL_calculate_ratio_min(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = BL_calculate_ratio_max.BL_calculate_ratio_max(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = BL_calculate_ratio_mean.BL_calculate_ratio_mean(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = BL_calculate_ratio_median.BL_calculate_ratio_median(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = BL_calculate_ratio_std.BL_calculate_ratio_std(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)

    print('Running local summary statistics.')
    stat = local_calculate_frac_basal.local_calculate_frac_basal(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = local_calculate_LBI_mean.local_calculate_LBI_mean(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)

    print('Running lineages through time summary statistics.')
    LTTs = get_LTT(tree)
    stat = LTT_calculate_max_lineages.LTT_calculate_max_lineages(tree, **LTTs, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = LTT_calculate_t_max_lineages.LTT_calculate_t_max_lineages(tree, **LTTs, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = LTT_calculate_mean_b_time.LTT_calculate_mean_b_time(tree, **LTTs, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = LTT_calculate_mean_s_time.LTT_calculate_mean_s_time(tree, **LTTs, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = LTT_calculate_slope_1.LTT_calculate_slope_1(tree, **LTTs, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = LTT_calculate_slope_2.LTT_calculate_slope_2(tree, **LTTs, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = LTT_calculate_slope_ratio.LTT_calculate_slope_ratio(tree, **LTTs, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    del LTTs

    print('Running network science time summary statistics.')
    stat = netSci_calculate_betweenness_max.netSci_calculate_betweenness_max(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = netSci_calculate_eigen_centrality_max.netSci_calculate_eigen_centrality_max(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    distance_mat = get_distance_mat(tree, topology_only=False)
    kwargs.update(get_eigenvalues_dist_lap(tree, topology_only=False, **distance_mat))
    stat = netSci_calculate_closeness_max.netSci_calculate_closeness_max(tree, **distance_mat, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = netSci_calculate_mean_path.netSci_calculate_mean_path(tree, **distance_mat, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = netSci_calculate_diameter.netSci_calculate_diameter(tree, **distance_mat, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    del distance_mat
    stat = netSci_calculate_eigen_centrality_max.netSci_calculate_eigen_centrality_max(tree, topology_only=True, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    distance_mat = get_distance_mat(tree, topology_only=True)
    kwargs.update(get_eigenvalues_dist_lap(tree, topology_only=True, **distance_mat))
    stat = netSci_calculate_closeness_max.netSci_calculate_closeness_max(tree, topology_only=True, **distance_mat, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = netSci_calculate_mean_path.netSci_calculate_mean_path(tree, topology_only=True, **distance_mat, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = netSci_calculate_diameter.netSci_calculate_diameter(tree, topology_only=True, **distance_mat, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    del distance_mat

    print('Running nearest neighbor distance summary statistics.')
    stat = mean_NN_distance.mean_NN_distance(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = mean_NN_distance.mean_NN_distance(tree, topology_only=True, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)

    print('Running small configurations summary statistics.')
    stat = smallConfig_calculate_cherries.smallConfig_calculate_cherries(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = smallConfig_calculate_double_cherries.smallConfig_calculate_double_cherries(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = smallConfig_calculate_pitchforks.smallConfig_calculate_pitchforks(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = smallConfig_calculate_fourprong.smallConfig_calculate_fourprong(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)

    print('Running spectral summary statistics.')
    adjacency_mat = get_adjacency_mat(tree, topology_only=False)
    eigen = get_eigenvalues_adj(tree, topology_only=False, **adjacency_mat)
    stat = spectral_calculate_min_adj_eigen.spectral_calculate_min_adj_eigen(tree, **eigen, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = spectral_calculate_max_adj_eigen.spectral_calculate_max_adj_eigen(tree, **eigen, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    eigen = get_eigenvalues_lap(tree, topology_only=False, **adjacency_mat)
    del adjacency_mat
    stat = spectral_calculate_min_lap_eigen.spectral_calculate_min_lap_eigen(tree, **eigen, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = spectral_calculate_max_lap_eigen.spectral_calculate_max_lap_eigen(tree, **eigen, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = spectral_calculate_max_distLap_eigen.spectral_calculate_max_distLap_eigen(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = spectral_calculate_eigen_gap.spectral_calculate_eigen_gap(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = spectral_calculate_skewness.spectral_calculate_skewness(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = spectral_calculate_kurtosis.spectral_calculate_kurtosis(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    adjacency_mat = get_adjacency_mat(tree, topology_only=True)
    eigen = get_eigenvalues_adj(tree, topology_only=True, **adjacency_mat)
    stat = spectral_calculate_min_adj_eigen.spectral_calculate_min_adj_eigen(tree, topology_only=True, **eigen, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = spectral_calculate_max_adj_eigen.spectral_calculate_max_adj_eigen(tree, topology_only=True, **eigen, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    eigen = get_eigenvalues_lap(tree, topology_only=True, **adjacency_mat)
    del adjacency_mat
    stat = spectral_calculate_min_lap_eigen.spectral_calculate_min_lap_eigen(tree, topology_only=True, **eigen, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = spectral_calculate_max_lap_eigen.spectral_calculate_max_lap_eigen(tree, topology_only=True, **eigen, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = spectral_calculate_max_distLap_eigen.spectral_calculate_max_distLap_eigen(tree, topology_only=True, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = spectral_calculate_eigen_gap.spectral_calculate_eigen_gap(tree, topology_only=True, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = spectral_calculate_skewness.spectral_calculate_skewness(tree, topology_only=True, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = spectral_calculate_kurtosis.spectral_calculate_kurtosis(tree, topology_only=True, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)

    print('Running topology summary statistics.')
    stat = top_calculate_B1.top_calculate_B1(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = top_calculate_B2.top_calculate_B2(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = top_calculate_colless.top_calculate_colless(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = top_calculate_frac_imbalance.top_calculate_frac_imbalance(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = top_calculate_frac_ladder.top_calculate_frac_ladder(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    #stat = top_calculate_FurnasR.top_calculate_FurnasR(tree, **kwargs)
    #treeData = pd.concat([treeData, stat], axis=1)
    stat = top_calculate_max_dW.top_calculate_max_dW(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = top_calculate_max_ladder.top_calculate_max_ladder(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = top_calculate_mean_imbalance_ratio.top_calculate_mean_imbalance_ratio(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = top_calculate_sackin.top_calculate_sackin(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = top_calculate_sackin_var.top_calculate_sackin_var(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = top_calculate_WD_ratio.top_calculate_WD_ratio(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)

    print('Running tree height summary statistics.')
    stat = tree_height_calculate_min.tree_height_calculate_min(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = tree_height_calculate_max.tree_height_calculate_max(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    stat = tree_height_calculate_mean.tree_height_calculate_mean(tree, **kwargs)
    treeData = pd.concat([treeData, stat], axis=1)
    
    return treeData