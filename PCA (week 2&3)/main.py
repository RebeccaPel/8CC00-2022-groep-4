# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 17:02:38 2022

@author: 20192157
"""

import pandas as pd
import numpy as np

from Molecule import Molecule
from AssignmentPCA import readAllDescriptors, Average, PCA, PCA_plot 

# (a) Testing Molecule -------------------------------------------------
df = pd.read_csv('QSAR_3_large_data.csv')
df_target_full = Molecule(df,'ppar').df_target
# (b) ------------------------------------------------------------------
random_molecule_descriptors = Molecule(df,'ppar').read_mol_descriptors(2)

# (c) Testing reading all descriptors ----------------------------------
df_target_only_descriptors = readAllDescriptors(df, 'ppar')
    
# (d) (e) Testing Plot -------------------------------------------------
columns_of_interest = ['nHet','nS','nAB','SlogP'] #some randomly chosen descriptors
plot = Average(df_target_full,columns_of_interest)
cum_mov_avg_dict = plot.cumulative_moving_average()
#plot.plot_cumulative_moving_average_alternative()

# (f) Covariance------------------------------------------------
lst1 = df['nHet'].values.tolist()
lst2 = df['SlogP'].values.tolist()
pca = PCA(df,'ppar')
covariance_two_lists = pca.calculate_covariance(lst1,lst2,bias=True)
covariance_test_two_lists = np.cov(lst1,lst2) # Give the correct output

# (g) Covariance matrix with scaled values ---------------------
df_target_only_descriptors_scaled = pca.df_target_only_descriptors_scaled
df_target_covariance_matrix = pca.calculate_covariance_matrix()

# (h) PCA ------------------------------------------------------
pca_eigval, pca_eigvec, pca_var_exp, pca_cum_var_exp = pca.perform_PCA()

# (i) Plotting -------------------------------------------------
pca_plot = PCA_plot(df)
pca_plot.PCA_plot_2D()
#pca_plot.PCA_plot_3D()

# (j) Loading plots-----------------------------------------------
# Test for all targets
# variable_names_list = list(Molecule(df,"ppar").descriptors)
# n_vars = 5
# n_PCs = 3
# for PC in range(n_PCs):
#     pca_plot.loading_plots_all_targets(
#         variable_names_list = variable_names_list,
#         eig_vals_real = pca_eigval, 
#         eig_vecs_real = pca_eigvec,
#         PC=PC+1,
#         n_vars = n_vars
#         )
        
## Per target
# n_vars = 'All'
# n_PCs = 3
# for target in ['ppar', 'thrombin', 'cox2']:
#     for PC in range(n_PCs):
#         PCA_plot(df).loading_plots_per_target(Molecule(df, 'ppar').descriptors[2:], 
#                       # eig_vals_real = pca_eigval, 
#                       # eig_vecs_real = pca_eigvec,
#                       target = target,
#                       PC = PC+1, 
#                       n_vars = n_vars)
    
## Var plot & scree plot ----------------------------------------
# PCA_plot(df).var_plot(pca_var_exp,
#                       pca_cum_var_exp,
#                       pca_eigval
#                       )





