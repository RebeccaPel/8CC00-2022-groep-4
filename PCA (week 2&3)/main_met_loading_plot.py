# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 17:02:38 2022

@author: 20192157
"""

import pandas as pd
import numpy as np

from Molecule_aanpassing_Max import Molecule
from AssignmentPCA_met_loading_plot import readAllDescriptors, Average, PCA, PCA_plot

# (a) Testing Molcule
df = pd.read_csv('QSAR_3_large_data.csv')
df_target_full = Molecule(df,'ppar').df_target
# (b)
random_molecule_descriptors = Molecule(df,'ppar').read_mol_descriptors(2)

# (c) Testing reading all descriptors
df_target_only_descriptors = readAllDescriptors(df, 'ppar')
    
# (d) (e) Testing Plot
columns_of_interest = ['nHet','nS','nAB','SlogP'] #some randomly chosen descriptors
plot = Average(df_target_full,columns_of_interest)
# Oude manier:
#plot.plot_cumulative_moving_average()
# Nieuwe manier:
cum_mov_avg_dict = plot.cumulative_moving_average_alternative()
#plot.plot_cumulative_moving_average_alternative()

# (f) Testing Covariance
lst1 = df['nHet'].values.tolist()
lst2 = df['SlogP'].values.tolist()
pca = PCA(df,'ppar')
covariance_two_lists = pca.calculate_covariance(lst1,lst2,bias=True)
covariance_test_two_lists = np.cov(lst1,lst2) # Give the correct output

# (g) Testing Covariance matrix with scaled values
df_target_only_descriptors_scaled = pca.df_target_only_descriptors_scaled
df_target_covariance_matrix = pca.calculate_covariance_matrix()

# (h) Testing PCA
pca_eigval, pca_eigvec = pca.perform_PCA()

# (i) Testing plotting
pca_plot = PCA_plot(df)
#pca_plot.PCA_plot_2D()
#pca_plot.PCA_plot_3D()

# (j) Testing loadings:
n_vars = 5
n_PCs = 3
for PC in range(n_PCs):
    pca_plot.loading_plots(Molecule(df, 'ppar').descriptors, 
                  0,
                  PC = PC+1, 
                  n_vars = n_vars)













