# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 17:02:38 2022

@author: 20192157
"""

import pandas as pd
import numpy as np

from Molecule import Molecule
from AssignmentPCA import readAllDescriptors, Plot, Covariance, CovarianceMatrix, PCA, PCA_plot, Loadings

# (a) Testing Molcule
df = pd.read_csv('QSAR_3_large_data.csv')
df_ppar_full = Molecule(df,'ppar').df_target
# (b)
random_molecule_descriptors = Molecule(df,'ppar').readMolDescriptors(2)

# (c) Testing reading all descriptors
df_ppar_only_descriptors = readAllDescriptors(df, 'ppar')
    
# (d) (e) Testing Plot
columns_of_interest = ['nHet','nS','nAB','SlogP'] #some randomly chosen descriptors
#plot = Plot(df_target,columns_of_interest)
#twoDplot = plot.twoDimPlot()

# (f) Testing Covariance
lst1 = df['nHet'].values.tolist()
lst2 = df['SlogP'].values.tolist()
covariance_two_lists = Covariance(lst1,lst2).calculateCovariance()
covariance_test_two_lists = np.cov(lst1,lst2) # If n-1 is done, the function works, but 
# the covariance matrix isn't 1 on the diagonal, however when -1 is removed, the
# covariance has the correct values...

# (g) Testing Covariance matrix with scaled values
covariance_matrix_class = CovarianceMatrix(df,'ppar')
df_ppar_only_descriptors_scaled = covariance_matrix_class.scaleVariables()
df_ppar_covariance_matrix = covariance_matrix_class.covMatrix()

# (h) Testing PCA
pca_class = PCA(df_ppar_covariance_matrix)
pca_eigval, pca_eigvec = pca_class.perform_PCA()

# (i) Testing plotting
pca_plot = PCA_plot(df)
pca_plot.PCA_plot_2D()
#pca_plot.PCA_plot_3D()

# (j) Testing loadings:
load = Loadings(df,'ppar')
X,u,s,vt = load.calculate_loadings()













