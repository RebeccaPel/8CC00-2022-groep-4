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
mol = Molecule(df,'ppar') 
df_target = mol.df_target
# (b)
one_descriptors = mol.readMolDescriptors(2)

# (c) Testing reading all descriptors
all_descriptors = readAllDescriptors(df, 'ppar')
    
# (d) (e) Testing Plot
columns_of_interest = ['nHet','nS','nAB','SlogP'] #some randomly chosen descriptors
plot = Plot(df_target,columns_of_interest)
#twoDplot = plot.twoDimPlot()

# (f) Testing Covariance
lst1 = df_target['nHet'].values.tolist()
lst2 = df_target['SlogP'].values.tolist()
cov = Covariance(lst1,lst2)
covariance = cov.calculateCovariance()
covariance_test = np.cov(lst1,lst2) # If n-1 is done, the function works, but 
# the covariance matrix isn't 1 on the diagonal, however when -1 is removed, the
# covariance has the correct values...

# (g) Testing Covariance matrix with scaled values
mat = CovarianceMatrix(df,'ppar')
mat_scaled = mat.scaleVariables()
mat_cov = mat.covMatrix()

# (h) Testing PCA
pca = PCA(mat_cov)
pca_eigval, pca_eigvec = pca.perform_PCA()

# (i) Testing plotting
pca_plot = PCA_plot(df)
pca_plot.PCA_plot_2D()
#pca_plot.PCA_plot_3D()

# (j) Testing loadings:
load = Loadings(df,'ppar')
X,u,s,vt = load.calculate_loadings()

# I haven't been able to finish the loadings, because I can't figure out how
# to multiply V*S, because the shapes (with ppar) are V=(32,32) and S=(26,),
# which can never be multiplied.
# Also because of this, I have no results to reflect on...













