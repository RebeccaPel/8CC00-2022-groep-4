# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 17:05:29 2022

@author: 20192157
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import svd

from Molecule import Molecule

def readAllDescriptors(df,target):
    '''
    This function reads in all molecule descriptors of the entire dataset using
    functions from class Molecule.

    Parameters
    ----------
    df : DataFrame
        a dataframe with 34 columns, all descriptors of the molecule which
        is defined in the first column with the smile
    target : str
        must be a string being one of the three options in the target
        column of the dataframe (ppar, thrombin,cox2)

    Returns
    -------
    all_descriptors : DataFrame
        All moleucle descriptors of all molecules in the dataset.
    '''
    # First use clas Molecule to be able to use functions in this class
    molecule_class = Molecule(df,target)
    df_target_full = molecule_class.df_target
    indexes = list(df_target_full.index)
    
    # Start with reading the first molecule so all other molecules can be
    # appended to this one
    df_target_only_descriptors = molecule_class.read_mol_descriptors(indexes[0])
    
    for i in range(1,len(indexes)): 
        # For each index in the lengt of the dataframe extract the descriptors
        descriptors_one_molecule = molecule_class.read_mol_descriptors(indexes[i])
        # Each time update all descriptors by adding the new descriptors to the dataframe
        df_target_only_descriptors = df_target_only_descriptors.append(descriptors_one_molecule)
    
    return df_target_only_descriptors

class Average:
    
    def __init__(self,columns_of_interest_list):
        '''
        Parameters
        ----------
        param_list : list
            List of values of which the moving average will be calculated
        '''
        self.columns_of_interest_list = columns_of_interest_list
        self.len = len(columns_of_interest_list)
        
    def cumulativeMovingAverage(self):
        '''
        This function calculates a moving average of a list.

        Returns
        -------
        C : list
            The moving average of the input list.
        '''
        # Firstly, an empty list is created so the values can be replaced
        cumulative_moving_average = [0]*self.len
        
        for i in range(1,self.len):
            # The formula for the cumulative moving average is applied:
            numerator = self.columns_of_interest_list[i-1] - cumulative_moving_average[i-1]
            denominator = (i-1)+1
            cumulative_moving_average[i] = cumulative_moving_average[i-1] + numerator/denominator
            
        return cumulative_moving_average

class Plot:
    
    def __init__(self,df,columns_of_interest):
        '''
        Parameters
        ----------
        df : DataFrame
            A dataframe with 34 columns, all descriptors of the molecule which
            is defined in the first column with the smile
            The dataframe can be of all targets or of a single target.
        columns_of_interest : List of strings
            A list of the columns of which the moving averages of the values will 
            be plotted, can be of any length between 1 and the amount of columns 
            that exist.
        '''
        self.df = df
        self.columns_of_interest = columns_of_interest
        self.descriptor_len = len(columns_of_interest)
        
    def twoDimPlot(self):
        '''
        This function creates a two-dimensional plot of the moving averages. On
        the x-axis the index of the moving average and the value on the y-axis.

        Returns
        -------
        No return, but a plot will be shown of the calculated moving average(s).

        '''
        # Looped for each moving average that will be plotted:
        for i in range(self.descriptor_len):
            # Extract the name of the column
            descriptor = self.columns_of_interest[i]
            # Extract values of the column and convert them to a list
            values = self.df[descriptor].values.tolist()
            # The moving average is now calculated with the function
            # cumulativeMovngAverage from class Average
            CMA = Average(values).cumulativeMovingAverage()            
            plt.plot(CMA)
            
            # Add axes and labels
            
        plt.show()
        
class Covariance:
    
    def __init__(self,list1,list2):
        '''
        Parameters
        ----------
        list1 : list
            List of numbers
        list2 : list
            List of numbers
        
        Raises ValueError if list1 is not the same length as list2
        '''
        
        self.list1 = list1
        self.list2 = list2
        # If lists are not the same length, raise ValueError
        if len(list1) != len(list2):
            raise ValueError('Lists are not of the same length')
        else:
            self.len = len(list1) # or list2, makes no difference
        
        self.avg1 = np.average(list1) #sum(list1)/len(list1)
        self.avg2 = np.average(list2) #sum(list2)/len(list2)
        
    def calculateCovariance(self):
        '''
        The covariance of two lists is calculated.

        Returns
        -------
        covariance : float
            The covariance of two lists.
        '''
        # Start with the sum at zero
        sum_of = 0
        for i in range(self.len):
            # Substract each value in the list with its average
            a = self.list1[i]-self.avg1
            b = self.list2[i]-self.avg2
            # Now multiply the values and add them to the sum
            sum_of += a*b
        # Apply last step to calculate covariance
        covariance = sum_of * (1/(self.len)) # -1 or not 
        return covariance
    
class CovarianceMatrix:
    
    def __init__(self,df_full,target):
        '''
        Parameters
        ----------
        df : DataFrame
            a dataframe with 34 columns, all descriptors of the molecule which
            is defined in the first column with the smile
        target : str
            One of three options: ppar, thrombin or cox2
        '''
        
        self.df_full = df_full
        
        self.df_target_full = Molecule(df_full,target).df_target
        
        self.df_target_only_descriptors = readAllDescriptors(df_full,target)
        self.descriptors = list(self.df_target_only_descriptors.columns)
        
        self.df_target_only_descriptors_scaled = self.scaleVariables()
        
    def scaleVariables(self): 
        '''
        This functions scales all values in the DataFrame with the standardization
        method.

        Returns
        -------
        df_descriptors_scaled : DataFrame
            The standardized DataFrame.
        '''
        # Create new DataFrame:
        df_target_only_descriptors_scaled = self.df_target_only_descriptors.copy()
        columns = list(self.df_target_only_descriptors.columns)
        
        for col in columns:
            # Extract all values per column
            column = self.df_target_only_descriptors[col]
            
            # Now calculate the standard deviation and average using numpy functions
            std = np.std(column.values.tolist())
            avg = np.average(column.values.tolist())
            
            for ind in self.df_target_full.index:
                # Now loop over each index
                if std == 0:
                    # If the standard deviation is zero, the scaling can't be done
                    # because this would mean dividing by zero. So 'skipping'
                    # these values and setting them to zero.                    
                    df_target_only_descriptors_scaled.at[ind,col] = 0
                else:
                    # Extracting value per index
                    value = self.df_target_only_descriptors.at[ind,col]
                    # Performing the standardization:
                    scaled_value = (value - avg)/std
                    # Replace each value in the new DataFrame
                    df_target_only_descriptors_scaled.at[ind,col] = scaled_value    
        return df_target_only_descriptors_scaled
    
    def covMatrix(self):
        '''
        This function creates the covariance matrix of the scaled dataframe
        created in scaleVariables.

        Returns
        -------
        cov_mat : DataFrame
            The covariation matrix of the DataFrame.
        '''
        # Create empty DataFrame with the molecule descriptors at the column
        # names and at the indexes. (symmetrical)
        zeroes = np.zeros(shape = (len(self.descriptors), len(self.descriptors)))
        covariance_matrix = pd.DataFrame(zeroes, columns = self.descriptors)
        covariance_matrix = covariance_matrix.set_axis(self.descriptors)
        
        for j in self.descriptors:
            
            for k in self.descriptors:
                # Now for each combination of descriptors extract the two descriptor list                
                columnj = self.df_target_only_descriptors_scaled[j].to_list()
                columnk = self.df_target_only_descriptors_scaled[k].to_list()
                # Calculate the covariance of the two descriptors using the function
                # calculateCovariance from class Covariance
                covariance_value = Covariance(columnj,columnk).calculateCovariance()
                # Add each value in the (previously) empty DataFrame.
                covariance_matrix.at[j,k] = covariance_value        
        return covariance_matrix

class PCA:
    
    def __init__(self,covariance_matrix):
        '''
        Parameters
        ----------
        covariance_matrix : DataFrame
            A covariance matrix with the molecule descriptors on the indexes and
            columns.
        '''
        
        self.covariance_matrix = covariance_matrix
        self.eigen_values, self.eigen_vectors = self.perform_PCA()
        
    def perform_PCA(self):
        '''
        Principal Component Analysis is performed on the covariance matrix.

        Returns
        -------
        eig_vals : Array
            Eigenvalues of the covariance matrix
        eig_vecs : Array
            Eigenvectors of the covariance matrix
        '''
        # Use the numpy function to extract eigenvalues and eigenvectors
        eigen_values, eigen_vectors = np.linalg.eig(self.covariance_matrix)
        # If there are complex values, convert them to real values, because plotting
        # with complex values will be difficult later on.
        eigen_values = np.real(eigen_values)
        eigen_vectors = np.real(eigen_vectors)
        return eigen_values, eigen_vectors
    
class PCA_plot():
    
    def __init__(self,df,targets=['ppar','thrombin','cox2']):
        '''
        Parameters
        ----------
        df : DataFrame
            a dataframe with 34 columns, all descriptors of the molecule which
            is defined in the first column with the smile
        targets : list, optional
            DESCRIPTION. The default is ['ppar','thrombin','cox2'].
        '''
        self.targets = targets
        
        self.mat_cov1 = CovarianceMatrix(df,targets[0]).covMatrix()
        self.mat_cov2 = CovarianceMatrix(df,targets[1]).covMatrix()
        self.mat_cov3 = CovarianceMatrix(df,targets[2]).covMatrix()
        
        self.target1_eigval, self.target1_eigvec = PCA(self.mat_cov1).perform_PCA()
        self.target2_eigval, self.target2_eigvec = PCA(self.mat_cov2).perform_PCA()   
        self.target3_eigval, self.target3_eigvec = PCA(self.mat_cov3).perform_PCA()
        self.color1, self.color2, self.color3 = 'r','b','g' 
        
    def PCA_plot_2D(self):
        '''
        A 2D plot is created with the two most importend principle components, 
        each on an axis. The values of the three targets are all plotted in a 
        different color.
        '''
        
       # Plot three times, for each target: 
        plt.scatter((self.target1_eigval[0]*self.target1_eigvec[:,0]), 
                  (self.target1_eigval[1]*self.target1_eigvec[:,1]),
                  label = str(self.targets[0]))
        plt.scatter((self.target2_eigval[0]*self.target2_eigvec[:,0]), 
                  (self.target2_eigval[1]*self.target2_eigvec[:,1]),
                  label = str(self.targets[1]))
        plt.scatter((self.target3_eigval[0]*self.target3_eigvec[:,0]), 
                  (self.target3_eigval[1]*self.target3_eigvec[:,1]),
                  label = str(self.targets[2]))
       
        plt.legend(loc='upper left')        
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        
        plt.show()
        
    def PCA_plot_3D(self):
        '''
        A 3D plot is created with the three most important components, each on
        an axis. The values of the three targets are all plotted in a different color.
        '''
        
        fig = plt.figure(1, figsize=(4, 3))
        ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
        
        # Plot three times for each target:
        ax.scatter((self.target1_eigval[0]*self.target1_eigvec[:,0]), 
                   (self.target1_eigval[1]*self.target1_eigvec[:,1]), 
                   (self.target1_eigval[2]*self.target1_eigvec[:,2]), 
                   c=self.color1, 
                   cmap=plt.cm.nipy_spectral, 
                   edgecolor="k",
                   label = str(self.targets[0]))
        ax.scatter((self.target2_eigval[0]*self.target2_eigvec[:,0]), 
                   (self.target2_eigval[1]*self.target2_eigvec[:,1]), 
                   (self.target2_eigval[2]*self.target2_eigvec[:,2]), 
                   c=self.color2, 
                   cmap=plt.cm.nipy_spectral, 
                   edgecolor="k",
                   label = str(self.targets[1]))
        ax.scatter((self.target3_eigval[0]*self.target3_eigvec[:,0]), 
                   (self.target3_eigval[1]*self.target3_eigvec[:,1]), 
                   (self.target3_eigval[2]*self.target3_eigvec[:,2]), 
                   c=self.color3, 
                   cmap=plt.cm.nipy_spectral, 
                   edgecolor="k",
                   label = str(self.targets[2])) 
        
        ax.legend(loc='upper left')        
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        
        plt.show()

def loading_plots(variable_names_list, eig_vals_real, eig_vecs_real, PC=1, 
                  n_vars='All'):
    """
    Generates a loading plot of a predefined number of variables with
    the largest loading for a principle component of choice. Loadings 
    represent the correlation between the original variables and the
    principal components.
  
    Parameters
    ----------
    variable_names_list : list[str]
        A list containing all variable names (str) in order of read-out.
    
    eig_vals_real : array
        The real eigenvalues, each repeated according to its
        multiplicity. The eigenvalues are real, i.e. 0 imaginary part.
        The eigenvalues are not necessarily ordered.
    
    eig_vecs_real : array
        The normalized (unit 'length') eigenvectors, such that the
        column eig_vecs_real[:,i] is the eigenvector corresponding to
        the eigenvalue eig_vals_real[i]. The eigenvectors are real, i.e.
        0 imaginary part.
        
    PC : int, optional
        Principle component number of which the loading plot is to be
        made. Default is 1.
    
    n_vars : int or 'All', optional
        The amount of variables with the largest loading that will be
        shown in the loading plot. Default is 'All'.
    
    Returns
    -------
    Loading plot of a predefined number of variables with the largest
    loading for a principal component of choice.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Make a list of (eigenvalue, eigenvector) tuples and sort the 
    # (eigenvalue, eigenvector) tuples from high to low
    
    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals_real[i]), eig_vecs_real[:,i]) 
                  for i in range(len(eig_vals_real))]
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda tup: tup[0], reverse=True)
    
    # eig_val_vec_list = [(np.abs(eig_vals_real[i]), eig_vecs_real[:,i]) 
    #                     for i in range(len(eig_vals_real))
    #                     ].sort(key=lambda tup: tup[0], reverse=True)
    # print(f'{type(eig_val_vec_list) = }\n')
    # # Calculate the loading scores of all variables on the PC of interest
    # print(f'{eig_val_vec_list[PC-1][1].T = }\n')
    # print(f'{eig_val_vec_list[PC-1][0] = }\n')
    # print(f'{PC-1}\n')
    eig_val_vec_list = eig_pairs
    loadings = eig_val_vec_list[PC-1][1].T / np.sqrt(eig_val_vec_list[PC-1][0])
    # Convert this array to a list
    loadings = loadings.tolist()
    
    # Make a list of tuples containing (variable_name, loading)
    var_names_loadings_tuples_list = [(variable_names_list[i], loadings[i]) 
                                      for i in range(len(loadings))]
    print(f'{var_names_loadings_tuples_list = }')
    # Order this list of tuples by their loading, in descending order
    var_names_loadings_tuples_list.sort(key=lambda num: abs(num[1]),
                                        reverse=True)
    
    # Only select the <n_vars> highest loadings
    if n_vars == 'All':
        selected = var_names_loadings_tuples_list[0:] 
    else:
        selected = var_names_loadings_tuples_list[0:n_vars+1]
    
    # Make two separate lists again: variable_names_list containing the
    # names of the variables (ordered) (X axis) and loadings containing
    # the loadings of these variables (Y axis). Their indices match.
    variable_names_list, loadings = zip(*selected)
    
    # Make a bar graph: configuration
    fig = plt.figure(figsize=(10,5), dpi=150)
    ax = fig.add_axes([0,0,1,1])
    plt.xticks(rotation=90)  # Rotate the labels so that they are readable
    ax.set_title(f"Loading plot PC{PC}")
    ax.set_ylabel('Loading')
    
    # Color all negative loadings in the plot red; all positive loadings
    # green
    red_green = []
    for item in loadings:
        item = item.real
        if item >= 0:
            red_green.append('green')
        else:
            red_green.append('red')
    
    # Create the bar graph
    ax.bar(variable_names_list, loadings, color=red_green)
    # Save and show the image
    plt.savefig(f'Loading plot PC{PC}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    # ==================================================================
    # VERSIE 2
    # ==================================================================
    
    # # Discard imaginary part of eigenvalues 
    # eig_vals = np.real(eig_vals_real)
    # # Make a list of (eigenvalue, eigenvector) tuples
    # eig_pairs = [(np.abs(eig_vals[i]), eig_vecs_real[:,i]) 
    #              for i in range(len(eig_vals))]
    
    # # Calculate the loading of all genes on the PC of interest
    # loading = eig_pairs[PC-1][1].T * np.sqrt(eig_pairs[PC-1][0])
    # # Convert this array to a list
    # loading = loading.tolist()
    
    # # Make a list of tuples containing (geneName, loading)
    # listoftup = []
    # for i in range(len(loading)):
    #     listoftup.append((variable_names_list[i],loading[i]))
    
    # # Order this list of tuples by their loading, in descending order
    # listoftup.sort(key=lambda num: abs(num[1]), reverse=True)
    
    # # Only select the highest loadings (see: var)
    # selected = listoftup[0:n_vars+1]
    
    # # Make two separate lists again: genes containing the names of the genes (ordered) 
    # # and loading containing the loadings of these genes. Their indices match.
    # genes,loading = zip(*selected)
    
    # # Make a bar graph: configuration
    # fig = plt.figure(figsize=(10,5), dpi=150)
    # ax = fig.add_axes([0,0,1,1])
    # plt.xticks(rotation=90) # Rotate the labels so that they are readable
    # ax.set_title("Loading plot PC{PCin}".format(PCin = PC))
    # ax.set_ylabel('Loading')
    
    # # Forloop to color all negative loadings in the plot red; all positive loadings green
    # Red_Green = []
    # for item in loading:
    #     item = item.real
    #     if item >= 0:
    #         Red_Green.append('green')
    #     else:
    #         Red_Green.append('red')
    
    # # Create the bar graph
    # ax.bar(genes,loading, color=Red_Green)
    # # Save and show the image
    # plt.savefig('loading.png', dpi=300, bbox_inches='tight')
    # plt.show()


