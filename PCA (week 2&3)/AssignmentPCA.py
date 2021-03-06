# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 17:05:29 2022

@author: 20192157
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

from Molecule import Molecule

def readAllDescriptors(df,target=''):
    """
    This function reads in all molecule descriptors of the entire dataset using
    functions from class Molecule.
    The result is that only the molecular descriptors are kept i.e. the columns
    SMILES and Target have been removed.

    Parameters
    ----------
    df : DataFrame
        A dataframe molecular descriptors as variables i.e. columns
        and SMILES (molecule names) as objects.
    target : str
        Must be a string being one of the options in the target
        column of the DataFrame (e.g. 'ppar', 'thrombin', 'cox2').

    Returns
    -------
    all_descriptors : pandas DataFrame
        All molecular descriptors of all molecules in the dataset.
    """
    # First use clas Molecule to be able to use functions in this class
    molecule_class = Molecule(df, target)
    df_target_full = molecule_class.df_target
    indexes = list(df_target_full.index)
    
    # Start with reading the first molecule so all other molecules can be
    # appended to this one
    df_target_only_descriptors = \
        molecule_class.read_mol_descriptors(indexes[0])
    
    for i in range(1, len(indexes)): 
        # For each index in the lengt of the dataframe extract the 
        # descriptors
        descriptors_one_molecule = \
            molecule_class.read_mol_descriptors(indexes[i])
        # Each time update all descriptors by adding the new descriptors
        # to the dataframe
        df_target_only_descriptors = df_target_only_descriptors.append(
            descriptors_one_molecule)
    
    return df_target_only_descriptors


class Average:
    """Calculate and plot cumulative moving average.
    
    Parameters
    ----------
    df: pandas DataFrame
        A dataframe molecular descriptors as variables i.e. columns
        and SMILES (molecule names) as objects.
        
    columns_of_interest_list: list of strings
        These are the columns of which the values are used to calculate (and 
        plot) the cumulative moving average.
        The strings must be a selection of strings that correspond to the column
        names in the parameter df.
        
    Attributes
    ----------
    df: pandas DataFrame
        A dataframe molecular descriptors as variables i.e. columns
        and SMILES (molecule names) as objects.
        
    len_values: int
        Number of rows in df.
        
    columns_of_interest_list: list of strings
        These are the columns of which the values are used to calculate (and 
        plot) the cumulative moving average.
        The strings must be a selection of strings that correspond to the column
        names in the parameter df.
        
    len_columns: int
        Number of columns in df
     
    Methods
    -------
    cumulative_moving_average():
        Calculates the cumulative moving average of the columns of interest
        of df.
        
    plot_cumulative_moving_average():
        Uses the output from cumulative_moving_average() and plots the values.    
    """
    
    def __init__(self,df,columns_of_interest_list = []):
        """
        Parameters
        ----------
        param_list : list
            List of values of which the moving average will be calculated
        """
        self.df = df
        self.len_values = len(df)
        self.columns_of_interest_list = columns_of_interest_list
        self.len_columns = len(columns_of_interest_list)  

        self.cumulative_moving_average_dictionary = self.cumulative_moving_average()
        
    def cumulative_moving_average(self): 
        """This function calculates the cumulative moving average by using the 
        following formula:
            If x(i:0<=i<N) is a list of N data items, then the list of cumulative 
            moving averages C[n], 0 <= n <= N is defined by:
            C(0) = 0
            for 0<n<N:
                C(n+1)=C(n)+(x(n)-C(n))/(n+1)

        Returns
        -------
        cumulative_moving_average_dictionary : list of integers
            The cumulative moving average of the columns of interest in the
            dataframe.
        """
        # Create empty dictionary         
        cumulative_moving_average_dictionary = {}
        for descriptor_name in self.columns_of_interest_list:
            # Extract values per descriptor    
            values = self.df[descriptor_name].values.tolist()
            # Create empty list for the CMA values
            cumulative_moving_average = [0]*len(values)
                
            for i in range(1,len(values)):
                # The formula for the cumulative moving average is applied:
                numerator = values[i-1] - cumulative_moving_average[i-1]
                denominator = (i-1)+1
                cumulative_moving_average[i] = cumulative_moving_average[i-1] + numerator/denominator
            # Now save the list of the CMA as values to the key of descriptor name in the dictionary        
            cumulative_moving_average_dictionary[descriptor_name] = cumulative_moving_average
                
        return cumulative_moving_average_dictionary
    
    def plot_cumulative_moving_average(self):
        """This function plots the cumulative moving average(s) as calculated
        by cumulative_moving_average().
        """
        
        for key in self.columns_of_interest_list:
            cma = self.cumulative_moving_average_dictionary.get(key)
            plt.plot(cma)
            
        plt.legend(self.columns_of_interest_list)
        plt.xlabel('Index')
        plt.ylabel('Cumulative Moving Average')
        plt.show()
                    
        
class PCA:
    """Perform the calculation steps of Principal Component Analysis.
    
    Parameters
    ----------
    df: pandas DataFrame
        A dataframe molecular descriptors as variables i.e. columns
        and SMILES (molecule names) as objects.
        
    target: str
        Must be a string being one of the options in the target
        column of the DataFrame (e.g. 'ppar', 'thrombin', 'cox2').
        
    Attributes
    ----------
    df_full: pandas DataFrame
        A dataframe molecular descriptors as variables i.e. columns
        and SMILES (molecule names) as objects.
        
    df_target_full: pandas DataFrame
        This is a copy of parameter df_full, but only the rows
        corresponding to the parameter target are kept, the rest of the
        rows have been discarded.
    
    descriptors: list of strings
        A list of the column names in df_full (and df_target_full)
        
    df_target_only_descriptors_scaled: pandas DataFrame
        Output from the function scale_variables()
        
    covariance_matrix: pandas DataFrame
        Output from the function calculate_covariance_matrix()
        
    eigen_values, eigen_vectors: Array of float64
        Output from the function perform_PCA()
        
    Methods
    -------
    calculate_covariance(list1,list2,bias):
        Calculates the covariance of two lists, with or without an added
        bias.
        
    scale_variables():
        Scales all the variables of df_target_full with standardization.
        
    calculate_covariance_matrix():
        This function creates the covariance matrix of the scaled
        DataFrame created in scaleVariables.
        
    perform_PCA(self):
        Principal Component Analysis is performed on the covariance
        matrix.
    """
    def __init__(self, df, target):
        self.df_full = df
        
        self.df_target_full = Molecule(df,target).df_target
        
        self.df_target_only_descriptors = readAllDescriptors(df, target)
        self.descriptors = list(self.df_target_only_descriptors.columns)
        
        # Scaled variables.
        self.df_target_only_descriptors_scaled = self.scale_variables()
        
        # Covariance matrix.
        self.covariance_matrix = self.calculate_covariance_matrix()
        self.eigen_values, self.eigen_vectors, self.var_exp, self.cum_var_exp \
            = self.perform_PCA()

        
    def calculate_covariance(self, list1=None, list2=None, bias=False):
        """The covariance of two lists is calculated.
        
        Parameters
        ----------
        list1, list2 : list[float | int]
            Lists of numbers.
        
        Raises ValueError if list1 is not the same length as list2.
        
        Returns
        -------
        covariance : float
            The covariance of two lists.
        """
        # If lists are not the same length, raise ValueError.
        if len(list1) != len(list2):
            raise ValueError('Lists are not of the same length')
        else:
            len_list = len(list1)  # or list2, makes no difference
        
        avg1 = np.average(list1)  # sum(list1)/len(list1)
        avg2 = np.average(list2)  # sum(list2)/len(list2)
        
        # Start with the sum at zero.
        sum_of = 0
        for i in range(len_list):
            # Substract each value in the list with its average.
            a = list1[i]-avg1
            b = list2[i]-avg2
            # Now multiply the values and add them to the sum.
            sum_of += a*b
        # Apply last step to calculate covariance.
        if bias == True:
            covariance = sum_of * (1/(len_list-1))
        if bias == False:
            covariance = sum_of * (1/len_list)
        
        return covariance
    
    
    def scale_variables(self): 
        """This functions scales all values in the DataFrame with the 
        standardization method.

        Returns
        -------
        df_descriptors_scaled : DataFrame
            The standardized DataFrame.
        """
        # Create new DataFrame:
        df_target_only_descriptors_scaled = \
            self.df_target_only_descriptors.copy()
        columns = list(self.df_target_only_descriptors.columns)
        
        for col in columns:
            # Extract all values per column
            column = self.df_target_only_descriptors[col]
            
            # Now calculate the standard deviation and average using
            # NumPy methods.
            std = np.std(column.values.tolist())
            avg = np.average(column.values.tolist())
            
            for ind in self.df_target_full.index:
                # Now loop over each index.
                if std == 0:
                    # If the standard deviation is zero, the scaling
                    # can't be done because this would mean dividing by
                    # zero. So 'skipping' these values and setting them
                    # to zero.                    
                    df_target_only_descriptors_scaled.at[ind,col] = 0
                else:
                    # Extracting value per index.
                    value = self.df_target_only_descriptors.at[ind,col]
                    # Performing the standardization:
                    scaled_value = (value - avg)/std
                    # Replace each value in the new DataFrame.
                    df_target_only_descriptors_scaled.at[ind,col] = \
                        scaled_value    
        return df_target_only_descriptors_scaled
    
    
    def calculate_covariance_matrix(self):
        """This function creates the covariance matrix of the scaled
        DataFrame created in scale_variables.

        Returns
        -------
        cov_mat : DataFrame
            The covariation matrix of the DataFrame.
        """
        # Create empty DataFrame with the molecule descriptors at the column
        # names and at the indexes. (symmetrical)
        zeroes = np.zeros(shape = (len(self.descriptors), 
                                   len(self.descriptors)))
        covariance_matrix = pd.DataFrame(zeroes, columns=self.descriptors)
        covariance_matrix = covariance_matrix.set_axis(self.descriptors)
        
        for j in self.descriptors:
            
            for k in self.descriptors:
                # Now for each combination of descriptors extract the two 
                # descriptor list                
                columnj = self.df_target_only_descriptors_scaled[j].to_list()
                columnk = self.df_target_only_descriptors_scaled[k].to_list()
                # Calculate the covariance of the two descriptors using
                # the function
                # calculate_covariance from class Covariance
                covariance_value = self.calculate_covariance(
                    columnj, columnk, bias=False)
                # Add each value in the (previously) empty DataFrame.
                covariance_matrix.at[j,k] = covariance_value 
        
        return covariance_matrix
    
    
    def perform_PCA(self):
        """Principal Component Analysis is performed on the covariance
        matrix.

        Returns
        -------
        eig_vals : Array
            Eigenvalues of the covariance matrix
            
        eig_vecs : Array
            Eigenvectors of the covariance matrix
            
        cum_var_exp : list
            List of the cumulative explained variance, with index 0
            corresponding to the CMA of PC 1, index 1 of PC2, etc.
            
        """
        # Use the NumPy method to extract eigenvalues and eigenvectors
        eigen_values, eigen_vectors = np.linalg.eig(self.covariance_matrix)
        # If there are complex values, convert them to real values, 
        # because plotting with complex values will be difficult later
        # on.
        eigen_values = np.real(eigen_values)
        eigen_vectors = np.real(eigen_vectors)
        
        
        # Calculation of the explained or overall variance
        sum_eig_vals = sum(eigen_values)
        var_exp = [(i / sum_eig_vals)*100
                   for i in sorted(eigen_values, reverse=True)]
        
        # Cumulative explained variance
        cum_var_exp = np.cumsum(var_exp) 
        
        return eigen_values, eigen_vectors, var_exp, cum_var_exp


class PCA_plot():
    """Plot methods for the results of the class PCA.
    
    Parameters
    ----------
    df : DataFrame
        A dataframe with the data.
    targets : list, optional
        The default is ['ppar','thrombin','cox2'].
            
    Attributes
    ----------
    targets: list, optional
        The default is ['ppar','thrombin','cox2'].
        
    mat_covX: pandas DataFrame
        The covariance matrix as given by the function
        calculate_covariance_matrix() of a target.
        
    targetX_eigenval, targetX_eigvec: Array of float64
        The eigenvectors and eigenvalues belonging to the covariance
        matrix of a target.
        
    colorX: str
        A string with what color a target will be plotted with.
    """
    def __init__(self, df, targets=['ppar','thrombin','cox2']):
        self.targets = targets
        
        pca_target_1, pca_target_2, pca_target_3 = \
            PCA(df,targets[0]), PCA(df,targets[1]), PCA(df,targets[2])
        
        self.mat_cov1 = pca_target_1.calculate_covariance_matrix()
        self.mat_cov2 = pca_target_2.calculate_covariance_matrix()
        self.mat_cov3 = pca_target_3.calculate_covariance_matrix()
        
        self.target1_eigval, self.target1_eigvec, self.target1_var_exp, \
            self.target1_cum_var_exp = pca_target_1.perform_PCA()
        self.target2_eigval, self.target2_eigvec, self.target2_var_exp, \
            self.target2_cum_var_exp = pca_target_2.perform_PCA()   
        self.target3_eigval, self.target3_eigvec, self.target3_var_exp, \
            self.target3_cum_var_exp = pca_target_3.perform_PCA()
        self.color1, self.color2, self.color3 = 'r','b','g'       
        
        
        pca = PCA(df, targets)
        
        self.mat_cov = pca.calculate_covariance_matrix()
        
        self.eigval, self.eigvec, self.var_exp, self.cum_var_exp = \
            pca.perform_PCA()
        
        
    def PCA_plot_2D(self):
        """A 2D plot is created with the two most importend principal
        components, each on an axis. The values of the three targets are
        all plotted in a different color.
        """
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
        plt.xlabel(f'Principal Component 1 -- {self.var_exp[0]:.2f}% variance')
        plt.ylabel(f'Principal Component 2 -- {self.var_exp[1]:.2f}% variance')
        plt.show()
    
    
    def PCA_plot_3D(self):
        """A 3D plot is created with the three most important
        components, each on an axis. The values of the three targets are
        all plotted in a different color.
        """
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
        ax.set_xlabel('Principal Component 1 -- {self.var_exp[0]:.2f}% variance')
        ax.set_ylabel('Principal Component 2 -- {self.var_exp[1]:.2f}% variance')
        ax.set_zlabel('Principal Component 3 -- {self.var_exp[2]:.2f}% variance')
        plt.show()
    
    def var_plot(self, var_exp, cum_var_exp, eigen_vals):
        """Generates figures with information on explained variance per
        PC and cumulative explained variance and a scree plot.
      
        Parameters
        -----------
        var_exp: list[float]
            List with len = n_features containing the variance explained
            for each principal component sorted from high to low.
            
        cum_var_exp: list[float]
            List with len = n_features containing the cumulative
            explained variance of the principal components.
            
        eigen_vals: array
            Array containing the eigenvalues of the system with shape
            (n_features,).
        
        Returns
        -----------
        Two figures: 
            First figure: bar graph of explained variance per PC and 
            step graph of cumulative explained variance.
            Second figure: scree plot
        """
        import matplotlib.pyplot as plt

        # First figure: step and bar plot of variance
        # Figure configurations 
        plt.figure(figsize=(15,15))
        plt.xlabel('Principle Components')
        plt.ylabel('Variance Explained')
        plt.title('Explained variance per principle component and '
                  'cumulative explained variance')
        
        # Plot bar graph - explained variance per PC
        plt.bar(range(len(self.var_exp)), self.var_exp,
                color='blue', alpha=0.5,
                label='Explained variance for each PC')
        
        # Plot step graph - cumulative explained variance
        plt.step(range(len(self.cum_var_exp)), 
                 self.cum_var_exp, color='red',
                 label='Cumulative explained variance')
        # Plot horizontal line at 70% and 99% expl. variance
        plt.axhline(y = 70, color = 'g', linestyle = 'dashed')
        plt.axhline(y = 99, color = 'g', linestyle = 'dashed')
        plt.xticks(fontsize=20)
        plt.legend(loc='best')
        plt.savefig('Varplot.png')
        plt.show()
        
        # Second figure: scree plot
        plt.figure(figsize=(15,15))
        plt.xlabel('Principle Component number')
        plt.ylabel('Eigenvalue')
        plot_name = 'Screeplot'
        plt.title(plot_name)
        plt.plot(range(len(self.eigval)), self.eigval,
                 'o-', linewidth=2)
        plt.show()
        plt.savefig(plot_name)
    
    
    def loading_plots_all_targets(variable_names_list, eig_vals_real, 
                                  eig_vecs_real, PC=1, n_vars='All'):
        """Generates a loading plot of a predefined number of variables with
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
        # Make a list of (eigenvalue, eigenvector) tuples
        eig_val_vec_list = [(abs(eig_vals_real[i]), eig_vecs_real[:,i]) 
                    for i in range(len(eig_vals_real))]
        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_val_vec_list.sort(key=lambda tup: tup[0], reverse=True)

        # Calculate the loading scores of all variables on the PC of interest
        loadings = eig_val_vec_list[PC-1][1].T / np.sqrt(eig_val_vec_list[PC-1][0])
        # Convert this array to a list
        loadings = loadings.tolist()
        
        # Make a list of tuples containing (variable_name, loading)
        var_names_loadings_tuples_list = [(variable_names_list[i], loadings[i]) 
                                        for i in range(len(loadings))]
        #print(f'{var_names_loadings_tuples_list = }')
        # Order this list of tuples by their loading, in descending order
        var_names_loadings_tuples_list.sort(key=lambda num: abs(num[1]),
                                            reverse=True)
        
        # Only select the <n_vars> highest loadings
        if n_vars == 'All':
            selected = var_names_loadings_tuples_list[0:] 
        else:
            selected = var_names_loadings_tuples_list[0:n_vars]
        
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


    def loading_plots_per_target(self, variable_names_list, target=None, PC=1, n_vars='All'):
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
        # Select the correct data.
        if target == 0 or target == 'ppar':
            eig_vals, eig_vecs = self.target1_eigval, self.target1_eigvec
        elif target == 1 or target == 'thrombin':
            eig_vals, eig_vecs = self.target2_eigval, self.target2_eigvec
        elif target == 2 or target == 'cox2':
            eig_vals, eig_vecs = self.target3_eigval, self.target3_eigvec
        # When no specific target is given, give error.
        elif target == None:  
            raise ValueError('Target unknown')
        
        # ==============================================================
        # Calculation    
        # ==============================================================
        # Make a list of (eigenvalue, eigenvector) tuples
        eig_val_vec_list = [(np.abs(eig_vals[i]), eig_vecs[:,i]) 
                       for i in range(len(eig_vals))]
        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_val_vec_list.sort(key=lambda tup: tup[0], reverse=True)
    
        # Calculate the loading scores of all variables on the PC of interest
        loadings = eig_val_vec_list[PC-1][1].T / np.sqrt(eig_val_vec_list[PC-1][0])
        # Convert this array to a list
        loadings = loadings.tolist()
        
        # Make a list of tuples containing (variable_name, loading)
        var_names_loadings_tuples_list = [(variable_names_list[i], loadings[i]) 
                                          for i in range(len(loadings))]
        
        # Order this list of tuples by their loading, in descending order
        var_names_loadings_tuples_list.sort(key=lambda num: abs(num[1]),
                                            reverse=True)
        
        # Only select the <n_vars> highest loadings
        if n_vars == 'All':
            selected = var_names_loadings_tuples_list[0:] 
        else:
            selected = var_names_loadings_tuples_list[0:n_vars]
        
        # Make two separate lists again: variable_names_list containing
        # the names of the variables (ordered) (X axis) and loadings
        # containing the loadings of these variables (Y axis). Their
        # indices match.
        variable_names_list, loadings = zip(*selected)
        
        # ==============================================================
        # Make a bar graph: configuration.
        # ==============================================================
        fig = plt.figure(figsize=(10,5), dpi=150)
        ax = fig.add_axes([0,0,1,1])
        plt.xticks(rotation = 90)  # Make labels readable.
        plot_title = f'Loading plot {target} -- PC{PC}'
        ax.set_title(plot_title)
        ax.set_ylabel('Loading')
        
        # Color all negative loadings in the plot red; all positive
        # loadings green
        red_green = []
        for item in loadings:
            item = item.real
            if item >= 0:
                red_green.append('green')
            else:
                red_green.append('red')
        
        # Create the bar graph.
        ax.bar(variable_names_list, loadings, color=red_green)
        # Save and show the image.
        plt.savefig(plot_title, dpi=300, bbox_inches='tight')
        plt.show()
    
