# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 16:36:10 2022

@author: 20192157
"""
import pandas as pd

class Molecule:
    """A data class that describes a molecule, with various molecular
    descriptors. Can also read data on molecules.
    
    Parameters
    ----------
    df : pandas DataFrame
        A dataframe molecular descriptors as variables i.e. columns
        and SMILES (molecule names) as objects.
        
    target : str
        Must be a string being one of the options in the target
        column of the DataFrame (e.g. 'ppar', 'thrombin', 'cox2').

    Attributes
    ----------
    target : str
        Must be a string being one of the options in the target
        column of the DataFrame (e.g. 'ppar', 'thrombin', 'cox2').
        
    df : pandas DataFrame
        A dataframe molecular descriptors as variables i.e. columns
        and SMILES (molecule names) as objects.
        
    df_target : pandas DataFrame
        This is a copy of parameter df, but only the rows corresponding to the
        parameter target are kept, the rest of the rows have been discarded.
        
    columns: int
        The number of columns that are present in df (and df_target)

    Methods
    -------
    read_mol_descriptors(index):
        This function reads in and returns the molecular descriptors of
        a single molecule, at the given index in the DataFrame.
    """    
    # Class variable = list[descriptors: str]
    df = pd.read_csv(r'QSAR_3_large_data.csv')
    descriptors = list(df.columns.values)
    
    def __init__(self, df, target):
        self.target = target
        self.df = df
        # self.df_target = df[(df['Target'] == target)]
        self.df_target = df
        self.columns = len(df.columns) # number of columns
        
        
    def read_mol_descriptors(self, index):
        """
        This function reads in and returns the molecular descriptors of
        a single molecule, at the given index in the DataFrame.

        Parameters
        ----------
        index : int
            The index of the molecule of which the descriptors are
            needed.

        Returns
        -------
        descriptors : Series
            All the molecular descriptors belonging to the molecule at
            the index.
        """
        # The descriptors are all the columns besides SMILES and Target
        descriptors = self.df.iloc[index, 2:self.columns]
        # The descriptors are now a series and should be a dataframe to be 
        # consistent, also the descritors are in one column and should be in a row:
        descriptors = descriptors.to_frame().transpose()
        
        return descriptors

    
        