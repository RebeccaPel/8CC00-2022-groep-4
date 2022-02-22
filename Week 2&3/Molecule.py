# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 16:36:10 2022

@author: 20192157
"""

class Molecule:
    
    # Class variable = descriptors
    
    def __init__(self,df,target):
        '''
        Parameters
        ----------
        df : pandas dataframe
            a dataframe with 34 columns, all descriptors of the molecule which
            is defined in the first column with the smile
        target : str
            must be a string being one of the three options in the target
            column of the dataframe (ppar, thrombin,cox2)

        Returns
        -------
        None.

        '''
        # Creating the instance variables:        
        self.target = target
        self.df = df
        self.df_target = df[(df['Target'] == target)]
        self.columns = len(df.columns) # number of columns
        
        
    def readMolDescriptors(self,index):
        '''
        This function reads in and returns the molecular descriptors of a single
        molecule.

        Parameters
        ----------
        index : int
            index of the molecule of which the descriptors are needed

        Returns
        -------
        descriptors : series
            all the molecular descriptors belonging to the molecule at the index
        '''
        # The descriptors are all the columns besides SMILE and Target
        descriptors = self.df.iloc[index,2:self.columns]
        # The descriptors are now a series and should be a dataframe to be 
        # consistent, also the descritors are in one column and should be in a row:
        descriptors = descriptors.to_frame().transpose()
        
        return descriptors

    
        