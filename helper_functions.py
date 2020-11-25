# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 23:03:21 2020

@author: dexte
"""

#imports packages 
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, Model, Parameter
import lmfit #note lmfit model requires a list for the y paramaters (this is confusing)
from scipy.optimize import fsolve 
import os #for join
import re #for regular expressions 
import copy
import seaborn as sns
import os 
from pathlib import Path

#isotherm plotting function
def plot_halotherms(X_data,alpha_data,alpha_func,params,title="Data with Best Fit"):
    
    """Plots halotherms along with data for a given fit
    INPUT: X_data, alpha_data, alpha_func parameters 
    OUTPUT: Plots halotherms along with data 
    """
    X_data_df = pd.DataFrame(X_data).transpose()
    alpha_data_df = pd.DataFrame(alpha_data)
    
    alpha_data_df.columns = ["alpha"]
    X_data_df.columns = ['FeCl3','Li-TFSI']
    salt_conc_list = X_data_df['Li-TFSI'].unique()
    salt_conc_list.sort()
    
    #make color iterator
    colors = sns.color_palette("husl", len(salt_conc_list))
    color_iter = iter(colors)
    
    fig,axfull = plt.subplots(1,2,figsize=(14,5),dpi=300)
    ax_flat = axfull.flatten()
    #plt.figure(100,figsize=(14,5))
    colors = sns.color_palette("husl", len(alpha_data))
    
    FeCl3_sim = np.linspace(0.001,5,1000)
    halotherm_df_sim = pd.DataFrame({'FeCl3 (mM)':FeCl3_sim})
    halotherm_df_real = pd.DataFrame()
    halotherm_df_real['blank'] = pd.Series([i for i in range(0,100)])
    for salt_conc in salt_conc_list:
        x = X_data_df[X_data_df['Li-TFSI']==salt_conc]['FeCl3']
        y = alpha_data_df[X_data_df['Li-TFSI']==salt_conc]

        Salt_sim = np.ones(len(FeCl3_sim))*salt_conc
        X_sim = (FeCl3_sim, Salt_sim)
        y_sim = alpha_func(X_sim,*params)
        color = next(color_iter)
        halotherm_df_sim[salt_conc] = y_sim

        """if type(x.to_numpy()) is list:
            x_tmp = x.to_numpy().flatten()
        else:
            x_tmp = [x.to_numpy()].flatten()
        
        if type(y.to_numpy()) is list:
            y_tmp = y.to_numpy().flatten()
        else:
            y_tmp = [y.to_numpy()].flatten()"""
        
        x_tmp = x.to_numpy().flatten()
        y_tmp = y.to_numpy().flatten()
            
        halotherm_df_real[str(salt_conc) + ' FeCl3 (mM)'] = pd.Series(x_tmp)
        halotherm_df_real[str(salt_conc) + ' alpha'] = pd.Series(y_tmp)

        
        ax_flat[0].semilogx(x,y,'o',color=color,label = "Data %0.2f mM Li-TFSI"%salt_conc,alpha=0.8)
        ax_flat[0].semilogx(FeCl3_sim,y_sim,'--',color=color,alpha=1,label = "Best Fit at %0.2f mM Li-TFSI"%salt_conc)
        ax_flat[1].plot(x,y,'o',color=color,label = "Data %0.2f mM Li-TFSI"%salt_conc,alpha=0.8)
        ax_flat[1].plot(FeCl3_sim,y_sim,'--',color=color,alpha=1,label = "Best Fit at %0.2f mM Li-TFSI"%salt_conc)
    
    
    fig.suptitle(title)
    ax_flat[1].set_xlabel(r"$FeCl_3$ (mM)")
    ax_flat[1].set_ylabel(r"Doped Sites/Total Sites $(\alpha)$")
    ax_flat[0].set_xlabel(r"$FeCl_3$ (mM)")
    ax_flat[0].set_ylabel(r"$(\Theta)$")
    ax_flat[0].set_xlim((0.05,5))
    ax_flat[1].legend(loc='center left',bbox_to_anchor=(1.0, 0.5))

    
def plot_data_and_fit_ian(c_FeCl3,alpha,modelFun,modelParams,title="Data with best fit"):
    c_FeCl3 = c_FeCl3
    alpha = alpha
    fig2, ax_box= plt.subplots(1,2,figsize=(14,5))
    ax = ax_box.flatten()
    x_sim = np.linspace(0,5)
    y_sim = modelFun(x_sim,*modelParams)
    #plot data and fit (normal vs normal)
    ax[0].semilogx(x_sim, y_sim,'--')  # plots fit
    ax[0].semilogx(c_FeCl3, alpha,'o') # plots datapoints 
    ax[0].set_xlabel(r"$FeCl_3$ Solution Concentration (mM) ")
    ax[0].set_ylabel(r"Doped Sites/Total Sites $(\alpha)$")
    #plot data and fit (semilogy plot)
    ax[1].plot(x_sim, y_sim,'--')  # plots fit
    ax[1].plot(c_FeCl3, alpha,'o') # plots datapoints 
    ax[1].set_xlabel(r"$FeCl_3$ Solution Concentration (mM) ")
    ax[1].set_ylabel(r"$(\Theta)$")
    fig2.suptitle(title)

    return fig2, ax_box

def plot_data(data):
    fig, ax = plt.subplots()
    colors = sns.color_palette("husl", data.shape[1])
    ax.set_prop_cycle('color',colors)
    data.plot(x='wavelength',figsize=(15,8),ax=ax) #take your "cleaned" data and plot it along with A+ and A0 peaks 
    plt.xlabel('wavelength (nm)')
    plt.ylabel('Absorbance (a.u.)')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5)) #puts legend on right
    return fig, ax

def plot_data_and_fit_nonclosed(c_FeCl3, c_S,alpha,modelFun,modelParams,title="Data with best fit") :
    
    
    X_sim = copy.deepcopy((c_FeCl3[np.argsort(c_FeCl3)], c_S[np.argsort(c_FeCl3)]))
    y_sim = modelFun(X_sim,*modelParams)
    
    
    fig3, ax_box= plt.subplots(1,2,figsize=(14,5))
    ax = ax_box.flatten()
 
    #plot data and fit (normal vs normal)
    ax[0].semilogx(X_sim[0], y_sim,'r--', label = "Best Fit")  # plots fit
    ax[0].semilogx(c_FeCl3, alpha,'bo', label = "Data") # plots datapoints 
    ax[0].set_xlabel(r"$FeCl_3$ Solution Concentration (mM) ")
    ax[0].set_ylabel(r"$(\Theta)$")
    #plot data and fit (semilogy plot)
    ax[1].plot(X_sim[0], y_sim,'r--', label = "Best Fit")  # plots fit
    ax[1].plot(c_FeCl3, alpha,'bo', label = "Data") # plots datapoints 
    ax[1].set_xlabel(r"$FeCl_3$ Solution Concentration (mM) ")
    ax[1].set_ylabel(r"$(\Theta)$")
    ax[1].legend(loc='center left',bbox_to_anchor=(1.0, 0.5))

    fig3.suptitle(title)

    return fig3, ax_box
    
    
    
    
    
    
    
    
    new_param_tuple = tuple([p for p in param_vals if not p == 1])
    plt.figure(1,figsize=(7,5))
    plt.title("Non closed form model, constant FeCl3 assumption \n" +r"$K_i:$ %e  $K_{ii}:$ %e $Ct^0:$ %0.4f  "%tuple(new_param_tuple))
    plt.semilogx(X_fake[0],y_fake,'r--',label="best fit")
    plt.legend(loc='best')
    plt.semilogx(X[0], alpha, 'bo')
    plt.xlabel(r"$FeCl_3$ concentration (mM)")
    plt.ylabel(r"$(\Theta)$")
    
    plt.figure(2,figsize=(7,5))
    
    plt.plot(X[0], alpha, 'bo',label="data")
    plt.plot(X_fake[0],y_fake,'r--',label="best fit")
    plt.legend(loc='best')
    plt.xlabel(r"$FeCl_3$ concentration (mM)")
    plt.ylabel(r"$(\Theta)$")
    
    X_only2 = (X[0],X[1])
    plot_halotherms(X_only2, alpha, ncf_constFeCl3, param_vals)
        
    
def get_conc_from_column_nane(col_name):
    '''Returns concentration data from column name input
    INPUT: column name as string
    OUTPUT: Dictionary of concentrations and dates
    '''
    if ':' in col_name:
        contains_day = True
    else:
        contains_day = False 
    if "Neat" in col_name or "neat" in col_name:
        c_FeCl3 = 0; c_Salt =0 
        return (c_FeCl3,c_Salt)
    
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", col_name)  # from https://stackoverflow.com/questions/4703390/how-to-extract-a-floating-number-from-a-string
    if(contains_day):    
        day = numbers[0]
        c_FeCl3 = numbers[1]
        if '+' not in col_name:
            c_Salt = 0
            return (c_FeCl3,c_Salt)
        c_Salt = numbers[3]
    else:
        day = None
        c_FeCl3 = numbers[0]
        if '+' not in col_name:
            c_Salt = 0
            return (c_FeCl3,c_Salt)
        c_Salt = numbers[2]

        
    return (c_FeCl3, c_Salt)

#testing funtion
#display(get_conc_from_column_nane('10.4 mM FeCl3 + 100.6 mM Li-TFSI'))
