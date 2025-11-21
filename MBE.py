#To be completely honest I am not even sure this file was needed but I wanted to do this to see if it would make my life easier for doing all of the feature plots. 

#File created: 11/18/2025

##imports for relevant packages 
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import math
import scipy.stats
import pickle
from scipy.stats import ttest_1samp
import matplotlib.ticker as mticker
#______________Open file___________________
def openfile(path1,path2):
    infile = open(f'./UFS_metrics/{path1}.p', 'rb') 
    actual = pickle.load(infile)
    infile.close()
    
    infile = open(f'./UFS_metrics/{path2}.p', 'rb') 
    forecast = pickle.load(infile)
    infile.close()
    return actual, forecast
    
#_____________Calculate Mean Bias___________________
def calculate_mbe(actual, forecast):   
    if len(actual) != len(forecast):
        raise ValueError("Actual and forecast must have the same length.")

    if forecast.ndim == 2:
        m = []
        for i in range(len(forecast)):
            m.append(forecast[i, :] - actual[i, :])
        errors = np.array(m).flatten()
    else:
        errors = forecast - actual
    mbe = np.nanmean(errors)

    tstat, pval = ttest_1samp(errors, 0, nan_policy='omit')
    sig_flag = (pval < 0.05) if not np.isnan(pval) else False

    return mbe, errors, pval, sig_flag

#_____________Calculate Forecast Error___________________
def calculate_errors(actual, forecast):
    if len(actual) != len(forecast):
        raise ValueError("Actual and forecast must have the same length.")

    errors = np.array(forecast) - np.array(actual) #just calculate and return the error. 
    return errors
    
#_____________Make Bulk MBE Array w/ Significance for Bar Plots_________________
def BP_bulk_array(actual, forecast): 
    error_bulk = np.zeros((4,4,8))
    sig_mask   = np.zeros((4,4,8), dtype=bool)
    sig_pvals  = np.zeros((4,4,8))
    
    for i in range(4): #prototypes
        f = forecast[:, :, i, :] #indicate chosen prototype           
        a = actual[:, :, :]              
        
        for j in range(8):                      
            #all days
            mbe, errs, p, sig = calculate_mbe(a[:, j, :], f[:, j, :])
            error_bulk[i,0,j] = mbe
            sig_mask[i,0,j]   = sig
            sig_pvals[i,0,j]  = p
        
            #14 days
            mbe, errs, p, sig = calculate_mbe(a[:, j, 14], f[:, j, 14])
            error_bulk[i,1,j] = mbe
            sig_mask[i,1,j]   = sig
            sig_pvals[i,1,j]  = p
        
            #20 days
            mbe, errs, p, sig = calculate_mbe(a[:, j, 20], f[:, j, 20])
            error_bulk[i,2,j] = mbe
            sig_mask[i,2,j]   = sig
            sig_pvals[i,2,j]  = p
        
            #30 days
            mbe, errs, p, sig = calculate_mbe(a[:, j, 30], f[:, j, 30])
            error_bulk[i,3,j] = mbe
            sig_mask[i,3,j]   = sig
            sig_pvals[i,3,j]  = p
    return error_bulk, sig_mask, sig_pvals

#_____________Make Bulk Array for BW Plots_________________
def BW_bulk_array(actual, forecast): 
    error_bulk = np.empty((4,4,8), dtype=object)
    for i in range(4):
        f = forecast[:,:,i,:]
        a   = actual[:,:,:]
    
        for j in range(8):
            #all days
            error_all = calculate_errors(a[:,j,:], f[:,j,:])
            error_all_yearly_mean = np.nanmean(error_all, axis=1) #need to take mean here bc the errors_all returns (7 years, 36 leads)
            error_bulk[i,0,j] = error_all_yearly_mean

            #14 days
            error_14 = calculate_errors(a[:,j,14], f[:,j,14])
            error_bulk[i,1,j] = error_14

            #20 days
            error_20 = calculate_errors(a[:,j,20], f[:,j,20])
            error_bulk[i,2,j] = error_20

            #30 days
            error_30 = calculate_errors(a[:,j,30], f[:,j,30])
            error_bulk[i,3,j] = error_30
            
    return error_bulk

#_____________Make MBE Barplot_________________
def MBE_barplot(lead_times,forecast_dates, prototypes, colors, 
                error_bulk, sig_mask, sig_pvals,
                title_str, save_str, ylim_low, ylim_high):
    
    nF = len(forecast_dates)
    nP = len(prototypes)
    
    x = np.arange(nF)
    bar_width = 0.18
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 14))
    plt.suptitle(f'{title_str} Mean Forecast Error by UFS Initalization and Prototype', fontsize=18)
    
    axes = axes.flatten()
    for i in range(4):
        ax = axes[i]
    
        for j in range(nP):  #prototype index
            y = error_bulk[j, i, :] 
            
            bars = ax.bar(x + j*bar_width, y,bar_width,color=colors[j],label=prototypes[j])
    
            #add significance stars
            for f, bar in enumerate(bars):
                if sig_mask[j, i, f]:
                    height = bar.get_height()
                    # place star a little above the top (positive or negative)
                    offset = 0.5 if height >= 0 else -0.5
                    ax.text(bar.get_x() + bar.get_width()/2,height + offset,"*",ha='center',va='bottom' if height >= 0 else 'top',fontsize=16)
    
        #x-axis formatting
        ax.set_xticks(x + bar_width*1.5)
        ax.set_xticklabels(forecast_dates, fontsize=12)
    
        #y-axis formatting
        ax.set_ylim(ylim_low, ylim_high)
        #ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
        ax.set_ylabel(lead_times[i], fontsize=14)
        ax.tick_params(labelsize=12)
    
        ax.axhline(y=0, color='k', linestyle='-', lw = 0.9)
        if i == 0:
            ax.legend(bbox_to_anchor=(0.85, 0.97), loc='upper left')
        if i == 3:
            ax.set_xlabel("Forecast Initalization", fontsize=15)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(f'{save_str}')
    plt.show()
    
#_____________Make Forecast Error BW Plot_________________
def Error_BWplot(lead_times,forecast_dates, prototypes, colors, 
                error_bulk, title_str, save_str, ylim_low, ylim_high):
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharey=True)
    plt.suptitle(f'{title_str} Forecast Error Distributions by UFS Initalization and Prototype', fontsize=18)
   
    nF = len(forecast_dates) #should be 8
    nP = len(prototypes) #should be 4... depending

    base_x = np.arange(nF) #create range for these ofc
    
    group_width = 0.4
    box_width = group_width / (nP + 1)
    offsets = np.linspace(-group_width/2, group_width/2, nP)
    
    for i in range(4):
        ax = axes[i]
    
        for j in range(nP):
            xpos = base_x + offsets[j]
            data = []
            for f in range(nF):
                arr = error_bulk[j,i,f]
                clean = arr[~np.isnan(arr)]
                if clean.size == 0:
                    clean = np.array([np.nan])
                data.append(clean)
    
            bp = ax.boxplot(data, positions=xpos, widths=box_width, patch_artist=True, manage_ticks=False,label=prototypes[j])
            # color each prototype
            for patch in bp['boxes']:
                patch.set_facecolor(colors[j])
    
        ax.set_xticks(base_x)
        ax.set_xticklabels(forecast_dates, fontsize=12)
        ax.set_ylim(ylim_low, ylim_high)
        ax.set_ylabel(lead_times[i], fontsize=14)
        ax.axhline(0, color='black', linewidth=1)
        ax.tick_params(axis='both', labelsize=12)
        
        ax.axhline(y=0, color='k', linestyle='-', lw = 0.9)
        if i == 0:
            ax.legend(bbox_to_anchor=(0.85, 0.97), loc='upper left')
        if i == 3:
            ax.set_xlabel("Forecast Initalization", fontsize=15)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(f'{save_str}')
    plt.show()