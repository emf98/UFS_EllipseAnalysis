##Code for reducing bulk in plotting files

#file generated 9/19/2025, E. Fernandez
#modified for use with UFS data 12/3/2025, E. Fernandez

#The following definition statements are contained within this file:

#DATA PROCESSING RELATED:
#extrapolate(data); returns same array with additional column added on
#preprocess_ellipse(input,shift,idx); returns input array
#BWcheckevent_label(posXtest,input2,idx); returns arrays for BW plots
#CScheckevent_label(posXtest,data_array,idx); returns 3D array for cross section plots

#PLOTTING RELATED:
##Box and WHisker Plot
#BWplot(Tpos,Tneg,Fpos,Fneg,metrics_list,loc_str,save_str)

#Horizontal GPH 
#GPH_horzCS(GPH_cpos,GPH_cneg,GPH_Fpos,GPH_Fneg,
               #GPHA_cpos,GPHA_cneg,GPHA_Fpos,GPHA_Fneg,
               #colorbarMin, colorbarMax, colorspace,
               #loc_str, lat, lon,save_loc):
            
#Vertical GPH 
#def GPH_vertCS(GPH_cpos,GPH_cneg,GPH_Fpos,GPH_Fneg,
               #colorbarMin, colorbarMax, colorspace,
               #loc_str, lat, lon, save_loc):
        
#Horizontal Temp
#Temp_horzCS(EHF_cpos,EHF_cneg,EHF_Fpos,EHF_Fneg, loc_str, lat, lon, save_loc,
               #colorbarMin, colorbarMax, colorspace):

#relevant import statements
import numpy as np
import math
import pandas as pd
import xarray as xr 
import pickle 
import matplotlib.pyplot as plt

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from geopy.distance import great_circle
import cartopy.crs as ccrs
from matplotlib.path import Path

#__________daily anomaly calculation__________

def daily_anomaly(target):
    dailymean = np.nanmean(target,axis=1)
    anom = np.zeros_like(target)
    for t in np.arange(target.shape[1]):
         anom[:,t] = target[:,t] - dailymean
    print(anom.shape)
    return anom; 

#_____________data extrapolation_____________

def extrapolate(data):
    data_test = data[..., 0:1] 
    data_test.shape
    data = np.concatenate([data, data_test], axis=-1)
    return data;

#____________process input dataset____________

def preprocess_andopen(region, dataset, ufs_index, region_type_file):
    print(region)
    print(dataset)
    
    input = np.empty((7*8*18,4)) # set empty array
    
    print("Load data...")
    ##open ellipse metric files
    infile = open(f"../UFS_metrics/{dataset}_wind10.p", 'rb')
    wind10 = pickle.load(infile)
    infile.close()

    infile = open(f"../UFS_metrics/{dataset}_size10.p", 'rb') 
    size10 = pickle.load(infile)
    infile.close()

    infile = open(f"../UFS_metrics/{dataset}_ratio10.p", 'rb') 
    rat10 = pickle.load(infile)
    infile.close()

    infile = open(f"../UFS_metrics/{dataset}_ephi10.p", 'rb') 
    ephi10 = pickle.load(infile)
    infile.close()

    infile = open(f"../UFS_metrics/{dataset}_cenlat10.p", 'rb')
    cenlat10 = pickle.load(infile)
    infile.close()

    infile = open(f"../UFS_metrics/{dataset}_cenlon10.p", 'rb')
    cenlon10 = pickle.load(infile)
    infile.close()

    infile = open(f"../UFS_metrics/{dataset}_gph.p", 'rb') 
    gph = pickle.load(infile)
    infile.close()
    #######################################
    print("Returning final input array.")
    ##need to check if ERA5 or UFS to account for choosing prototype from UFS. 
    if dataset == str("ERA5"):
        if region == str('Europe') or region == str('Canada'):
            metrics_list = ["Geopotential Height (m)","Central Latitude ($^o$)",
                        "Central Longitude ($^o$)","Zonal-Mean Wind (m/s)"]
            input[:,0] = gph[:,:,:18].reshape((7*8*18))
            input[:,1] = cenlat10[:,:,:18].reshape((7*8*18))
            input[:,2] = cenlon10[:,:,:18].reshape((7*8*18))
            input[:,3] = wind10[:,:,:18].reshape((7*8*18))
            
        elif region == str('SEUS'):
            metrics_list = ["Geopotential Height (m)","Central Latitude ($^o$)",
                        "Size ( *10$^7$m)","Zonal-Mean Wind (m/s)"]
            input[:,0] = gph[:,:,:18].reshape((7*8*18))
            input[:,1] = cenlat10[:,:,:18].reshape((7*8*18))
            input[:,2] = size10[:,:,:18].reshape((7*8*18))
            input[:,3] = wind10[:,:,:18].reshape((7*8*18))
            
    if dataset == str("UFS"):
        if region == str('Europe') or region == str('Canada'):
            metrics_list = ["Geopotential Height (m)","Central Latitude ($^o$)",
                        "Central Longitude ($^o$)","Zonal-Mean Wind (m/s)"]
            input[:,0] = gph[:,:,ufs_index,:18].reshape((7*8*18))
            input[:,1] = cenlat10[:,:,ufs_index,:18].reshape((7*8*18))
            input[:,2] = cenlon10[:,:,ufs_index,:18].reshape((7*8*18))
            input[:,3] = wind10[:,:,ufs_index,:18].reshape((7*8*18))
            
        elif region == str('SEUS'):
            metrics_list = ["Geopotential Height (m)","Central Latitude ($^o$)",
                        "Size ( *10$^7$m)","Zonal-Mean Wind (m/s)"]
            input[:,0] = gph[:,:,ufs_index,:18].reshape((7*8*18))
            input[:,1] = cenlat10[:,:,ufs_index,:18].reshape((7*8*18))
            input[:,2] = size10[:,:,ufs_index,:18].reshape((7*8*18))
            input[:,3] = wind10[:,:,ufs_index,:18].reshape((7*8*18))
    
    ###################################################
    print("Opening corresponding temperature data and indices...")
    #load output data file for TESTING
    infile = open(f'./new_data/2classtemps_{str(region_type_file)}.p',"rb",)
    TEST_temp = pickle.load(infile) 
    infile.close()
    #true pos
    infile = open(f'./index/pos_{str(region_type_file)}.p', 'rb') 
    pos = pickle.load(infile)
    infile.close()
    #false pos
    infile = open(f'./index/Fpos_{str(region_type_file)}.p', 'rb') 
    Fpos = pickle.load(infile)
    infile.close()
    #true neg
    infile = open(f'./index/neg_{str(region_type_file)}.p', 'rb') 
    neg = pickle.load(infile)
    infile.close()
    #false neg
    infile = open(f'./index/Fneg_{str(region_type_file)}.p', 'rb') 
    Fneg = pickle.load(infile)
    infile.close()
    
    return input, TEST_temp, metrics_list, pos, Fpos, neg, Fneg;

#____________calculate rates of events____________

#only for box and whisker plots
def BWcheckevent_label(posXtest,inp):
    
    ####now I wanna make these plots SO ... I am adding an index column on to X_test ... full version. 
    ranges = np.array([x for x in range(0,len(inp[:,0]),1)])
    print("Check shape of index array ...", ranges.shape)
    
    ##Check whether event is in the desired list (true pos/neg or false pos/neg)
    posXtest_set = set(posXtest) # as a set. 

    pos_corr_events = []
    pos_corr_total_events = []
    print("Checking indices ...")
    for i in range(len(ranges)):
        date_index = ranges[i]
        if date_index not in posXtest_set:
            continue
        elif date_index in posXtest_set:
            features = inp[i, :]
            pos_corr_events.extend(features)
            pos_corr_total_events.append(0)               
    ##reshape
    Tpos = np.array(pos_corr_events).reshape(len(pos_corr_total_events),4)
    
    return Tpos;

##for cross sections
def CScheckevent_label(posXtest,data_array):
    
    ##get data array lat/lon dimensions
    array_shape = data_array.shape
    lat_dim = array_shape[1]
    lon_dim = array_shape[2]

    ranges = np.array([x for x in range(0,len(data_array[:,0]),1)])
    print("Check shape of index array ...", ranges.shape)
    
    ##Check whether event is in the desired list (true pos/neg or false pos/neg)
    posXtest_set = set(posXtest)

    pos_corr_total_events = []
    data_list = []
    
    print("Checking indices ...")
    for i in range(len(ranges)):
        date_index = ranges[i]
        if date_index not in posXtest_set:
            continue
        elif date_index in posXtest_set:
            pos_corr_total_events.append(0)
            data_list.extend(data_array[i,:,:])
            
    ##reshape
    data_list = np.array(data_list).reshape(len(pos_corr_total_events),lat_dim,lon_dim)
    
    return data_list;

#___________________________PLOTTING RELATED DEFINITIONS___________________________

def BWplot(Tpos,Tneg,Fpos,Fneg,metrics_list,loc_str,save_str):
    import matplotlib.ticker as mticker
    myLocator = mticker.MultipleLocator(2)

    metrics = metrics_list
    ticks = ['True +', 'False -', 'True -', 'False +'] #set tick numbers for dataset
    ind = [2, 4, 6, 8]  # the x locations for the groups
    w = 0.25 #box-plot width
    c = ["midnightblue","royalblue","mediumvioletred","magenta"]
    fs = 14

    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    plt.suptitle("Distribution of RF Input Features, "+str(loc_str), fontsize = 18, x=0.53)
    axes = axes.flatten()
    for i in range(0,4):
        C_pos = Tpos[:,i]
        F_neg = Fneg[:,i]
        C_neg = Tneg[:,i]
        F_pos = Fpos[:,i]

        ##remove NaNs...
        C_pos = C_pos[~np.isnan(C_pos)]
        F_neg = F_neg[~np.isnan(F_neg)]
        C_neg = C_neg[~np.isnan(C_neg)]
        F_pos = F_pos[~np.isnan(F_pos)]

        a1 =axes[i].boxplot([C_pos,F_neg,C_neg,F_pos], positions= [2,4,6,8], widths=w, patch_artist=True)
        for bplot in (a1,):
            for patch, color in zip(bplot['boxes'], c):
                patch.set_facecolor(color)
        axes[i].set_xticks(ind)                  
        axes[i].set_xticklabels(ticks, fontsize=fs)  
        axes[i].set_ylabel(str(metrics[i]), fontsize=fs)
        axes[i].tick_params(axis='y', labelsize=fs)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)   
    plt.savefig(str(save_str),bbox_inches = 'tight')
    plt.show()
    return ;

########################################################################
# #horizontal cross sections
def GPH_horzCS(GPH_cpos,GPH_cneg,GPH_Fpos,GPH_Fneg,
               GPHA_cpos,GPHA_cneg,GPHA_Fpos,GPHA_Fneg,
               colorbarMin, colorbarMax, colorspace,
               loc_str, lat, lon,save_loc):
    fs = 18
    fig, axes = plt.subplots(2, 2, figsize=(14, 14), subplot_kw={'projection': ccrs.NorthPolarStereo()})
    plt.suptitle("Composites of 10hPa GPH during 90th Percentile Confident Predictions,"+str(loc_str), fontsize=21)   

    titles = ["True Positive", "True Negative", "False Positive", "False Negative"]
    data1 = [
        np.nanmean(extrapolate(GPH_cpos), axis=0),
        np.nanmean(extrapolate(GPH_cneg), axis=0),
        np.nanmean(extrapolate(GPH_Fpos), axis=0),
        np.nanmean(extrapolate(GPH_Fneg), axis=0)
    ]

    data2 = [
        np.nanmean(extrapolate(GPHA_cpos), axis=0),
        np.nanmean(extrapolate(GPHA_cneg), axis=0),
        np.nanmean(extrapolate(GPHA_Fpos), axis=0),
        np.nanmean(extrapolate(GPHA_Fneg), axis=0)
    ]

    axes = axes.flatten()

    for i in range(4):
        ax = axes[i]
        ax.set_title("10hPa GPH " + str(titles[i]), fontsize=fs-1, y=1.01, x=0.5)
        ax.coastlines()

        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)

        level = np.arange(colorbarMin, colorbarMax + colorspace, colorspace)

        h = ax.contourf(lon, lat, data2[i], level, transform=ccrs.PlateCarree(), cmap="RdBu_r")

        #highlight_level1 = 15500
        highlight_level1 = 30000
        #highlight_level2 = 31250
        level_cont = np.arange(28500, 32500, 250)
        #level_cont = np.arange(14000, 16501, 150)

        for level in level_cont:
            linestyle = 'dashed'
            linewidth = 1.0
            color = 'black'

            if level == highlight_level1:
                linestyle = 'solid'
                linewidth = 2.5

            g_cont = ax.contour(
                lon, lat, data1[i], levels=[level],
                transform=ccrs.PlateCarree(),
                colors=[color], linestyles=[linestyle], linewidths=[linewidth]
            )

            labels = ax.clabel(g_cont, inline=True, fontsize=12)
            for text in labels:
                text.set_backgroundcolor("white")
                text.set_bbox(dict(facecolor="white", edgecolor="none", pad=.35))

        # Colorbar
        cbar = fig.colorbar(h, ax=ax, orientation="vertical", shrink=0.75, pad=0.05, aspect=30)
        cbar.ax.tick_params(labelsize=fs - 2)

    plt.tight_layout()
    plt.subplots_adjust(top=0.98)
    plt.savefig(str(save_loc),bbox_inches = 'tight')
    plt.show()
    return ;
