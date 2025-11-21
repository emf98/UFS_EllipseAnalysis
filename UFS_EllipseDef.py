## DEFINITION FILE FOR UFS Data
#Solve for the best fit for an ellipse.. then plot it!
# Generated: 17 November 2025, E Fernandez
##This is a continuation off the fitEllipse3_new.py python file wherein the actual ellipse calculation definitions are included here. 

#Follows an approach suggested by Fitzgibbon, Pilu and Fischer in Fitzgibbon, A.W., Pilu, M., 
# and Fischer R.B., Direct least squares fitting of ellipsees, Proc. of the 13th Internation 
# Conference on Pattern Recognition, pp 253–257, Vienna, 1996.  
# Discussed on http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html and uses relationships 
#  found at http://mathworld.wolfram.com/Ellipse.html.
#
# ***Update from Authors: Andrew Fitzgibbon, Maurizio Pilu, Bob Fisher
# http://research.microsoft.com/en-us/um/people/awf/ellipse/
# Reference: "Direct Least Squares Fitting of Ellipses", IEEE T-PAMI, 1999
# Citation:  Andrew W. Fitzgibbon, Maurizio Pilu, and Robert B. Fisher
# Direct least-squares fitting of ellipses,
# IEEE Transactions on Pattern Analysis and Machine Intelligence, 21(5), 476--480, May 1999
#  @Article{Fitzgibbon99,
#   author = "Fitzgibbon, A.~W.and Pilu, M. and Fisher, R.~B.",
#   title = "Direct least-squares fitting of ellipses",
#   journal = pami,
#   year = 1999,
#   volume = 21,
#   number = 5,
#   month = may,
#   pages = "476--480"
#  }
# 
# This is a more bulletproof version than that in the paper, incorporating
# scaling to reduce roundoff error, correction of behaviour when the input 
# data are on a perfect hyperbola, and returns the geometric parameters
# of the ellipse, rather than the coefficients of the quadratic form.


import datetime as dt
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.feature
from cartopy.util import add_cyclic_point
import cartopy.crs as ccrs
import xarray as xr
import math
import netCDF4
#from get_ellipse_metrics import get_emetrics_max_min
from fitEllipse3_new import fitEllipseContour
from fitEllipse3_new import point_inside_polygon
#from EllipseCalc import ellipse_calc
from geopy.distance import great_circle
from matplotlib.patches import Polygon
from matplotlib import cm #colormaps!
import pickle

### Start definition statement for UFS data.

def UFS_one_ellipse_calc(plot_lev,the_contour,windlat,y,m,d,t_d):
    #Start with defining the desired period of record
    year = y
    month = m
    day = d
    date1 = dt.datetime(year,month,day,0)  #first date to plot
    total_days = t_d
    first_fhr = 0
    hours = total_days * 24
    hr_inc = 24
    times = [date1 + dt.timedelta(hours=x) for x in range(0,hours,hr_inc)]
    date_list = netCDF4.date2num(times,units="hours since 1800-01-01 00:00:00",calendar="gregorian") #change dates to netcdf times

    date_str = f"{y}{m:02d}{d:02d}"

    #indicate list of potential prototypes
    prototype = ["Prototype5","Prototype6","Prototype7","Prototype8"]

    #set empty arrays for values
    rat10 = np.empty((len(prototype),total_days))
    cenlat10 = np.empty((len(prototype),total_days))
    cenlon10 = np.empty((len(prototype),total_days))
    wind10 = np.empty((len(prototype),total_days))
    size10 = np.empty((len(prototype),total_days))
    ep10 = np.empty((len(prototype),total_days))
    num10 = np.empty((len(prototype),total_days))
    
    for mm in range(len(prototype)):
        print("##########################")
        print(prototype[mm])
        rat = []
        cenlt = []
        cenln = []
        w = []
        sz = []
        ep = []
        num = []
        
        gfile = xr.open_dataset( f"/network/rit/lab/wulab/forecast/s2s/ufs/{prototype[mm]}/gh/gh_plevs_{date_str}.nc"
)
        g_files = gfile["gh"]
        tfile = xr.open_dataset( f"/network/rit/lab/wulab/forecast/s2s/ufs/{prototype[mm]}/t/t_plevs_{date_str}.nc"
)
        t_files = tfile["t"]
        ufile = xr.open_dataset( f"/network/rit/lab/wulab/forecast/s2s/ufs/{prototype[mm]}/u/u_plevs_{date_str}.nc"
)
        u_files = ufile["u"]
        
        #print("Partitioning and averaging data.")
        #print("First, g data")
        g_data = g_files.loc[dict(latitude=slice(90,0),lev=plot_lev)]
        #print("u data")
        u_data = u_files.loc[dict(latitude=slice(windlat+1,windlat-1),lev=plot_lev)].mean(dim='longitude')
        #print("t data")
        t_data = t_files.loc[dict(latitude=slice(90,75),lev=plot_lev)]
        tlats = t_data['latitude'].values
        weights=np.cos(np.deg2rad(t_data.latitude))
        weights.name = "weights"
        
        #print("lat and lon")
        lats_era = g_data["latitude"].values
        lons_era = g_data["longitude"].values
        
        for date in date_list: #loop through each day 
            #print(f"date: {date_list}")
            ind = np.where(date_list == date)[0]
            for d in ind:
                t = d
                height = g_data.loc[dict(time=times[t],lev=plot_lev)]
                height = height[0,:] #for whatever reason, it was hanging onto the lev dimension here. 
                u = u_data.loc[dict(time=times[t],lev=plot_lev)].mean(dim='latitude')
                tdata = t_data.loc[dict(time=times[t],lev=plot_lev)].mean(dim='longitude')   
                #tdata2 = t2_data.loc[dict(time=times[t])]
            temp = tdata.weighted(weights).mean(dim='latitude')
            
            #print("t2data")
            formatted_date = times[t].strftime("%Y%m%d")
            emark = []
            eline = []
            cs_temp = []
            #print(date)
            print("Date "+times[t].strftime("%d %b %Y"))
            valid_label = (times[t].strftime("Valid: %H UTC %d %b %Y"))
            
            plt.Figure(figsize=(15,15),dpi=120)#figure(figsize=(12,12),dpi=1200) <---Set fig size here!
            #ax = plt.subplot(1,1,1,projection=ccrs.Orthographic(0,90))
            ax = plt.axes(projection=ccrs.Orthographic(0,90))
            ax.add_feature(cartopy.feature.LAND, zorder=1, edgecolor='dimgray',facecolor='none')
            #ax.set_extent([-180,180,0,90],ccrs.PlateCarree())
            ax.outline_patch.set_edgecolor('none')
            gl = ax.gridlines(color="grey",linestyle=":",linewidth=0.5)#=lat_lon_width)
            
            clevs = range(18000,33500,250)
            [x,y] = np.meshgrid(lons_era,lats_era)
            #[x2,y2] = np.meshgrid(tlons_era,tlats_era)
            
            plt.title(str(plot_lev)+"hPa Elliptical Diagnostics\n"+valid_label)
            mem_color = "blue"
            mem_lw = 3.0
            mem_ms = 8
            cont_color = "#aec2e2"
            cont_lw = 1.0
            cs_temp.append(ax.contour(x,y,height,levels=clevs,linewidths=cont_lw,colors=cont_color))
            
            line_w = np.array(clevs,dtype=np.float)
            for i,c in enumerate(clevs):
                if c == the_contour:
                    line_w[i] = 1.5 # Make the contour we're using for the ellipse calc fatter
                else:
                    line_w[i] = 0.5
            #print(clevs)
            
            ax.contour(x,y,height,clevs,transform=ccrs.PlateCarree(),extend='both',colors='black',linewidths=line_w)
            #ax.contour(x2,y2,tdata2, lev = [tmean], transform=ccrs.PlateCarree(),extend='both',colors='blue',linewidths=line_w)
            
            try:
                lev_contour_ind = np.where(np.array(cs_temp[-1].levels)==the_contour)[0][0]
                isoline_list = cs_temp[-1].allsegs[lev_contour_ind]
                #print("in try")
                #print("Number of ellipses: ",len(isoline_list))
            except:
                #print("Contours not found for",the_contour,"meter level at",plot_lev,"hPa.")
                isoline_list = []
            
            if len(isoline_list) > 0:
                lev_contour_ind = np.where(np.array(cs_temp[-1].levels)==the_contour)[0][0]
                isoline_list = cs_temp[-1].allsegs[lev_contour_ind]
                #print(isoline_list)
                #print("in elipse ...")
                #print("Number of ellipses: ",len(isoline_list))
            else:
                #print("Contours not found for",the_contour,"meter level at",plot_lev,"hPa.")
                isoline_list = []
            
            ##number of contours
            isocount = 0
            small = 0
            #empty lists for upcoming metric calculations
            ratio1 = []
            uvalues1 = []
            cenlat1 = []
            cenlon1 = []
            size1 = []
            ephi1 = []
            
            for isoline in isoline_list:
                ##next few lines will check if the contour crossing the prime meridian is one continuous contour or two separate ones
                #print('isoline',isoline)
                #[iso_lon,iso_lat] = mm(isoline[:,0],isoline[:,1],inverse=True)
                [iso_lon,iso_lat] = [isoline[:,0],isoline[:,1]]
                if len(iso_lon)<15:
                    #print("-----Not analyzing ellipse with",len(iso_lon),"points, continuing...")  # Check for size!
                    small = 1
                    continue
                # Checking to see if contours are closed (0) or need to be joined (1)  Checking before convert lat/lon to radians
                lon_diff = abs(iso_lon[0] - iso_lon[len(iso_lon)-1])
                lat_diff = abs(iso_lat[0] - iso_lat[len(iso_lat)-1])
                join = 0
                if lon_diff > 1 or lat_diff > 1:
                    join = 1
                    #print("Diffs lat/lon: ",lat_diff,lon_diff)
                iso_lon = np.deg2rad(iso_lon)
                iso_lat = np.deg2rad(iso_lat)
                
                ex = np.array((np.cos(iso_lon)*np.cos(iso_lat))/(1+np.sin(iso_lat)))
                ey = np.array((np.sin(iso_lon)*np.cos(iso_lat))/(1+np.sin(iso_lat)))
                ## does this contour include the pole?
                overpole = point_inside_polygon(0,0,ex,ey)  #returns true if poly includes the pole, false if not 
                #print("It is",overpole, "that the contour includes the pole")
                if not overpole and len(isoline_list) > 1 and join > 0 and small < 1:
                    if isocount > 0:
                        ex = np.append(ex,ex2)
                        ey = np.append(ey,ey2)
                        #print("Point didn't include pole - add to these to previous set")
                    else:
                        ex2 = ex
                        ey2 = ey
                        #print("Point didn't include pole - keeping these to add to next set")
                        isocount = 1
                        continue
                ##begin actual calculation        
                #print("Running ellipse diagnostic now")
                exx,eyy,eaax,ebax,ecenterx,ecentery,ephi = fitEllipseContour(ex,ey)
                
                ## Convert back to lat/lon
                elons = np.where(exx<0,np.where(eyy>0,np.arctan(eyy/exx)+math.pi,np.arctan(eyy/exx)-math.pi),np.arctan(eyy/exx))
                yysinxxlon = eyy/np.sin(elons)
                elats = -2*(np.arctan(yysinxxlon) - (math.pi/4.0))
                elats = np.rad2deg(elats)
                elons = np.rad2deg(elons)
                
                ## Still not really sure what this is for...
                for g in range(1,len(elats)):
                    if abs(elats[g]-elats[g-1]) > 1.5:
                        elats[g] = elats[g-1]
                
                ## Center points back to lat/lon
                cenlon = np.where(ecenterx<0,np.where(ecentery>0,np.arctan(ecentery/ecenterx)+math.pi,np.arctan(ecentery/ecenterx)-math.pi),np.arctan(ecentery/ecenterx))
                ysinlon = ecentery/np.sin(cenlon)
                cenlat = np.rad2deg(-2 * (np.arctan(ysinlon) - (math.pi/4.0)))
                cenlon = np.rad2deg(cenlon)
                #print("Center of ellipse:",cenlat,"N",cenlon,"E")
                
                ## Calculate endpoints of the axes of the vortex, convert to lat/lon
                xa = eaax * np.cos(ephi)
                ya = eaax * np.sin(ephi)
                xb = ebax * np.sin(ephi)
                yb = ebax * np.cos(ephi)
                endx = np.array([ecenterx+xa,ecenterx-xa,ecenterx+xb,ecenterx-xb])
                endy = np.array([ecentery+ya,ecentery-ya,ecentery-yb,ecentery+yb])
                endlon = np.where(endx < 0,np.where(endy>0,np.arctan(endy/endx)+math.pi,np.arctan(endy/endx)-math.pi),np.arctan(endy/endx))
                ysinelon = endy/np.sin(endlon)
                endlat = np.rad2deg(-2 *(np.arctan(ysinelon) - (math.pi/4)))
                endlon = np.rad2deg(endlon)
                
                ## Calc great circle distances; (still necessary?) 
                a1gc = great_circle((endlat[0],endlon[0]),(cenlat,cenlon)).km
                a2gc = great_circle((cenlat,cenlon),(cenlat,endlat[1])).km
                b1gc = great_circle((endlat[2],endlon[2]),(cenlat,cenlon)).km
                b2gc = great_circle((cenlat,cenlon),(endlat[3],endlon[3])).km
                
                ephi = np.rad2deg(ephi)
                if a1gc < b1gc:
                    ephi -= 90
                    #print("Emetrics phi:",ephi)
                if ephi < -45:
                    ephi += 180
                ratio = a1gc/b1gc
                if ratio < 1.0:
                    ratio = 1.0/ratio
                size = math.pi*a1gc*b1gc
                #estr = times[t].strftime("%H %Y%m%d%H")+" "+f'{cenlon:.3f}'+" "+f'{cenlat:.3f}'+" "+f'{a1gc:.3f}'+" "+f'{b1gc:.3f}'+" "+f'{ephi:.3f}'+" "+f'{u.values:.3f}'+" "+f'{temp.values:.3f}'
                
                exy=np.array(list(zip(elons,elats)))
                plt.plot(elons,elats,color='red',transform=ccrs.Geodetic(),linewidth=mem_lw)
                plt.plot(cenlon,cenlat,markersize=4,marker='o',color='red',transform=ccrs.PlateCarree())
                #ax.add_patch(Polygon(exy,closed=True,color='magenta',fill=False,lw=mem_lw,transform=ccrs.PlateCarree()))
                
                ##append values to the empty lists for this day/iteration
                ratio1.append(ratio)
                uvalues1.append(u.values)
                cenlat1.append(cenlat)
                cenlon1.append(cenlon)
                size1.append(size)
                ephi1.append(ephi)
    
            if len(ratio1) >= 1:
                rat.append(np.mean(ratio1))
                w.append(np.mean(uvalues1))
                cenlt.append(np.median(cenlat1))
                cenln.append(np.median(cenlon1))
                sz.append(np.mean(size1))
                ep.append(np.mean(ephi1))
                num.append(len(ratio1))
            
            if len(ratio1) == 0:
                rat.append(np.nan)
                w.append(np.nan)
                cenlt.append(np.nan)
                cenln.append(np.nan)
                sz.append(np.nan)
                ep.append(np.nan)
                num.append(0)
                
        rat10[mm]= rat[:]
        wind10[mm]= w[:]
        cenlat10[mm]= cenlt[:]
        cenlon10[mm]= cenln[:]
        size10[mm]= sz[:]
        ep10[mm]= ep[:]
        num10[mm]= num[:]

    return rat10, wind10, cenlat10, cenlon10, size10, ep10, num10; 
                
        

    