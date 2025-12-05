# About

**UFS_EllipseAnalysis** is a repository containing code relevant to analyzing the use of best-fit stratospheric polar vortex ellipse geometries/metrics within UFS data. This method, presented in a manuscript by [*Fernandez et al. 2025*](https://github.com/emf98/SPVMD), provides complementary metrics for determining stratospheric polar vortex strength and SSW variability to those established in [*Seviour et al. 2013*](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/grl.50927).


## Contents

* `fitEllipse3_new.py` contains code for mathematical definition statements related to the full geometric calculation of the ellipse metrics.

* `EllipseDef_ERA5.py` contains a definition statement code supporting ellipse metric calculations using the best-fit method for ERA-5 datasets.
    * This code can be used at most pressure levels and can contour other features of interest through a user-defined geopotential height value. The manuscript for this repository supports the use of a climatologically representative contour in defining the vortex edge.

* `UFS_EllipseDef.py` contains a definition statement code supporting ellipse metric calculations for UFS data.

    * `UFS_Bulk` and `UFSEllipse_Test` utilize this definition statement for pulling data. 

Please check `EllipseDef_ERA5.py` and `UFS_EllipseDef.py` to ensure your correctly saved ERA-5 or UFS file locations are used. 

Additionally, the code uses an older version (**Python 3**) of cartopy for plotting circumpolar views of the vortex. If the plotting lines return issues, and you do not wish to modify the cartopy distinction, comment them out or remove them. It will not affect the running of the code. 

* `MBE.py` contains functions to calculate and plot forecast biases and errors.

    * There are three MBE *.ipynbs* here that use the forecast bias calculation to plot a box and whisker/barplots for the desired feature (*TestwSig*) or for plotting temperatures/ellipse metrics over SSWs (*SSW_Temps, SSW_Test*).
    * The test images from these are in the main `images` subfolder.

* `Save_AreaWeighted` and `Save_AreaComposite` are used to modify GPH and Temp data to be used for input/output data, and in creating cross sections of RF results. The GPH files referenced are NOT in this repo; they were downloaded from ECMWF in a separate file location. They were too large to include here.

The `testing_RF` contains all files relevant to testing the Random Forest model using UFS prototypes as testing data. 

## Data

Data for this analysis are contained within the `UFS_metrics` and `era5` subfolders. 


## References

Any questions regarding the code files may be directed to Elena Fernandez (emfernandez@albany.edu/elenamf98@gmail.com). 