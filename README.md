# Ensemble model output statistics and ensemble copula coupling for post-processing of weather regime forecasts
This repository provides python code for the post-processing of probabilistic sub-seasonal weather regime forecasts. The code has been written for the paper

!!!placeholder arxiv !!!

# Data
The data which is used for the above mentioned paper is not publically available, please contact fabian.mockert@kit.edu in case you are interested in the raw data.

# Code
- Read in a netcdf file
  - dimensions/coordinates: IC (initial condition dates), WR (weather regimes), leadtime (lead time of forecast in days)
  - variables: 01-10, CF, ERA, EM (10+1 ensemble members, ERA as truth and Ensemble mean)
- The dataset is split into training and testing data. 
- Iterate over all initial dates in the testing period
  - select all data with training dates in a 31d running window around the day of year of the current initial date
  - train/optimise parameters for the ensemble model output statistics (EMOS-G)
  - compute the EMOS-G for the test data
  - compute the ensemble copula coupling (ECC-Q) from the EMOS-G data
- Compute skill scores for the EMOS-G and ECC-Q data (continuous ranked probability score, energy score, variogram score)

 
