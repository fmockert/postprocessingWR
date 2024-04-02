"""
Post-processing of uncalibrated weather regime index (IWR) with ensemble model output statistics (EMOS)/ensemble copula coupling (ECC)
Written by Fabian Mockert, 
Karlsruhe Institute of Technology (KIT)
Institute of Meteorology and Climate Research Troposphere Research (IMKTRO)
contact: fabian.mockert@kit.edu
"""

import os
import pandas as pd
import numpy as np
import xarray as xr
import properscoring as ps #https://github.com/properscoring/properscoring
from functools import partial
from scipy.stats import norm
from scipy.optimize import minimize

""" List of regime abbreviations """
regimes = ['AT', 'ZO', 'ScTr', 'AR', 'EuBL', 'ScBL', 'GL']

""" Path and variable defintion """
train_end = "INCLUDE_YOUR_TRAINING_END" #in paper: (pd.Timestamp('2015-05-31'))

members_train = [str(i).zfill(2) for i in np.arange(1,10+1)]+['CF']#, 'EM']
members_test = [str(i).zfill(2) for i in np.arange(1,10+1)]+['CF']#, 'EM']

""" Create the path where a file is saved if it does not exist yet """
def create_path(path):
  directory = os.path.dirname(path)
  if not os.path.exists(directory): os.makedirs(directory)

""" Generate CRPS values for each combination of IC, WR, and leadtime. Applies ps.crps_ensemble"""
def crps_wrapper_ensemble(y_true, y_ensemble):
  def crps_ensemble_partial(y_true, y_ensemble):
    return ps.crps_ensemble(y_true, y_ensemble)
  
  # apply crps_ensemble on the dataset
  crps_ds = xr.apply_ufunc(
    partial(crps_ensemble_partial),
    y_true, y_ensemble.to_array('member').transpose('IC', 'WR', 'leadtime', 'member'),
    input_core_dims=[['IC', 'WR', 'leadtime'], ['IC', 'WR', 'leadtime', 'member']],
    output_core_dims=[['IC', 'WR', 'leadtime']],
    vectorize=True,
    dask='parallelized',
    output_dtypes=[float]
    ).to_dataset(name='crps_ensemble')
  return crps_ds['crps_ensemble']

""" gaussian crps """
def crps_wrapper_gauss(y_true, mu, sig):
  def crps_gaussian_partial(y_true, mu, sig):
    return ps.crps_gaussian(y_true, mu, sig)
  
  # apply crps_guassian on the dataset
  crps_ds = xr.apply_ufunc(
    partial(crps_gaussian_partial),
    y_true, mu, sig,
    input_core_dims=[['IC', 'WR', 'leadtime'], ['IC', 'WR', 'leadtime'], ['IC', 'WR', 'leadtime']],
    output_core_dims=[['IC', 'WR', 'leadtime']],
    vectorize=True,
    dask='parallelized',
    output_dtypes=[float]
    ).to_dataset(name='crps_gaussian')
  return crps_ds['crps_gaussian']

""" Optimise parameters for EMOS-G on training data and returen them """
def pp_training(noPP_data, members):
  # parameter optimisation on training data
  def pp_parameter(ensmean_train, ensvar_train, obs_train):
    def optimize(x0, *args):
      ensmean_train, ensvar_train, obs_train = args
      a_0, a_1, b_0, b_1 = x0
      mu = a_0 + a_1 * ensmean_train
      sigsq = b_0 + b_1 * ensvar_train
      if any(sigsq<0): return(999999)
      else: 
        sig = np.sqrt(sigsq)
        return np.mean(ps.crps_gaussian(x=obs_train, mu=mu, sig=sig, grad=False)) #crps_gaussian or crps_ensemble?
  
    res = minimize(fun=optimize, 
                 x0=[1,1,1,1],#Initialisation weights of parameters
                 args=(ensmean_train, ensvar_train, obs_train),
                 method='BFGS'
                )
    return res.x #res.x[0], res.x[1], res.x[2], res.x[3]
    # Parameter optimisation on training data
  noPP_ensmean = noPP_data[members].to_array(dim='members').mean(dim='members') #data including test period, needed as given back from function pp_parameter
  noPP_ensvar = noPP_data[members].to_array(dim='members').var(dim='members') #data including test period
  
  # Apply the function to the input data using apply_ufunc
  #param1, param2, param3, param4 = xr.apply_ufunc(
  optimised_parameters = xr.apply_ufunc(
      pp_parameter, noPP_ensmean.where(noPP_ensmean.IC.isin(date_train.values),drop=True), 
      noPP_ensvar.where(noPP_ensvar.IC.isin(date_train.values),drop=True), noPP_data['ERA'].where(noPP_data.IC.isin(date_train.values),drop=True),
      input_core_dims=[['IC'], ['IC'], ['IC']],
      #output_core_dims=[[], [], [], []],
      output_core_dims=[['parameters']],
      #output_sizes={'WR': 7, 'leadtime': 37},
      vectorize=True,
      dask='parallelized',
      output_dtypes=[np.float64]#, np.float64, np.float64, np.float64],
  ).to_dataset(dim='parameters')
  return optimised_parameters, noPP_ensmean, noPP_ensvar

""" emosG optimisation on test data """
def pp_test_data(optimised_parameters, ensmean, ensvar):
  loc_test = optimised_parameters[0].broadcast_like(ensmean) + optimised_parameters[1].broadcast_like(ensmean) * ensmean
  ssq_test = optimised_parameters[2].broadcast_like(ensvar) + optimised_parameters[3].broadcast_like(ensvar) * ensvar
  #if (ssq_test < 0).any(): print("negative scale, taking absolute value")
  std_test = np.sqrt(abs(ssq_test))
  return(loc_test, std_test)

""" Compute eccQ from emosG """
def compute_eccQ_schaake(forecast, emosG, members):
  def eccQ_schaake_function(forecast, loc, std):
    def get_rank(lst):
      sorted_lst = sorted(lst)
      rank_dict = {}
      for i, val in enumerate(sorted_lst):
        if val not in rank_dict:
          rank_dict[val] = i + 1
      ranks = [rank_dict[val] for val in lst]
      return ranks
    
    # Create normal distribution for IWR with mu and sig
    dist = norm(loc=loc, scale=std) #input
    # Get the 11 member quantiles
    quantiles = np.linspace(0,1,len(members)+2)[1:-1]
    quantile_values = dist.ppf(quantiles)
    noPP_members = [forecast[m] for m in np.arange(0,len(members),1)]#[r1, r2] #[noPP_members[m] for m in np.arange(0,11,1)]

    ranks = get_rank(noPP_members)

    quantile_values_sorted = quantile_values[[r-1 for r in ranks]] #output ECC-Q   
    #quantile_values_sorted = quantile_values[np.argsort(noPP_members)] #output ECC-Q #--------------------------------------------------  np.argsort in this way is most likely wrong! Check plot_statistics_presentation
    return quantile_values_sorted
  
  # Apply the function to the datasets using xr.apply_ufunc()
  dependency = xr.apply_ufunc(
      eccQ_schaake_function, forecast.to_array('member').transpose('IC', 'WR', 'leadtime', 'member'), emosG['loc'], emosG['std'],
      input_core_dims=[['member'], [], []],
      output_core_dims=[['member']],
      dask='parallelized',
      vectorize=True,
      output_dtypes=[float]*len(members)
  ).to_dataset(dim='member')
  return dependency

""" Energy Score """ 
def energy_score(ERA, forecast):
  def ES_function(ERA, forecast):
    es_1 = np.sum(np.linalg.norm(forecast - ERA[:, np.newaxis], axis=0))
    es_2 = np.sum(np.sum(np.linalg.norm(forecast[:, :, np.newaxis] - forecast[:, np.newaxis, :], axis=0), axis=0))
    es = 1/forecast.shape[1]*es_1 - 0.5 /(forecast.shape[1]*forecast.shape[1])*es_2
    return es
  
  ES = xr.apply_ufunc(
      ES_function, ERA, forecast,
      input_core_dims=[['WR'],['WR', 'members']],
      output_core_dims=[[]],
      dask='parallelized',
      vectorize=True,
      output_dtypes=[float]
  )
  return ES

""" Variogram Score """
def variogram_score(ERA, forecast, p=0.5):
  def VS_function(ERA, forecast):
    # Define the order of the variogram
    p = 0.5
    w = 1
    # Compute the sum over i and j
    sum_ij = 0
    for i in range(len(ERA)):
      for j in range(len(ERA)):
        if i != j:
          sum_m = 0
          for m in range(forecast.shape[1]):
            sum_m += (np.abs(forecast[i, m] - forecast[j, m]) ** p)
          sum_ij += w*(np.abs(ERA[i] - ERA[j]) ** p - (1 / forecast.shape[1]) * sum_m) ** 2
    
    # Compute the final result
    return sum_ij #/ (len(ERA) * (len(ERA) - 1))
  
  VS = xr.apply_ufunc(
      VS_function, ERA, forecast,
      input_core_dims=[['WR'],['WR', 'members']],
      output_core_dims=[[]],
      dask='parallelized',
      vectorize=True,
      output_dtypes=[float]
  )
  return VS

# ^ Definitions
#----------------------------------------------------------------------
# v Actual Program

""" Load raw unprocessed weather regime index forecasts """
iwr_reforecasts = xr.open_dataset("FILEPATH_TO_REFORECAST_DATA") 
"""File format: 
coordinates/dimensions: IC (initial condition dates), WR (weather regimes), leadtime (lead time of forecast in days)
variables: 01-10, CF, ERA, EM (10+1 ensemble members, ERA as truth and Ensemble mean)
"""

""" Load all reforecasts for training EMOS/ECC """
noPP_re = iwr_reforecasts.sel(IC=iwr_reforecasts['IC']<=train_end, drop=True)
# Remove all IC with -999 values!
mask_re = noPP_re != -999
noPP_re = noPP_re.where(mask_re, drop=True)

""" Loading all reforecast for performing EMOS/ECC """
noPP_op = iwr_reforecasts.sel(IC=iwr_reforecasts['IC']>train_end, drop=True)
# Remove all IC with -999 values!
mask_op = noPP_op != -999
noPP_op = noPP_op.where(mask_op, drop=True)

""" Iterate through all testing IC, save ECC for each IC individual and merge later together """
IC_dates=[pd.to_datetime(i) for i in noPP_op.IC.values]
emosG_full = None
eccQ_full = None
for IC in IC_dates:
  noPP_test = noPP_op.sel(IC=[IC])
  if (noPP_test[members_test].to_array(dim='members') == -999.0).any().item():
    print(f'In {IC} there are -999 values, therefore skip.')
    continue
  print(f'Computing {IC}')
  date_train = pd.Series([pd.to_datetime(i) for i in (noPP_re.IC.values)])

  w31 = [] # dates in reforecasts with running window 31days in training period
  for year in np.arange(date_train.iloc[0].year,date_train.iloc[-1].year+1): # iterate over all years in training data
    w31_range = pd.date_range(IC-pd.Timedelta(15, 'd'), IC+pd.Timedelta(15, 'd'))
    for w in w31_range: # create the running window in each year
      if ((year%4!=0) & (w.month==2) & (w.day==29)): # exclude leap days
        continue  
      elif pd.Timestamp(year, w.month, w.day) in date_train.values:
        w31+=[pd.Timestamp(year, w.month, w.day)]

  noPP_train = noPP_re.sel(IC=w31)

  # Initiate emosG xarray dataset
  emosG = xr.DataArray(np.nan, dims=['WR', 'leadtime', 'vars'], coords={'WR':regimes, 'leadtime':np.arange(-10,36+0.5,1), 'vars':['loc', 'std']}).to_dataset(dim='vars')
  
  """ Train EMOS-G: find the optimised parameters on the training data set """
  optimised_parameters, noPP_ensmean, noPP_ensvar = pp_training(noPP_data=noPP_train, members=members_train)
  
  """ Compute EMOS-G: Post-process test data with parameters from optimisation """
  mu, sig = pp_test_data(optimised_parameters, ensmean=noPP_test.sel(IC=[IC])[members_test].to_array(dim='members').mean(dim='members'), ensvar=noPP_test.sel(IC=[IC])[members_test].to_array(dim='members').var(dim='members'))
  emosG['loc'] = mu; emosG['std'] = sig
  emosG.to_netcdf("FILEPATH_FOR_DAILY_EMOS_OUTPUT")

  """ Compute (EMOS-Q)/ECC-Q """
  leadtimes=np.arange(-10,36+0.25, 1)
  eccQ = compute_eccQ_schaake(forecast=noPP_test.sel(IC=[IC], leadtime=leadtimes, drop=True)[members_test], 
                              emosG=emosG.sel(leadtime=leadtimes, drop=True), members=members_test)
  eccQ.to_netcdf("FILEPATH_FOR_DAILY_ECC_OUTPUT")

  if emosG_full is None: emosG_full=emosG
  else: emosG_full = xr.concat([emosG_full, emosG], dim='IC')
  if eccQ_full is None: eccQ_full=eccQ
  else: eccQ_full = xr.concat([eccQ_full, eccQ], dim='IC')

""" Save ERA member also in emos/ecc netcdfs """
emosG_full = emosG_full.assign(ERA=noPP_op['ERA'])
eccQ_full = eccQ_full.assign(ERA=noPP_op['ERA'])

""" Compute Scores (CRPS/ES/VS) """
emosG_full = emosG_full.assign(crps=crps_wrapper_gauss(y_true=emosG_full['ERA'], mu=emosG_full['loc'], sig=emosG_full['std']))
eccQ_full = eccQ_full.assign(crps=crps_wrapper_ensemble(y_true=eccQ_full['ERA'], y_ensemble=eccQ_full[members_test]))
eccQ_full = eccQ_full.assign(es=energy_score(ERA=eccQ_full['ERA'], forecast=eccQ_full[members_test].to_array(dim='members')))
eccQ_full = eccQ_full.assign(vs=variogram_score(ERA=eccQ_full['ERA'], forecast=eccQ_full[members_test].to_array(dim='members')))

""" Save netcdf files """
emosG_full.to_netcdf("FILEPATH_FOR_EMOS_OUTPUT")
eccQ_full.to_netcdf("FILEPATH_FOR_ECC_OUTPUT")