##################################################
# 
# Basic test of Stan models for spaceless data.
# To test the model fits, we generate data from a simple
# generative model and analyze it with the model and with
# simpler methods.
# 
##################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotnine as pn
from cmdstanpy import CmdStanModel
import seaborn as sns
import os
import arviz as az
import scipy.stats as stats
from sklearn.metrics import roc_curve
from scipy.stats import rankdata
from functions_analysis import *


###############
# 1) LOAD THE DATA
###############

dataFile = '../data/1_excitatory_spaceless_longer.csv'
# Create a DataFrame with the data
expData = pd.read_csv(dataFile)
expData = expData[expData['Target'] != 1]

renameDict = {'NeuronID': 'Neuron', 'Activation': 'Response', 'Stimulation': 'Stim'}
expData.rename(columns=renameDict, inplace=True)

###############
# 2) COMPUTE SUMMARY STATISTICS
###############

#####################################
### STATISTICS FOR EACH CONDITION
#####################################

### NEURON-WISE
# Group the dataframe
groupVars = ['Neuron', 'Target', 'DirectConn', 'SamePop', 'Stim']
df = expData.groupby(groupVars)
# Compute the mean of activation
means_df = df['Response'].mean()
means_df.rename('mean', inplace=True)
sd_df = df['Response'].std()
sd_df.rename('sd', inplace=True)
sem_df = df['Response'].sem()
sem_df.rename('sem', inplace=True)
# Merge the two dataframes
nrnStats = pd.merge(means_df, sd_df, on=groupVars)
nrnStats = pd.merge(nrnStats, sem_df, on=groupVars)
nrnStats.reset_index(inplace=True)
# Make categorical variable of population, popID
nrnStats['popID'] = nrnStats['DirectConn'] + nrnStats['SamePop']
nrnStats['popID'] = pd.Categorical(nrnStats['popID'])
popNames = {0: 'DiffPop', 1: 'SamePop', 2: 'Connected'}
nrnStats['popID'] = nrnStats['popID'].cat.rename_categories(popNames)

### POPULATION-WISE
# Group the dataframe
groupVars = ['Target', 'DirectConn', 'SamePop', 'Stim']
df = nrnStats.groupby(groupVars)
# Compute the mean of activation
means_df = df['mean'].mean()
means_df.rename('mean', inplace=True)
sd_df = df['mean'].std()
sd_df.rename('sd', inplace=True)
sem_df = df['mean'].sem()
sem_df.rename('sem', inplace=True)
# Merge the dataframes
popStats = pd.merge(means_df, sd_df, on=groupVars)
popStats = pd.merge(popStats, sem_df, on=groupVars)
popStats.reset_index(inplace=True)
# Make categorical variable of population, popID
popStats['popID'] = popStats['DirectConn'] + popStats['SamePop']
popStats['popID'] = pd.Categorical(popStats['popID'])
popNames = {0: 'DiffPop', 1: 'SamePop', 2: 'Connected'}
popStats['popID'] = popStats['popID'].cat.rename_categories(popNames)


# PLOT THE STATISTICS FOR EACH CONDITION
(pn.ggplot(nrnStats, pn.aes(x='Stim', y='mean', color='popID',
                           group='Neuron')) +
  # Jitter the points
  pn.geom_jitter(width=0.1, height=0.1, alpha=0.2) +
  # Add population means
  #pn.geom_line(alpha=0.2)
  pn.geom_point(data=popStats, mapping=pn.aes(x='Stim', y='mean',
                                              color='popID',
                                              group='popID'), size=4) +
  pn.geom_line(data=popStats, mapping=pn.aes(x='Stim', y='mean',
                                              color='popID',
                                              group='popID'),
               size=1, linetype='dashed') +
  # Add error bars
  pn.geom_errorbar(data=popStats, mapping=pn.aes(ymin='mean - sem',
                                                 ymax='mean + sem',
                                                 color='popID',
                                                 group='popID'),
                   size=1, width=0.1) +
  # Set y-axis limit to 0-6
  pn.theme_bw() 
)


#####################################
### STATISTICS OF THE DIFFERENCE
#####################################

### NEURON-WISE
# Unstack the dataframe by Stim
nrnStatsDiff = nrnStats.set_index(['Neuron', 'popID', 'Stim'])
nrnStatsDiff.drop(['Target', 'DirectConn', 'SamePop'], axis=1, inplace=True)
nrnStatsDiff = nrnStatsDiff.unstack('Stim')
# Compute the difference between the two conditions
nrnStatsDiff['diff'] = nrnStatsDiff['mean'][True] - nrnStatsDiff['mean'][False]
nrnStatsDiff['diff_sem'] = nrnStatsDiff['sem'][True] + nrnStatsDiff['sem'][False]
nrnStatsDiff.reset_index(inplace=True)

# Compute the average SEM for each population and the mean
# difference between conditions, all in one dataframe
popStatsDiff = nrnStatsDiff.groupby(['popID'])
meanDiff = popStatsDiff['diff'].mean()
meanDiff.rename('pop_diff', inplace=True)
meanDiff = meanDiff.reset_index()
semDiff = popStatsDiff['diff_sem'].mean()
semDiff.rename('pop_sem', inplace=True)
semDiff = semDiff.reset_index()
## DF with effects of populations
popStatsDiff = pd.merge(meanDiff, semDiff, on='popID')

# PLOT THE DIFFERENCE BETWEEN CONDITIONS
(pn.ggplot(nrnStatsDiff, pn.aes(x='diff', fill='popID')) +
pn.geom_density(alpha=0.3) +
pn.geom_vline(xintercept=0, linetype='dashed') +
 # Plot the mean 2*sem for each population at the mean
 # population point, as a horizontal bar
pn.geom_errorbarh(data=popStatsDiff,
                 mapping=pn.aes(x='pop_diff', y=[2, 2.25, 2.5],
                                xmin='pop_diff - pop_sem',
                                xmax='pop_diff + pop_sem',
                                color='popID'),
                    size=1) +
pn.xlab('Difference in activation between conditions') +
pn.ylab('Density') +
pn.xlim(-0.3, 1) +
pn.theme_bw()
)

# PLOT THE DENSITY OF NEURON-WISE ERROR
(pn.ggplot(nrnStatsDiff, pn.aes(x='diff_sem', fill='popID')) +
pn.geom_density(alpha=0.4) +
pn.xlab('SEM of activation difference') +
pn.ylab('Density') +
#pn.xlim(0.08, 0.095) +
pn.theme_bw()
)


###############
# 3) DO SIMPLE ANALYSES OF THE DATA, WITH SUBSAMPLING
###############

# We compute neuron-wise and population-wise parameters
# as a function of the number of used trials, to see
# how many trials are needed to get stable estimates

# COMPUTE NEURON-WISE SIGNIFICANCE
nTrials = [20, 30, 40, 50, 75, 100, 150, 200, 500, 700, 1000]
neuronIDs = expData['Neuron'].unique()
# Extract columns from nrnStats
templateDf = nrnStats[['Neuron', 'popID']].drop_duplicates()
dfList = []
for n in range(len(nTrials)):
    expDataSubs = expData[expData['Trial'] < nTrials[n]]
    pVals, meanDiff = neuron_specific_analysis_2_cond(expDataSubs)
    pValsHolm = holm_correction(pVals)
    pValsHoch = hochberg_correction(pVals)
    # Store the results in a dataframe
    dfList.append(templateDf.copy())
    dfList[-1]['pVal'] = pVals
    dfList[-1]['pValHolm'] = pValsHolm
    dfList[-1]['pValHoch'] = pValsHoch
    dfList[-1]['diff'] = meanDiff
    dfList[-1]['nTrials'] = nTrials[n]


# Concatenate the dataframes
nrnStatsSubs = pd.concat(dfList, ignore_index=True)

# COMPUTE POPULATION-WISE PROBABILITY OF EFFECT

# Get proportion of p<0.05 for each number of trials
nrnStatsSubs['pSignif'] = nrnStatsSubs['pVal'] < 0.05
nrnStatsSubs['pSignifHolm'] = nrnStatsSubs['pValHolm'] < 0.05
nrnStatsSubs['pSignifHoch'] = nrnStatsSubs['pValHoch'] < 0.05

cols = ['pSignif', 'pSignifHolm', 'pSignifHoch']
correction = ['None', 'Holm', 'Hoch']
dfList = []
for c in range(len(cols)):
    signifDf = nrnStatsSubs.groupby(['nTrials', 'popID'])[cols[c]].mean().reset_index()
    signifAllDf = nrnStatsSubs.groupby(['nTrials'])[cols[c]].mean().reset_index()
    signifAllDf['popID'] = 'All'
    dfList.append(pd.concat([signifDf, signifAllDf], axis=0))
    dfList[c]['correction'] = correction[c]
    dfList[c].rename(columns={cols[c]: 'propSignif'}, inplace=True)

# Concatenate the dataframes
popStats = pd.concat(dfList, ignore_index=True)

# Count the number of neurons of each population
neuronN = nrnStats[['Neuron', 'popID']].drop_duplicates().groupby('popID').count()
# add row with the total number of neurons
neuronN.loc['All'] = neuronN.sum()

propConnected = neuronN.loc['Connected', 'Neuron'] / neuronN.loc['All', 'Neuron']

# PLOT THE PROPORTION OF SIGNIFICANT NEURONS
(pn.ggplot(popStats, pn.aes(x='nTrials', y='propSignif', color='popID')) +
pn.geom_line() +
pn.geom_point() +
pn.facet_wrap('~correction') +
pn.geom_hline(yintercept=propConnected, linetype='dashed') +
pn.xlab('Number of trials') +
pn.ylab('Proportion of significant neurons') +
pn.theme_bw()
)


# COMPUTE THE ESTIMATED POPULATION-WISE EFFECT SIZE

cols = ['pSignif', 'pSignifHolm', 'pSignifHoch']
correction = ['None', 'Holm', 'Hoch']
dfList = []
for c in range(len(cols)):
    effectDf = nrnStatsSubs.copy()[nrnStatsSubs[cols[c]]]
    effectDf = effectDf.groupby(['nTrials']).mean('diff').reset_index()
    effectDf['correction'] = correction[c]
    dfList.append(effectDf[['nTrials', 'diff', 'correction']])

# Concatenate the dataframes
popEffect = pd.concat(dfList, ignore_index=True)
popEffect.rename(columns={0: 'effectSize'}, inplace=True)


# PLOT THE ESTIMATED EFFECT
connectedEffect = popStatsDiff[popStatsDiff['popID']=='Connected']['pop_diff'].values[0]
(pn.ggplot(popEffect, pn.aes(x='nTrials', y='diff', color='correction')) +
pn.geom_line() +
pn.geom_point() +
pn.geom_hline(yintercept=connectedEffect, linetype='dashed') +
pn.xlab('Number of trials') +
pn.ylab('Effect size estimate') +
pn.theme_bw()
)


###############
# 3) FIT A MIXTURE HIERARCHICAL MODEL
###############

nTrials = 50

subsampleData = expData[expData['Trial'] < nTrials]

# Turn Stim and Response into matrices where each row is a neuron
# and each column is a trial (for each condition)
stimMatrix = subsampleData.pivot(index='Neuron', columns=['Trial', 'Stim'],
                            values='Stim').values
responseMatrix = subsampleData.pivot(index='Neuron', columns=['Trial', 'Stim'],
                                values='Response').values

# Modify response
#responseMatrix = responseMatrix + stimMatrix * 0.2

nNeurons = 100
stimMatrix = stimMatrix[:nNeurons, :]
responseMatrix = responseMatrix[:nNeurons, :]

# Initialization


# Prepare the data
stan_data = {
    'nNeurons': nNeurons,
    'nObsPerNeuron': nTrials * 2,
    'Stimulation': stimMatrix.astype(int),
    'Response': responseMatrix
}

#stan_file = './stan_models/2_spaceless_mixed.stan'
stan_file = './stan_models/2_spaceless_mixed_prior2.stan'
#stan_file = './stan_models/2_spaceless_mixed_poisson.stan'
model = CmdStanModel(stan_file=stan_file)


init_dict = {
    'effectPop': [0.4, 0.4],
    'effectSigma': [0.2, 0.2],
    'pMix': [0.08, 0.06],
    'interceptPop': [12, 12],
    'interceptSigma': [1, 1],
    'residualSigma': [1, 1],
    'effectNrn': [np.ones(nNeurons) * 0.4, np.ones(nNeurons) * 0.4],
    'interceptNrn': [np.ones(nNeurons) * 12, np.ones(nNeurons) * 12]
}

# Fit the model. SAMPLING
import time
start = time.time()
posterior2 = model.sample(data=stan_data, chains=2, iter_warmup=1500,
                         iter_sampling=1500, adapt_delta=0.99,
                         inits=init_dict)
end = time.time()
summary = az.summary(posterior2, hdi_prob=0.68)

print(summary.loc[summary.index.str.startswith('effectPop'), 'hdi_84%'])
print(summary.loc[summary.index.str.startswith('interceptPop'), 'hdi_84%'])

az.plot_trace(posterior2, var_names=['effectPop', 'interceptPop'])


# Get naive estimates of pMix for this number of trials and neurons
nrnTypes = nrnStatsSubs[nrnStatsSubs['nTrials']==50].copy()
nrnTypes = nrnTypes[nrnTypes['Neuron']<nNeurons+2]

# True proportion of connected neurons
pMixReal = np.mean(nrnTypes['popID']=='Connected')
pMixNaive = np.mean(nrnTypes['pSignif'])
pMixHoch = np.mean(nrnTypes['pSignifHoch'])
pMixHolm = np.mean(nrnTypes['pSignifHolm'])

pMixModel = summary.loc[summary.index.str.startswith('pMix'), 'mean'].values

pNrn = summary.loc[summary.index.str.startswith('pNrn'), 'mean'].values
nrnTypes['pNrn'] = pNrn

# Plot pNrn by popID
pPlot = 'pValHoch'
#pPlot = 'pNrn'
(pn.ggplot(nrnTypes, pn.aes(x='popID', y=pPlot)) +
 pn.geom_boxplot() +
  pn.theme_bw()
)




