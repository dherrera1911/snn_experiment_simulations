#################################################
# 
# Basic test of Stan models for spaceless data.
# To test the model fits, we generate data from a simple
# generative model and fit it with the model.
# 
##################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import stan
import matplotlib.pyplot as plt
import seaborn as sns
import os

###############
# 1) GENERATE THE DATA
###############

# Dataset parameters
nTrials = 50  # number of trials per condition
neuronsConnN = 30  # number of neurons connected to the target
neuronsNetN = 100  # number of neurons in the network with target
neuronsOutNetN = 50  # number of neurons outside of the target's network
baselineFiring = 10  # mean baseline firing rate
neuronBaselineSd = 3  # variability in baseline firing rate
connEffect = 0.5  # mean effect of stimulation on FR of connected neuron
netEffect = 0.1  # mean effect of stimulation on FR of in-network neuron
residualSd = 3  # trial response variability

nNeurons = neuronsConnN + neuronsNetN + neuronsOutNetN
neuronType = ['C'] * neuronsConnN + ['N'] * neuronsNetN + \
  ['O'] * neuronsOutNetN
typeEffects = {'C': connEffect, 'N': netEffect, 'O': 0}

# Define the means for each condition and group
neuronBaselines = np.random.normal(0, neuronBaselineSd, nNeurons) + \
  baselineFiring

# Generate the dataset
data = []
for neuron in range(nNeurons):
    for stimulation in [0, 1]:
        samples = np.random.normal(neuronBaselines[neuron] +
                                   typeEffects[neuronType[neuron]] * stimulation,
                                   residualSd, nTrials)
        data.extend(zip([neuron] * nTrials,
                        [neuronType[neuron]] * nTrials,
                        range(1, nTrials + 1),
                        [stimulation] * nTrials, samples))

# Create a DataFrame from the generated data
expData = pd.DataFrame(data, columns=['Neuron', 'Type', 'Trial',
                                      'Stim', 'Response'])


###############
# 2) PLOT THE RAW DATA
###############

# Calculate the mean and standard deviation for each group and condition
grouped_data = expData.groupby(['Neuron', 'Stim'])
means = grouped_data['Response'].mean().unstack()
means['Type'] = neuronType # add the neuron type to the means DataFrame

# Set up the plot
fig, ax = plt.subplots()

# Plot the means and standard deviations
for nt in ['C', 'N', 'O']:
    x = [0, 1]
    y = means.loc[means['Type'] == nt, [0, 1]].values[0]
    std = expData.loc[expData['Type'] == nt].groupby('Stim')['Response'].std().values
    ax.errorbar(x, y, yerr=std, marker='o', linestyle='-', capsize=4, label=f'Neuron Type {nt}')
    # Join the means of the two conditions with a line
    ax.plot(x, y, marker='o', linestyle='-', color='black', linewidth=0.5)

# Set the labels and title
ax.set_xlabel('Condition')
ax.set_ylabel('Response')
ax.set_title('Mean Response by Condition and Neuron Type')
# Add a legend
ax.legend()
# Show the plot
plt.show()


###############
# 3) FIT MODEL WITH A SINGLE RANDOM EFFECTS DISTRIBUTION
###############

# Prepare the data
stan_data = {
    'nNeurons': nNeurons,
    'N': nNeurons * nTrials * 2,
    'Stimulation': expData['Stim'].values,
    'Response': expData['Response'].values,
    'neuronId': expData['Neuron'].values + 1,
    'grainsize': 50
}

# Load model code
with open('./stan_models/2_spaceless_mixed.stan', 'r') as file:
    model_code = file.read()

# Compile the model
model = stan.build(program_code=model_code, data=stan_data)
#httpstan.cache.delete_model_directory(posterior.model_name)

initValues = {
    'baselineFR': baselineFiring,
    'sigmaBaselineFR': neuronBaselineSd,
    'neuronBaseFR': neuronBaselines,
    'neuronEffect': np.ones(nNeurons)*0.2,
    'sigmaResidual': residualSd,
    'meanStimEffect': 0.2,
    'sigmaEffect': 0.2,
    'p': 1-(neuronsOutNetN/nNeurons)}


initValuesChains = [initValues, initValues]

import os
os.environ["STAN_NUM_THREADS"] = "4"

# Fit the model
posterior = model.sample(num_chains=2, init=initValuesChains)

# Save posterior
posterior.to_frame().to_csv('./stan_models/2_spaceless_mixed_posterior.csv')


###############
# 4) PROCESS MODEL FIT OUTPUT
###############

#### GET THE NEURON-SPECIFIC PARAMETERS IN A TIDY LONG FORMAT
## Extract the samples
#fitDf = posterior.to_frame()
## Get list of columns related to neuronBaseFR
#baseFR_cols = [col for col in fitDf.columns if 'neuronBaseFR' in col]
## Get list of columns related to neuronEffect
#effect_cols = [col for col in fitDf.columns if 'neuronEffect' in col]
## Add a 'draws' column using the DataFrame's index
#fitDf['draws'] = fitDf.index
#
## Melt these columns to long format
#baseFR_df = fitDf.melt(id_vars=['draws'], value_vars=baseFR_cols,
#                    var_name='neuronId', value_name='baseFR')
#effect_df = fitDf.melt(id_vars=['draws'], value_vars=effect_cols,
#                    var_name='neuronId', value_name='stimEffect')
#
## Extract neuron numbers from the neuronId column
#baseFR_df['neuronId'] = baseFR_df['neuronId'].str.extract('(\d+)').astype(int)
#effect_df['neuronId'] = effect_df['neuronId'].str.extract('(\d+)').astype(int)
#
## Merge the two datasets
#neuronDf = pd.merge(baseFR_df, effect_df,  how='left',
#                    left_on=['neuronId','draws'], right_on = ['neuronId','draws'])
#
## Calculate the proportion of positive stimEffect draws for each neuron
#neuronDf['positive'] = neuronDf['stimEffect'] > 0
#propPositive = neuronDf.groupby('neuronId')['positive'].mean()
#
## Calculate whether this proportion is significantly high
#significant = (propPositive > 0.975).astype(int)
#
#### Compute the neuron-specific means of each parameter
#neuronMeans = neuronDf.groupby('neuronId')[['baseFR', 'stimEffect']].mean().reset_index()
#neuronMeans['propPositive'] = propPositive.values
#neuronMeans['significant'] = significant.values
## Rename columns for clarity
#neuronMeans.columns = ['neuronId', 'meanBaseFR', 'meanStimEffect', 'propPositive', 'significant']
## Add neuron type
#neuronMeans['Type'] = neuronType
#
#### GET THE POPULATION PARAMETERS
## Get the population-level parameters
#
#baselineFR = fitDf['baselineFR'].mean()
#meanEffect = fitDf['meanStimEffect'].mean()
#sigmaEffect = fitDf['sigmaEffect'].mean()
#sigmaBaseline = fitDf['sigmaBaselineFR'].mean()
#sigmaRes = fitDf['sigmaResidual'].mean()
#pValBayes = (fitDf['meanStimEffect'] > 0).mean()
#
#populationDict = {'baselineFR': baselineFR,
#                  'meanEffect': meanEffect,
#                  'sigmaEffect': sigmaEffect,
#                  'sigmaBaseline': sigmaBaseline,
#                  'sigmaRes': sigmaRes,
#                  'pValBayes': pValBayes}
#
################
## 5) DO SIMPLER ANALYSIS
################
#
## Do paired t-test of the Response across conditions, pairing by neuron
#import scipy.stats as stats
#
## GET NEURON-WISE PARAMETERS
#pVals = np.zeros(nNeurons)
#meanDiff = np.zeros(nNeurons)
#for neuron in range(nNeurons):
#    neuronData = expData.loc[expData['Neuron'] == neuron, ['Stim', 'Response']]
#    # Get p-value for the neuron
#    pairedTtest = stats.ttest_ind(neuronData.loc[(neuronData['Stim'] == 0), 'Response'],
#                                  neuronData.loc[(neuronData['Stim'] == 1), 'Response'])
#    pVals[neuron] = pairedTtest.pvalue
#    # Get mean difference for the neuron
#    meanDiff[neuron] = neuronData.loc[(neuronData['Stim'] == 1), 'Response'].mean() - \
#                neuronData.loc[(neuronData['Stim'] == 0), 'Response'].mean()
#
#neuronMeans['pValTtest'] = pVals
#neuronMeans['naiveEffect'] = meanDiff
#
## GET POPULATION-LEVEL PARAMETERS
## Paired t-test across conditions, pairing by neuron
#pairedTtest = stats.ttest_rel(expData.loc[expData['Stim'] == 0, 'Response'],
#                                expData.loc[expData['Stim'] == 1, 'Response'])
#pValPop = pairedTtest.pvalue
## Wilcoxon signed-rank test across conditions, pairing by neuron
#wilcoxon = stats.wilcoxon(meanDiff)
## Get the mean difference across conditions
#popDiff = np.mean(meanDiff)
#
#populationDict['pValTtest'] = pValPop
#populationDict['pValWilcoxon'] = wilcoxon.pvalue
#populationDict['popEffectNaive'] = popDiff
#
#
################
## 5) PLOT MODEL FIT RESULTS
################
#
## Plot histogram of meanStimEffect
#fig, ax = plt.subplots()
#sns.histplot(neuronMeans['meanStimEffect'], ax=ax)
#ax.set_title('Histogram of Mean Stimulus Effect')
#plt.show()
#
## Plot histogram of meanStimEffect by Type
#fig, ax = plt.subplots()
#sns.histplot(x='meanStimEffect', hue='Type', data=neuronMeans, ax=ax)
#ax.set_title('Histogram of Mean Stimulus Effect by Type')
#plt.show()
#
## Plot histogram of propPositive by type
#fig, ax = plt.subplots()
#sns.histplot(x='propPositive', hue='Type', data=neuronMeans, ax=ax, bins=10)
#ax.set_title('Histogram of Proportion of Positive Stimulus Effect by Type')
#plt.show()
#
## First subplot: meanStimEffect vs Type
#sns.scatterplot(x='Type', y='meanStimEffect', data=neuronMeans, ax=axes[0])
#axes[0].set_title('Mean Stimulus Effect by Neuron Type')
#
## Second subplot: propPositive vs Type
#sns.scatterplot(x='Type', y='propPositive', data=neuronMeans, ax=axes[1])
#axes[1].set_title('Proportion of Positive Stimulus Effect by Neuron Type')
#
## Third subplot: Bar plot of proportion of 'significant' neurons by Type
#proportion_significant = neuronMeans.groupby('Type')['significant'].mean()
#proportion_significant.plot(kind='bar', ax=axes[2])
#axes[2].set_title('Proportion of Significant Neurons by Type')
#
#plt.tight_layout()
#plt.show()
#
#
