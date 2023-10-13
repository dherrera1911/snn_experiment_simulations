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
from functions_analysis import *


###############
# 1) GENERATE THE DATA
###############

# Dataset parameters
nTrials = 15  # number of trials per condition
neurons1N = 50  # number of neurons with effect (population 1)
neurons2N = 100  # number of neuronsw without effect (population 2)
baselineFiring = 14  # mean baseline firing rate
neuronBaselineSd = 3  # variability in baseline firing rate
effect = 2  # mean effect of stimulation on population 1
effectSd = 0.8  # variability in effect of stimulation
residualSd = 3  # trial response variability

nNeurons = neurons1N + neurons2N
neuronType = np.array([0] * neurons1N + [1] * neurons2N)
typeEffects = [effect, 0]

# Generate the dataset
data = []
# Generate the baseline firing rates for each neuron
neuronBaselines = np.random.normal(0, neuronBaselineSd, nNeurons) + \
  baselineFiring
# Define the means for each condition and group
nrnEffect = np.zeros(nNeurons)
for neuron in range(nNeurons):
    if neuronType[neuron] == 0:
        nrnEffect[neuron] = np.max(np.random.normal(typeEffects[0], effectSd), 0)
    else:
        nrnEffect[neuron] = 0
    for stimulation in [0, 1]:
        samples = np.random.normal(neuronBaselines[neuron] +
                                   nrnEffect[neuron] * stimulation,
                                   residualSd, nTrials)
        data.extend(zip([neuron] * nTrials,
                        [neuronType[neuron]] * nTrials,
                        range(1, nTrials + 1),
                        [stimulation] * nTrials, samples))

# Create a DataFrame from the generated data
expData = pd.DataFrame(data, columns=['Neuron', 'Type', 'Trial',
                                      'Stim', 'Response'])


###############
# 2) DO SIMPLE ANALYSES OF THE DATA
###############

# GET NEURON-WISE ANALYSES
pVals, meanDiff = neuron_specific_analysis_2_cond(expData)

pSignif = pVals < 0.05
truePos = np.sum(pSignif[neuronType == 0])
falsePos = np.sum(pSignif[neuronType == 1])
trueNeg = np.sum(~pSignif[neuronType == 1])
falseNeg = np.sum(~pSignif[neuronType == 0])

## GET POPULATION-LEVEL PARAMETERS

## Wilcoxon signed-rank test across conditions, pairing by neuron
wilcoxon = stats.wilcoxon(meanDiff)
pValPopWil = wilcoxon.pvalue
## Paired t-test across conditions, pairing by neuron
# compute the mean for each condition for each neuron
meanCond0 = expData[expData['Stim'] == 0].groupby('Neuron')['Response'].mean()
meanCond1 = expData[expData['Stim'] == 1].groupby('Neuron')['Response'].mean()
pairedTtest = stats.ttest_rel(meanCond0, meanCond1)
pValPopT = pairedTtest.pvalue

# Pop mean
popEffect = meanDiff.mean()
popEffectSig = meanDiff[pSignif].mean()


###############
# 3) FIT A MIXTURE HIERARCHICAL MODEL
###############

# Turn Stim and Response into matrices where each row is a neuron
# and each column is a trial (for each condition)
stimMatrix = expData.pivot(index='Neuron', columns=['Trial', 'Stim'],
                            values='Stim').values
responseMatrix = expData.pivot(index='Neuron', columns=['Trial', 'Stim'],
                                values='Response').values

# Prepare the data
stan_data = {
    'nNeurons': nNeurons,
    'nObsPerNeuron': nTrials * 2,
    'Stimulation': stimMatrix,
    'Response': responseMatrix
}

stan_file = './stan_models/2_spaceless_mixed.stan'
model = CmdStanModel(stan_file=stan_file)

# Fit the model
posterior = model.sample(data=stan_data, chains=2, iter_warmup=1500,
                         iter_sampling=1500, adapt_delta=0.999)
summary = az.summary(posterior, hdi_prob=0.68)

#print(summary.loc[summary.index.str.startswith('effectPop'), 'mean'])

#posteriorVar = model.variational(data=stan_data)
#print(posteriorVar.variational_params_dict['pMix'])
#print(posteriorVar.variational_params_dict['effectPop'])
#print(posteriorVar.variational_params_dict['interceptPop'])



###############
# 4) COMPARE RESULTS OF THE MODEL TO THE SIMPLE ANALYSES
###############

### NEURON LEVEL-PARAMETERS

# CLUSTERING OF THE NEURONS

# Get the probability that each neuron has an effect
pNrn = summary.loc[summary.index.str.startswith('pNrn'), 'mean']

# Compute ROC curve for the model and simple analysis
fprBay, tprBay, thresholdsBay = roc_curve(1-neuronType, pNrn)
fpr, tpr, thresholds = roc_curve(1-neuronType, 1-pVals)

# Plot ROC curve
# concatenate the two pandas dataframes
rocDf = pd.concat((pd.DataFrame({'fpr': fprBay, 'tpr': tprBay, 'model': 'Model'}),
  pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'model': 'Simple'})))

(pn.ggplot(rocDf, pn.aes(x='fpr', y='tpr', color='model')) +
    pn.geom_line() +
    pn.geom_abline(linetype='dashed') +
    pn.labs(x='False positive rate', y='True positive rate') +
    pn.theme_bw())

# Get the true positives, etc, for the model
plt.hist(pNrn, 30)
plt.show()

threshold = 0.5
pSignifBay = pNrn > threshold
truePosBay = np.sum(pSignifBay[neuronType == 0])
falsePosBay = np.sum(pSignifBay[neuronType == 1])
trueNegBay = np.sum(~pSignifBay[neuronType == 1])
falseNegBay = np.sum(~pSignifBay[neuronType == 0])

results_df = pd.DataFrame({
    'Metric': ['True Positives', 'False Positives',
               'True Negatives', 'False Negatives'],
    'Basic Analysis': [truePos, falsePos, trueNeg, falseNeg],
    'Model': [truePosBay, falsePosBay, trueNegBay, falseNegBay]
})
print(results_df.to_string(index=False))


# EFFECTS OF THE NEURONS
effectNrnBay = summary.loc[summary.index.str.startswith('effectNrn'),
                           'mean'].values
neuronDf = pd.DataFrame({'neuronType': neuronType,
                         'effectTrue': nrnEffect,
                         'effectBasic': meanDiff,
                         'effectModel': effectNrnBay,
                         'pVal': pVals,
                         'pModel': pNrn,
                         'pSignifBasic': pSignif,
                         'pSignifModel': pSignifBay})

# Plot the estimated effects vs the true effects
# Filter neuronDf to only show neuronType==0 neurons
(pn.ggplot(neuronDf[neuronType==0],
           pn.aes(x='effectTrue', y='effectModel', shape='pSignifModel')) +
    pn.geom_point(color='k') +
    pn.geom_point(pn.aes(y='effectBasic', shape='pSignifBasic'), color='r') +
    pn.geom_abline(linetype='dashed') +
    pn.labs(x='True effect', y='Estimated effect') +
    pn.theme_bw())

### POPULATION LEVEL-PARAMETERS

# MIXING PROBABILITIES

# Real mixing probability
pMixReal = neurons1N/nNeurons
# Get posterior of pMix from model fit
pMixPost = posterior.stan_variable(var='pMix')
pMix = np.mean(pMixPost)
# Get mean and CI for basic analysis
pMixBasic = np.mean(pSignif)
# Get the density of binomial for the basic test
x = np.arange(0, 1, 0.01)
pMixBasicDens = stats.binom.pmf(k=np.sum(pSignif), n=nNeurons, p=x)
pMixBasicDens = pMixBasicDens / np.sum(pMixBasicDens*0.01)


vline_data = pd.DataFrame({
    'xintercept': [pMixReal, pMix, pMixBasic],
    'color': ['True value', 'Model mean', 'Basic analysis']
})

(pn.ggplot() +
 pn.geom_density(pn.aes(x='pMixPost'), color='k') +
 pn.geom_line(pn.aes(x='x', y='pMixBasicDens'),
                 data=pd.DataFrame({'x': x, 'pMixBasicDens': pMixBasicDens}),
                 color='b') +
 pn.geom_vline(pn.aes(xintercept='xintercept', color='color'), vline_data) +
 pn.scale_color_manual(values=['b', 'k', 'r']) +
 pn.labs(x='Mixing probability', color='Legend Title') +
 pn.theme_bw())


# EFFECT SIZE

# Get posterior of effect size from model fit
effectPost = posterior.stan_variable(var='effectPop')
effectMean = np.mean(effectPost)
effectSigmaPost = posterior.stan_variable(var='effectSigma')
effectSd = np.mean(effectSigmaPost)

# Get mean and CI for basic analysis
### GET BETTER PDF MAYBE, USING THE LIKELIHOOD OF DIFFERENT
# MEANS GIVEN THE DATA
x = np.arange(0, 3, 0.01)
effectBasic = np.mean(meanDiff[pSignif])
effectBasicSd = stats.sem(meanDiff[pSignif])
effectBasicDens = stats.norm.pdf(x, loc=effectBasic, scale=effectBasicSd)
effectSigmaBasic = np.std(meanDiff[pSignif])

vline_data = pd.DataFrame({
    'xintercept': [effect, effectMean, effectBasic],
    'color': ['True value', 'Model mean', 'Basic analysis']
})


(pn.ggplot() +
 pn.geom_density(pn.aes(x='effectPost'), color='k') +
 pn.geom_line(pn.aes(x='x', y='effect'),
                 data=pd.DataFrame({'x': x, 'effect': effectBasicDens}),
                 color='b') +
 pn.geom_vline(pn.aes(xintercept='xintercept', color='color'), vline_data) +
 pn.scale_color_manual(values=['b', 'k', 'r']) +
 pn.labs(x='Effect size', color='Legend Title') +
 pn.theme_bw())


