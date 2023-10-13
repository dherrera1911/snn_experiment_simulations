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
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel
import seaborn as sns
import os
import arviz as az

###############
# 1) GENERATE THE DATA
###############

# Dataset parameters
nTrials = 20  # number of trials per condition
neurons1N = 40  # number of neurons with effect (population 1)
neurons2N = 20  # number of neuronsw without effect (population 2)
baselineFiring = 14  # mean baseline firing rate
neuronBaselineSd = 3  # variability in baseline firing rate
effect = 3  # mean effect of stimulation on population 1
effectSd = 0.5  # variability in effect of stimulation
residualSd = 3  # trial response variability

nNeurons = neurons1N + neurons2N
neuronType = ['1'] * neurons1N + ['2'] * neurons2N
typeEffects = {'1': effect, '2': 0}

# Generate the dataset
data = []
# Generate the baseline firing rates for each neuron
neuronBaselines = np.random.normal(0, neuronBaselineSd, nNeurons) + \
  baselineFiring
# Define the means for each condition and group
nrnEffect = np.zeros(nNeurons)
for neuron in range(nNeurons):
    if neuronType[neuron] == '1':
        nrnEffect[neuron] = np.max(np.random.normal(typeEffects['1'], effectSd),
                                   0)
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


# See gamma distributions
from scipy.stats import gamma

#x = np.linspace(0, 3, 1000)
#plt.plot(x, gamma.pdf(x, a=1.4, loc=0), label='1, 1')
#plt.show()


###############
# 3) FIT MODEL WITH A MIXTURE OF MIXED EFFECTS
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
posterior = model.sample(data=stan_data, chains=2, iter_warmup=1000,
                         iter_sampling=1000, adapt_delta=0.99)
summary = az.summary(posterior)

effectNrn = summary.loc[summary.index.str.startswith('effectNrn'), 'mean']
pNrn = summary.loc[summary.index.str.startswith('pNrn'), 'mean']
pMix = summary.loc[summary.index.str.startswith('pMix'), 'mean']

plt.scatter(effectNrn[:neurons1N], nrnEffect[:neurons1N])
plt.show()


