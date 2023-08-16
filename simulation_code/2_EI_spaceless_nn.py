#####################################################
# 
# Simulate a network of neurons with inhibitory and excitatory
# neurons but no spatial structure. Two populations of E/I
# neurons are simulated, with the E neurons receiving input
# from the I neurons and vice versa.
# 
#####################################################


import numpy as np
import pandas as pd
import nest
import nest.voltage_trace
import matplotlib.pyplot as plt
import auxiliary_functions as af
import json
import time
import os


start = time.time()
saveDataDir = './data/'
#############
# Simulation parameters
#############

plotOrSave = 'save'  # 'plot' or 'save' the results
nThreads = 4

# Experiment parameters
nTrials = 1000
simTime = 1800.0  # Total simulation time in ms
recordDelay = 300.0

# Population size
nPopulations = 2
nExcitatory = 700    # Number of excitatory neurons per population
nInhibitory = 100    # Number of inhibitory neurons per population

# Connectivity parameters
inDegreeE2Iwithin = 7        # Number of incoming connections from E to I
inDegreeE2Iout = 14        # Number of incoming connections from E to I
inDegreeE2E = 25        # Number of incoming connections from E to E
inDegreeI2Ewithin = 8  # Number of incoming connections from I to E
inDegreeI2Eout = 0     # Number of incoming connections from I to E
weightSynE = 35.0
weightSynI = -110.0

# Noisy drive parameters
noiseStrength = 120
noiseRate = 35   # Rate of poisson noise generators

# Stimulation drive
stimCurrent = 600.0  # Current in pA

# Neuron biophysics parameters
#neuronType = 'iaf_psc_alpha'
#edict = {"I_e":  140.0,      # External input current pA
#         "tau_m": 20.0}     # Membrane time constant in ms
#idict = {"I_e": 90.0,      # External input current pA
#         "tau_m": 20.0}     # Membrane time constant in ms

I_e1 = 130.0
I_e2 = 90.0
g_L = 30.0
t_ref = 10.0
V_reset = -70
neuronType = 'aeif_cond_alpha'
edict = {'I_e': I_e1, # External input current pA
         'g_L': g_L,  # Leak conductivity
         't_ref': t_ref,  # Refractivity period
         'V_reset': V_reset}  # Reset voltage
idict = {'I_e': I_e2,
         'g_L': g_L,
         't_ref': t_ref,
         'V_reset': V_reset}

#############
# Turn parameters into appropriate structures
#############

# Implement the connectivity in the network
connMatE2E = af.connectivity_mat_2pop(withinPop=inDegreeE2E, betweenPop=0)
connMatI2E = af.connectivity_mat_2pop(withinPop=inDegreeI2Ewithin,
                                   betweenPop=inDegreeI2Eout)
connMatE2I = af.connectivity_mat_2pop(withinPop=inDegreeE2Iwithin,
                                   betweenPop=inDegreeE2Iout)
connMatI2I = af.connectivity_mat_2pop(withinPop=0, betweenPop=0)

# Noise characteristics in dictionary
noiseDict = {'rate': noiseRate}
noiseWDict = {'weight': noiseStrength}

trialDfList = []

for trial in range(nTrials):
    for stimulation in [False, True]:
        nest.ResetKernel()
        #############
        # Set random seed at constant value to have same connectivity always
        #############
        nest.SetKernelStatus({'rng_seed': 1911,
                              'update_time_limit': 1.0,
                              'local_num_threads': nThreads})

        #############
        # Create neurons
        #############
        # Create the neuron classes to use
        nest.CopyModel(neuronType, 'excitatory', edict)
        nest.CopyModel(neuronType, 'inhibitory', idict)
        # Create neurons
        popE = []
        popI = []
        noiseE = []
        noiseI = []
        allNeurons = []
        for n in range(nPopulations):
            popE.append(nest.Create('excitatory', nExcitatory))
            popI.append(nest.Create('inhibitory', nInhibitory))
            allNeurons = allNeurons + list(popE[n].get('global_id'))
            allNeurons = allNeurons + list(popI[n].get('global_id'))
            # Add noise
            noiseE.append(nest.Create('poisson_generator', n=nExcitatory, params=noiseDict))
            nest.Connect(pre=noiseE[n], post=popE[n], conn_spec={'rule': 'one_to_one'},
                         syn_spec=noiseWDict)
            noiseI.append(nest.Create('poisson_generator', n=nInhibitory, params=noiseDict))
            nest.Connect(pre=noiseI[n], post=popI[n], conn_spec={'rule': 'one_to_one'},
                         syn_spec=noiseWDict)

        #############
        # Connect neurons
        #############
        # Implement the connectivity in the network
        af.implement_EI_connectivity(popE=popE, popI=popI, connMatE2E=connMatE2E,
                                  connMatE2I=connMatE2I, connMatI2E=connMatI2E,
                                  connMatI2I=connMatI2I, weightSynE=weightSynE,
                                  weightSynI=weightSynI)

        #############
        # Add excitatory device
        #############
        targetId = popE[0][0].get('global_id')
        if stimulation:
            # Optogenetic stimulation
            opto = nest.Create('dc_generator', params={'amplitude': stimCurrent,
                                                       'start': 100.0, 'stop': simTime})
            # Connect the optogenetic device to the excitatory population
            nest.Connect(opto, [targetId])

        #############
        # Connect recording devices
        #############
        # Create spike detectors
        spikeRecorder = nest.Create('spike_recorder')
        spikeRecorderID = spikeRecorder.get('global_id')
        multimeterE = []
        multimeterI = []
        for n in range(nPopulations):
            nest.Connect(popE[n] + popI[n], spikeRecorder)
            # Set up representative multimeters
            multimeterE.append(nest.Create('multimeter', params={'record_from': ['V_m']}))
            multimeterI.append(nest.Create('multimeter', params={'record_from': ['V_m']}))
            nest.Connect(multimeterE[n], popE[n][0])
            nest.Connect(multimeterI[n], popI[n][0])

        #############
        # Run the simulation
        #############
        # Change seed to random seed, so that each trial is different
        nest.SetKernelStatus({'rng_seed': np.random.randint(low=1, high=1000000)})
        # Set up and run simulation
        nest.Simulate(simTime)

        #############
        # Extract and tidy the data
        #############
        # Get events from spike recorder
        data = nest.GetStatus(spikeRecorder, keys='events')[0]

        # Extract times and senders
        senders = data['senders']
        spikeTimes = data['times']
        # Filter out spikes before recording time
        senders = senders[spikeTimes > recordDelay]
        spikeTimes = spikeTimes[spikeTimes > recordDelay]

        # Create a dictionary that will store neuron ID as keys and number of spikes as values
        neuronSpikes = {}
        # Initialize the dictionary with all neurons and set their spike counts to 0
        for neuron in allNeurons:
            neuronSpikes[neuron] = 0
        # Increase the spike count for each spike detected
        for sender in senders:
            neuronSpikes[sender] += 1

        # Now, let's create a dataframe and fill it with the desired information
        neuronDfList = []  # We will store individual dataframes here and concatenate them later

        directTargets = nest.GetConnections(source=popE[0][0]).get('target')
        directTargets.remove(spikeRecorderID) # Remove the spike recorder from the list
        for neuronID, spikes in neuronSpikes.items():
            # Check if the neuron is the optogenetic target
            target = int(neuronID == targetId)
            # Check if the neuron receives connections from the target
            directConn = int(neuronID in directTargets)
            # Check if the neuron is in same population as the target
            inNetwork = int(neuronID in popE[0] or neuronID in popI[0])
            # Get neuron type
            if neuronID in popE[0] or neuronID in popE[1]:
                nType = 'E'
            else:
                nType = 'I'
            # Create a dataframe for this neuron and add it to the list
            neuronDf = pd.DataFrame({'Trial': [trial],
                                     'Stimulation': stimulation,
                                     'NeuronID': [neuronID],
                                     'Target': [target],
                                     'DirectConn': [directConn],
                                     'SamePop': [inNetwork],
                                     'Type': [nType],
                                     'Activation': [spikes]})
            neuronDfList.append(neuronDf)

        # Concatenate all the dataframes
        trialDf = pd.concat(neuronDfList, ignore_index=True)
        trialDf = trialDf.sort_values(by='NeuronID')
        trialDfList.append(trialDf)
        print(f'Trial {trial+1} complete')


# Concatenate the trial Df's into the full experiment Df
experimentDf = pd.concat(trialDfList, ignore_index=True)

# Make dictionary with simulation parameters
paramDict = {'nTrials': nTrials,
             'simTime': simTime,
             'recordDelay': recordDelay,
             'nExcitatory': nExcitatory,
             'nInhibitory': nInhibitory,
             'nPopulations': nPopulations,
             'inDegreeE2E': inDegreeE2E,
             'inDegreeE2Iwithin': inDegreeE2Iwithin,
             'inDegreeE2Iout': inDegreeE2Iout,
             'weightSynE': weightSynE,
             'weightSynI': weightSynI,
             'noiseStrength': noiseStrength,
             'noiseRate': noiseRate,
             'stimCurrent': stimCurrent,
             'neuronType': neuronType,
             'I_e1': I_e1,
             'I_e2': I_e2,
             'g_L': g_L}

if plotOrSave == 'save':
    os.makedirs(saveDataDir, exist_ok=True)
    # Save param dict
    with open(f'{saveDataDir}2_EI_spaceless_params.json', 'w') as fp:
        json.dump(paramDict, fp)
    # Save the dataframe
    experimentDf.to_csv(f'{saveDataDir}2_EI_spaceless.csv', index=False)

elif plotOrSave == 'plot':
    #############
    # Plot results
    #############
    trialDf = trialDf[trialDf['Stimulation']]
    # Raster plot of spiking activity
    colors = {'E': 'blue', 'I': 'red'}
    labels = ['Population 1', 'Population 2']
    # Set up matplotlib figure and axes
    histMax = np.max(trialDf['Activation'])
    for n in range(2):
        popE_ids = [neuron.get('global_id') for neuron in popE[n]]
        popI_ids = [neuron.get('global_id') for neuron in popI[n]]
        # Retrieve the data from the spike detector
        events = nest.GetStatus(spikeRecorder, keys='events')[0]
        senders = events['senders']
        times = events['times']
        # Separate excitatory and inhibitory neurons for plotting
        times_E = times[np.isin(senders, popE_ids)]
        senders_E = senders[np.isin(senders, popE_ids)]
        times_I = times[np.isin(senders, popI_ids)]
        senders_I = senders[np.isin(senders, popI_ids)]
        # Plot using matplotlib's scatter function
        plt.subplot(2,2,n*2+1)
        plt.scatter(times_E, senders_E, color=colors['E'], label='Excitatory', s=5)
        plt.scatter(times_I, senders_I, color=colors['I'], label='Inhibitory', s=5)
        # Format the plot
        plt.ylabel(f'Neuron ID\n{labels[n]}')
        plt.legend()
        # Histogram of activations
        plt.subplot(2,2,n*2+2)
        # Extract data for this row's population
        if n==0:
            popDf = trialDf[trialDf['SamePop']==1]
        else:
            popDf = trialDf[trialDf['SamePop']==0]
        plt.hist(popDf[popDf['Type']=='E']['Activation'], color=colors['E'],
                 label='Excitatory', alpha=0.5, bins=np.arange(histMax))
        plt.hist(popDf[popDf['Type']=='I']['Activation'], color=colors['I'],
                  label='Inhibitory', alpha=0.5, bins=np.arange(histMax))
        plt.xlim(0, histMax)
    # Display the plot
    plt.show()

    samePopInd = trialDf['SamePop']==1
    excitatoryInd = trialDf['Type']=='E'
    targetInd = trialDf['Target']==1
    excSamePop = np.logical_and(samePopInd, excitatoryInd)
    excSamePop[targetInd] = False
    inhSamePop = np.logical_and(samePopInd, ~excitatoryInd)
    excDiffPop = np.logical_and(~samePopInd, excitatoryInd)
    inhDiffPop = np.logical_and(~samePopInd, ~excitatoryInd)
    directConnExc = np.logical_and(trialDf['DirectConn']==1, excitatoryInd)
    directConnInh = np.logical_and(trialDf['DirectConn']==1, ~excitatoryInd)

    print(f"Total mean: {np.mean(trialDf['Activation'])}")
    print(f"E-same: {np.mean(trialDf['Activation'][excSamePop])}")
    print(f"I-same: {np.mean(trialDf['Activation'][inhSamePop])}")
    print(f"E-diff: {np.mean(trialDf['Activation'][excDiffPop])}")
    print(f"I-diff: {np.mean(trialDf['Activation'][inhDiffPop])}")
    print(f"E-direct: {np.mean(trialDf['Activation'][directConnExc])}")
    print(f"I-direct: {np.mean(trialDf['Activation'][directConnInh])}")


#    # Plot the membrane potential of a neuron
#    vmI = []
#    tsI =[]
#    vmE = []
#    tsE = []
#    for n in range(nPopulations):
#        # Plot inhibitory neuron trace
#        vmI.append(multimeterI[n].get("events")['V_m'])
#        tsI.append(multimeterI[n].get("events")["times"])
#        plt.subplot(2,2,1+n*2)
#        plt.plot(tsI[n], vmI[n])
#        # Plot excitatory neuron trace
#        vmE.append(multimeterE[n].get("events")['V_m'])
#        tsE.append(multimeterE[n].get("events")["times"])
#        plt.subplot(2,2,2+n*2)
#        plt.plot(tsE[n], vmE[n])
#    plt.show()
#
