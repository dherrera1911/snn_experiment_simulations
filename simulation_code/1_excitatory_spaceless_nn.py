#####################################################
# 
# Simulate a network of neurons with only excitatory
# neurons but no spatial structure. Two populations of
# excitatory neurons are simulated. The two populations don't
# share any connections, so one of them is not affected by the
# optogenetic stimulation.
# 
#####################################################

import numpy as np
import pandas as pd
import nest
import nest.voltage_trace
import matplotlib.pyplot as plt
import json


#############
# Simulation parameters
#############

plotOrSave = 'save' # 'plot' or 'save' the results

# Experiment parameters
nTrials = 500
simTime = 800.0  # Total simulation time in ms
recordDelay = 300.0

# Population size
nNeurons1 = 400    # Number of neurons connected to stimulated neuron
nNeurons2 = 400    # Number of neurons not in network with stimulated neuron

# Conectivity parameters
inDegreeConnect = 25  # incoming connections to each neuron
wSyn = 25.0

# Noisy drive parameters
noiseStrength = 150
noiseRate = 30   # Rate of poisson noise generators

# Stimulation drive
stimCurrent = 500.0

# Neuron biophysics parameters
neuronType = 'aeif_cond_alpha'
I_e = 130.0
g_L = 30.0
t_ref = 10.0
V_reset = -70
edict = {'I_e': I_e,
         'g_L': g_L,
         'gsl_error_tol': 1e-8,
         't_ref': t_ref,
         'V_reset': V_reset}      # External input current pA

#############
# Turn parameters into appropriate structures
#############

# Noise characteristics in dictionary
noiseDict = {'rate': noiseRate}
noiseWDict = {'weight': noiseStrength}
connDict = {'rule': 'fixed_indegree', 'indegree': inDegreeConnect,
            'allow_autapses': False, 'allow_multapses': False}

trialDfList = []

for trial in range(nTrials):
    for stimulation in [False, True]:
        nest.ResetKernel()
        #############
        # Set random seed at constant value to have same connectivity always
        #############
        nest.SetKernelStatus({'rng_seed': 1911,
                              'update_time_limit': 1.0})

        #############
        # Create neurons
        #############
        # Create the neuron classes to use
        nest.CopyModel(neuronType, 'excitatory', edict)
        # Create neurons
        pop1 = nest.Create('excitatory', n=nNeurons1)
        pop2 = nest.Create('excitatory', n=nNeurons2)
        allNeurons = list(pop1.get('global_id')) + list(pop2.get('global_id'))
        # Create noise generators
        noise1 = nest.Create('poisson_generator', n=nNeurons1, params=noiseDict)
        noise2 = nest.Create('poisson_generator', n=nNeurons2, params=noiseDict)

        #############
        # Connect neurons
        #############
        # Connect noise generators to neurons
        nest.Connect(pre=noise1, post=pop1, conn_spec={'rule': 'one_to_one'},
                      syn_spec={'weight': noiseStrength})
        nest.Connect(pre=noise2, post=pop2, conn_spec={'rule': 'one_to_one'},
                      syn_spec={'weight': noiseStrength})
        # Connect neuron populations
        nest.Connect(pre=pop1, post=pop1, conn_spec=connDict,
                     syn_spec={'weight': wSyn})
        nest.Connect(pre=pop2, post=pop2, conn_spec=connDict,
                     syn_spec={'weight': wSyn})

        #############
        # Create and connect recording devices
        #############
        # Create spike recorder
        spikeRecorder = nest.Create('spike_recorder')
        spikeRecorderID = spikeRecorder.get('global_id')
        nest.Connect(pre=pop1, post=spikeRecorder)  # Put recorder on neurons
        nest.Connect(pre=pop2, post=spikeRecorder)  # Put recorder on neurons
        # Create multimeter
        multimeter = nest.Create("multimeter")
        multimeter.set(record_from=["V_m"])
        nest.Connect(pre=multimeter, post=pop1[1])  # Put recorder on neurons

        #############
        # Add excitatory device
        #############
        targetId = pop1[0].get('global_id')
        if stimulation:
            # Optogenetic stimulation
            opto = nest.Create('dc_generator', params={'amplitude': stimCurrent,
                                                       'start': 100.0, 'stop': simTime})
            # Connect the optogenetic device to the excitatory population
            nest.Connect(opto, [targetId])

        #############
        # Simulate
        #############
        # Change seed to random seed, so that each trial is different
        nest.SetKernelStatus({'rng_seed': np.random.randint(low=1, high=100000)})
        # Simulate
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
        for senderIter in senders:
            neuronSpikes[senderIter] += 1

        # Now, let's create a dataframe and fill it with the desired information
        neuronDfList = []  # We will store individual dataframes here and concatenate them later

        directTargets = nest.GetConnections(source=pop1[0], target=pop1).get('target')
        for neuronID, spikes in neuronSpikes.items():
            # Check if the neuron is the optogenetic target
            target = int(neuronID == targetId)
            # Check if the neuron receives connections from the target
            directConn = int(neuronID in directTargets)
            # Check if the neuron is in same population as the target
            inNetwork = int(neuronID in pop1)
            # Create a dataframe for this neuron and add it to the list
            neuronDf = pd.DataFrame({'Trial': [trial],
                                     'Stimulation': stimulation,
                                     'NeuronID': [neuronID],
                                     'Target': [target],
                                     'DirectConn': [directConn],
                                     'SamePop': [inNetwork],
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
             'nNeurons1': nNeurons1,
             'nNeurons2': nNeurons2,
             'inDegreeConnect': inDegreeConnect,
             'wSyn': wSyn,
             'noiseStrength': noiseStrength,
             'noiseRate': noiseRate,
             'stimCurrent': stimCurrent,
             'neuronType': neuronType,
             'I_e': I_e,
             'g_L': g_L}

if plotOrSave == 'save':
    # Save param dict
    with open('../data/1_excitatory_spaceless_params.json', 'w') as fp:
        json.dump(paramDict, fp)
    # Save the dataframe
    experimentDf.to_csv('../data/1_excitatory_spaceless.csv', index=False)
elif plotOrSave == 'plot':
    #############
    # Plot
    #############
    trialDf = trialDf[trialDf['Stimulation']]
    dmm = multimeter.get("events")['V_m']
    ts = multimeter.get("events")["times"]
    senders = spikeRecorder.get('events')['senders']
    spikeTimes = spikeRecorder.get('events')['times']

    plt.subplot(1,3,1)
    plt.plot(ts, dmm)

    plt.subplot(1,3,2)
    plt.scatter(spikeTimes, senders, s=1)
    tgtTimes = spikeTimes[senders == targetId]
    plt.scatter(tgtTimes, np.repeat(targetId, len(tgtTimes)), s=1, c='r')

    plt.subplot(1,3,3)
    plt.hist(trialDf['Activation'])

    plt.show()

    directConnBool = trialDf['DirectConn'] == 1
    samePopBool = (trialDf['SamePop'] - trialDf['DirectConn'] - trialDf['Target']) == 1
    diffPopBool = (trialDf['SamePop'] == 0)
    print(f"Total mean: {np.mean(trialDf['Activation'])}")
    print(f"Direct: {np.mean(trialDf['Activation'][directConnBool])}")
    print(f"InNet: {np.mean(trialDf['Activation'][samePopBool])}")
    print(f"OutNet: {np.mean(trialDf['Activation'][diffPopBool])}")

