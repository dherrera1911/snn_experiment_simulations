#####################################################
# 
# Simulate a network of neurons with only excitatory
# neurons but with spatial structure. One network of
# neurons is simulated, with wrapped edges. The connectivity
# between the neurons decays with their spatial distance.
# 
#####################################################


import numpy as np
import pandas as pd
import nest
import nest.voltage_trace
import matplotlib.pyplot as plt
import json
import time


start = time.time()
saveDataDir = './data/'
#############
# Simulation parameters
#############

plotOrSave = 'save' # 'plot' or 'save' the results
nThreads = 4

# Experiment parameters
nTrials = 1000
simTime = 1800.0  # Total simulation time in ms
recordDelay = 300.0

# Population size
nColumns = 37
nRows = 37

# Conectivity parameters
distMask = 0.15
pConn = 0.2
wSyn = 60
wSpaceStd = 0.12

# Noisy drive parameters
noiseStrength = 130
noiseRate = 30  # Rate of poisson noise generators

# Stimulation drive
stimCurrent = 500.0

# Neuron biophysics parameters
neuronType = 'aeif_cond_alpha'
I_e = 130.0
g_L = 30.0
t_ref = 20.0
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
connDict = {'rule': 'pairwise_bernoulli',
            'p': pConn,
            'mask': {'circular': {'radius': distMask}},
            'allow_autapses': False, 'allow_multapses': False}

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
        # Create neurons
        spatialGrid = nest.spatial.grid(shape=[nColumns, nRows],
                                        center=[0, 0], edge_wrap=True)
        pop = nest.Create('excitatory', positions=spatialGrid)
        # Create noise generators
        noise = nest.Create('poisson_generator', n=nColumns*nRows, params=noiseDict)

        #############
        # Connect neurons
        #############
        # Connect noise generators to neurons
        nest.Connect(pre=noise, post=pop, conn_spec={'rule': 'one_to_one'},
                      syn_spec={'weight': noiseStrength})
        # Connect neuron populations
        weightFun = wSyn * nest.spatial_distributions.gaussian(nest.spatial.distance,
                                                               std=wSpaceStd)
        sdict = {'weight': weightFun}
        nest.Connect(pre=pop, post=pop, conn_spec=connDict,
                     syn_spec={'weight': weightFun})

        #############
        # Create and connect recording devices
        #############
        # Create spike recorder
        spikeRecorder = nest.Create('spike_recorder')
        spikeRecorderID = spikeRecorder.get('global_id')
        nest.Connect(pre=pop, post=spikeRecorder)  # Put recorder on neurons
        # Create multimeter
        multimeter = nest.Create("multimeter")
        multimeter.set(record_from=["V_m"])
        nest.Connect(pre=multimeter, post=pop[1])  # Put recorder on neurons

        #############
        # Add excitatory device
        #############
        ctr = nest.FindCenterElement(pop)
        if stimulation:
            # Optogenetic stimulation
            opto = nest.Create('dc_generator', params={'amplitude': stimCurrent,
                                                       'start': 100.0, 'stop': simTime})
            # Connect the optogenetic device to the excitatory population
            nest.Connect(opto, ctr)

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


        # Create a dictionary that will store neuron ID as keys and
        # number of spikes as values
        neuronSpikes = {}
        # Initialize the dictionary with all neurons and set their spike counts to 0
        allNeurons = list(pop.get('global_id'))
        for neuron in allNeurons:
            neuronSpikes[neuron] = 0
        # Increase the spike count for each spike detected
        for senderIter in senders:
            neuronSpikes[senderIter] += 1

        # Create a dataframe and fill it with the desired information
        neuronDfList = []  # Store neuron dataframes to concatenate later
        centerPos = nest.GetPosition(ctr)
        directTargets = np.sort(nest.GetConnections(source=ctr, target=pop).get('target'))
        ctrID = ctr.get('global_id')
        for neuronID, spikes in neuronSpikes.items():
            # Check if the neuron is the optogenetic target
            target = int(neuronID == ctrID)
            # Check if the neuron receives connections from the target
            directConn = int(neuronID in directTargets)
            # Compute distance to target neuron
            distance = nest.Distance(ctr, pop[neuronID-1])[0]
            # Check if neuron is in range
            inRange = int(distance < wSpaceStd)
            # Get neuron position in network
            position = nest.GetPosition(pop[neuronID-1])
            # Get weight to the neuron
            if directConn == 1:  # This is the directly stimulated neuron
                neuronWeight = nest.GetConnections(source=ctr, target=pop[neuronID-1]).get('weight')
            elif neuronID == ctrID:
                neuronWeight = np.nan
            else:
                neuronWeight = 0
            # Create a dataframe for this neuron and add it to the list
            neuronDf = pd.DataFrame({'Trial': [trial],
                                     'Stimulation': stimulation,
                                     'NeuronID': [neuronID],
                                     'Target': [target],
                                     'DirectConn': [directConn],
                                     'SynW': [neuronWeight],
                                     'InRange': [inRange],
                                     'Distance': [distance],
                                     'PosX': [position[0]],
                                     'PosY': [position[1]],
                                     'Activation': [spikes]})
            neuronDfList.append(neuronDf)

        # Concatenate all the dataframes
        trialDf = pd.concat(neuronDfList, ignore_index=True)
        trialDf = trialDf.sort_values(by='NeuronID')
        trialDfList.append(trialDf)
        print(f'Trial {trial} complete')


# Concatenate the trial Df's into the full experiment Df
experimentDf = pd.concat(trialDfList, ignore_index=True)

# Make dictionary with simulation parameters
paramDict = {'nTrials': nTrials,
             'simTime': simTime,
             'recordDelay': recordDelay,
             'nColumns': nColumns,
             'nRows': nRows,
             'distMask': distMask,
             'wSpaceStd': wSpaceStd,
             'wSyn': wSyn,
             'pConn': pConn,
             'noiseStrength': noiseStrength,
             'noiseRate': noiseRate,
             'stimCurrent': stimCurrent,
             'neuronType': neuronType,
             't_ref': t_ref,
             'V_reset': V_reset,
             'I_e': I_e,
             'g_L': g_L}


end = time.time()
print(f'Time elapsed: {end-start}')

#############
# Plot
#############

if plotOrSave == 'save':
    os.makedirs(saveDataDir, exist_ok=True)
    # Save param dict
    with open(f'{saveDataDir}3_excitatory_spatial_params.json', 'w') as fp:
        json.dump(paramDict, fp)
    # Save the dataframe
    experimentDf.to_csv(f'{saveDataDir}3_excitatory_spatial.csv', index=False)

elif plotOrSave == 'plot':

    #############
    # Plot
    #############
    trialDf = trialDf[trialDf['Stimulation']]
    dmm = multimeter.get("events")['V_m']
    ts = multimeter.get("events")["times"]

    plt.subplot(2,2,1)
    plt.plot(ts, dmm)

    plt.subplot(2,2,2)
    plt.scatter(spikeTimes, senders, s=1)
    tgtTimes = spikeTimes[senders == ctrID]
    plt.scatter(tgtTimes, np.repeat(ctrID, len(tgtTimes)), s=1, c='r')

    plt.subplot(2,2,3)
    plt.hist(trialDf['Activation'])

    plt.subplot(2,2,4)
    plt.scatter(trialDf['Distance'], trialDf['Activation'], s=5, alpha=0.3)

    plt.show()

    directConnBool = trialDf['DirectConn'] == 1
    print(f"Total mean: {np.mean(trialDf['Activation'])}")
    print(f"Direct: {np.mean(trialDf['Activation'][directConnBool])}")


