import numpy as np
import pandas as pd
import nest
import nest.voltage_trace
import matplotlib.pyplot as plt


# Function to make 2 by 2 connectivity matrices
def connectivity_mat_2pop(withinPop, betweenPop):
    """ For a network with two populations of neurons, make
    a connectivity matrix with specified within-group and
    between-group connectivities

    Arguments:
    ---------------------
      - withinPop: Connectivity strenght of neurons with same population
      - betweenPop: Connectivity strenght of neurons with different population

    Output:
    ---------------------
      - connectivityMat: 2x2 connectivity matrix with the within-population
      values in the diagonal and off-population values in the off-diagonal
    """
    withinPopMat = np.eye(2) * withinPop
    betweenPopMat = (1-np.eye(2)) * betweenPop
    connectivityMat = withinPopMat + betweenPopMat
    return connectivityMat



def implement_EI_connectivity(popE, popI, connMatE2E, connMatE2I,
                              connMatI2E, connMatI2I, weightSynE, weightSynI):
    """
      Connect the different neuron populations given in the
      lists popE and popI. The connectivity is as indicated by the
      connectivity matrices, with rows indicating sender and columns
      receiver
   
      Arguments:
      ---------------------
        - popE: List containing the populations of excitatory neurons
        - popI: List containing the populations of inhibitory neurons
        - connMatE2E: Matrix with connectivity degree from E to E
          across populations. Rows are senders, columns receivers
        - connMatE2I: Matrix with connectivity degree from E to I
          across populations. Rows are senders, columns receivers
        - connMatI2E: Matrix with connectivity degree from I to E
          across populations. Rows are senders, columns receivers
        - connMatI2I: Matrix with connectivity degree from I to I
          across populations. Rows are senders, columns receivers
        - weightSynE: Weight of synapses from E to E
        - weightSynI: Weight of synapses from I to I
    """
    nPopulations = len(popE)  # Number of populations
    # Create the synapse dictionaries
    synE = {'weight': weightSynE}
    synI = {'weight': weightSynI}  # inhibitory weight
    for i in range(nPopulations):
        for j in range(nPopulations):
            # Get the connection degrees from the matrices
            degE2E = int(connMatE2E[i, j])
            degE2I = int(connMatE2I[i, j])
            degI2E = int(connMatI2E[i, j])
            degI2I = int(connMatI2I[i, j])
            # Create the connection dictionaries
            connE2E = {'rule': 'fixed_indegree', 'indegree': degE2E}
            connE2I = {'rule': 'fixed_indegree', 'indegree': degE2I}
            connI2E = {'rule': 'fixed_indegree', 'indegree': degI2E}
            connI2I = {'rule': 'fixed_indegree', 'indegree': degI2I}
            # Connect the populations
            nest.Connect(popE[i], popE[j], connE2E, synE)
            nest.Connect(popE[i], popI[j], connE2I, synE)
            nest.Connect(popI[i], popE[j], connI2E, synI)
            nest.Connect(popI[i], popI[j], connI2I, synI)

