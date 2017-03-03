import numpy as np
import os
import csv

#TODO: Training by counting, show emission, transition, start probabilities for model
#TODO: Use viterbi decoding for prediction, do a 10-fold experiement (leave on out each time)
# Viterbi decoding can be from project 2, should work here, as training-by-counting gives the probablities
num_of_states = 3
index_to_states = {0: "i", 1: "M", 2: "o"}
states_to_index = {"i": 0, "M": 1, "o": 2}

observable_to_index = {'A':0,'C':1,'E':2,'D':3,'G':4,'F':5,'I':6,'H':7,'K':8,'M':9,'L':10,'N':11,'Q':12,'P':13,'S':14,'R':15,'T':16,'W':17,'V':18,'Y':19}
observables = []

sequences = []
sequence_annotations = []
sequence_names = []
spot = 1
data_file_num = 0
with open(os.path.join("data", "set160." + str(data_file_num) + ".labels.txt")) as sequence_data:
    seq_reader = csv.reader(sequence_data, delimiter=" ")
    for line in sequence_data:
        if ">" in line:
            sequence_names.append(line.split(">")[1].strip())
            spot = 1
        elif spot == 1:
            sequences.append(line.strip())
            spot = 0
        elif "#" in line:
            sequence_annotations.append(line.split("#")[1].strip())
    data_file_num += 1

print(sequence_names)
print(sequences)
print(sequence_annotations)

##################### FINISH READING IN DATA ###################################
##################### CALCUATE PROBABLITIES ####################################
# Transition: Prob each states changes to the next one
# Emission: Prob each state gives a specific amino acid
# Initial: Probability of starting with that one
emissions_table = np.ones([len(index_to_states), len(observable_to_index)])
transitions_table = np.ones([len(index_to_states), len(index_to_states)])
pi_table = np.ones(len(index_to_states))
# Starts it with the "pseudo" count, every state has 1 right now
# As go through the sequences and files, keep the same tables for the training data,
# Final table values are after all sets are read in, and only used for the last set

# Go through every sequence, matching it up with the annotation, counting the states
for index, seq in enumerate(sequences):
    sequence_index = [observable_to_index[observation] for observation in seq]
    annotation = [states_to_index[c] for c in sequence_annotations[index]]
    for j, amino_acid in enumerate(sequence_index):
        if j == 0:
            # First one, so count in pi_table
            pi_table[amino_acid] += 1
        emissions_table[annotation[j]][amino_acid] += 1
        if j > 0:
            # Can get a transition from past value to current one
            transitions_table[annotation[j-1]][annotation[j]] += 1


##################### END CALCULATE PROB #######################################

decoding = False

if decoding:
    # Viterbi decoding, taken from project 2
    for index, item in enumerate(sequences):
        seq1 = item
        print("The test sequence: " + str(sequence_names[index]))
        print(seq1)

        # remapping of sequence to a list of numbers, so it can be easily
        # accessed as an index in an array
        sequence_index = [observable_to_index[observation] for observation in seq1]

        # Creation of the ω table:
        # Setting all the values to -inf for if the value is 0 in log space, should be -inf
        omega_table = [len(pi_table) * [float("-inf")] for index in range(len(sequence_index))]

        # Calculation of the 1st column of the ω table , which is the basis for the recursion of the algorithm:
        for j in range(len(pi_table)):
            omega_table[0][j] = np.log(pi_table[j]) + np.log(emissions_table[j][sequence_index[0]])

        # Calculation of the rest of the ω table:
        for n in range(1,len(sequence_index)):
            for k in range(len(states_to_index)):
                transition_value = float("-inf")
                if emissions_table[k][sequence_index[n]]!=0.0 and emissions_table[k][sequence_index[n]] != float("-inf"):
                    for j in range(len(states_to_index)):
                        if transitions_table[j][k]!=0.0 and transitions_table[j][k] != float("-inf") :
                            if omega_table[n-1][j] + np.log(transitions_table[j][k]) > transition_value:
                                transition_value = omega_table[n-1][j] + np.log(transitions_table[j][k])
                    omega_table[n][k] = np.log(emissions_table[k][sequence_index[n]]) + transition_value

        # Creation of the Z* sequence as a list, and calculation of its last element
        Z = []
        list2 = [] # list to keep the 3 elements of the last column of the omega table and
        # find their maximum
        for i in range(len(sequence_index)):
            Z.append(sequence_index[i])
            # Here we fill the Z list with the same letters of the test sequence.
            # But we could fill it with any other characters. Only its length must be the same with this of the test sequence.
            # Now just need to get the max value from the last row in omega. Need that to start
            # the backtracking
        for j in range(len(states_to_index)):
            list2.append(omega_table[len(omega_table) - 1][j])
        for j in range(len(states_to_index)):
            if omega_table[len(omega_table) - 1][j]==max(list2):
                Z[len(Z)-1]=j
        # Calculation of the rest of the Z sequence backwards, from the right to the left
        for n in range(len(Z)-1,0,-1):
            for k in range(len(states_to_index)):
                if transitions_table[k][Z[n]] != 0.0:
                    if emissions_table[Z[n]][sequence_index[n]] != 0.0:
                        if omega_table[n-1][k] + np.log(transitions_table[k][Z[n]]) + np.log(emissions_table[Z[n]][sequence_index[n]]) == omega_table[n][Z[n]]:
                            Z[n-1] = k
        print("The Z* sequence of the hidden states:\n")
        print("".join([index_to_states[c] for c in Z])) # the Z* sequence printed as a string
        print("Log value:" + str(omega_table[-1][Z[-1]]))
        print("Finished")