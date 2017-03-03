import numpy as np
import os
import csv

index_to_states = {1: "i", 2: "M", 4: "o"}
states_to_index = {"i": 1, "M": 2, "o": 4}


sequences = []
sequence_annotations = []
sequence_names = []
spot = 1
with open(os.path.join("data", "set160.0.labels.txt")) as sequence_data:
    seq_reader = csv.reader(sequence_data, delimiter=" ")
    for line in sequence_data:
        if ">" in line:
            sequence_names.append(line.split(">")[1])
            spot = 1
        elif spot == 1:
            sequences.append(line.strip())
            spot = 0
        elif "#" in line:
            sequence_annotations.append(line.split("#")[1])

print(len(sequence_names))
print(len(sequences))
print(len(sequence_annotations))
