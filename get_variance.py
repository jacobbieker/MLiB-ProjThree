import numpy as np
import os

ac_all = []

with open("ac_out", "r") as data:
    for line in data:
        ac_all.append(float(line.strip()))

print(ac_all)
print("Variance:")
print(np.var(ac_all))
print("Mean: ")
print(np.mean(ac_all))
