import numpy as np
import os

# Convert lower m to M in end result
index_to_states = {1: "i", 2: "M", 3: "m", 4: "o"}
states_to_index = {"i": 1, "M": 2, "o": 4, "m": 3}