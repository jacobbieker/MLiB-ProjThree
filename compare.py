import os
import subprocess

# Run the comparing thing, and then get the mean and variance from it for the report

for number in range(0, 10):
    # Running 3 comparison
    process = subprocess.run(["python3", "compare_tm_pred.py", os.path.join("data", "set160." + str(number) + ".labels.txt"),
                    "set160." + str(number) + ".3prediction.txt"])
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    processTwo = subprocess.run(["python3", "compare_tm_pred.py", os.path.join("data", "set160.all.txt"),
                              "set160.3all.txt"])
    print(number)
    print("+++++++++++++++++_______________________+++++++++++++++++++++_____________________+++++++++++++++")
    # Running 4 comparison
    processOne = subprocess.run(["python3", "compare_tm_pred.py", os.path.join("data", "set160." + str(number) + ".labels.txt"),
                              "set160." + str(number) + ".4prediction.txt"])
    print("_______________________________________________________________________________________________")
    processThree = subprocess.run(["python3", "compare_tm_pred.py", os.path.join("data", "set160.all.txt"),
                                 "set160.4all.txt"])