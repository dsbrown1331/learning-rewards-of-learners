import numpy as np
import argparse

env_name = 'beamrider'
returns = []
for rep in range(5):
    if rep == 0:
        checkpoint = '09800'
    else:
        checkpoint = '15000'
        
    f = open(env_name + "_" + checkpoint + "_" + str(rep) + "eval.txt")
    for line in f:
        returns.append(float(line.strip()))
print(np.mean(returns))
    
