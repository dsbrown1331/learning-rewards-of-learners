import numpy as np
import argparse

env_name = 'beamrider'
returns = []
for rep in range(5):
    if rep == 0:
        checkpoint = '09800'
    else:
        checkpoint = '15000'
        
    f = open("./eval/" + env_name + "_" + checkpoint + "_" + rep + "eval.txt")
    for line in f:
        returns.append(int(line.strip()))
print(np.mean(returns))
    
