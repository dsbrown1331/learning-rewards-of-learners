#!/usr/bin/env bash
cd ..
source ~/.bashrc
conda activate deeplearning
python evaluateLearnedPolicy_condor.py --env_name $1 --checkpoint $2 --rep 0
python evaluateLearnedPolicy_condor.py --env_name $1 --checkpoint $2 --rep 1
python evaluateLearnedPolicy_condor.py --env_name $1 --checkpoint $2 --rep 2
python evaluateLearnedPolicy_condor.py --env_name $1 --checkpoint $2 --rep 3
python evaluateLearnedPolicy_condor.py --env_name $1 --checkpoint $2 --rep 4
