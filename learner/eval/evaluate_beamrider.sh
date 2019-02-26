#!/usr/bin/env bash
cd ..
source ~/.bashrc
conda activate deeplearning
python evaluateLearnedPolicy_condor.py --env_name beamrider --checkpoint 09800 --rep 0
python evaluateLearnedPolicy_condor.py --env_name beamrider --checkpoint 15000 --rep 1
python evaluateLearnedPolicy_condor.py --env_name beamrider --checkpoint 15000 --rep 2
python evaluateLearnedPolicy_condor.py --env_name beamrider --checkpoint 15000 --rep 3
python evaluateLearnedPolicy_condor.py --env_name beamrider --checkpoint 15000 --rep 4
