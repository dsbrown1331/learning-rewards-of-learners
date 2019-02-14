#! /usr/bin/env bash

#already ran seaquest.. Add back in later
for i in beamrider breakout enduro hero mspacman pong qbert spaceinvaders videopinball; do 
    echo "condor_submit $i_jobsubmit"
    condor_submit $i_jobsubmit
done
