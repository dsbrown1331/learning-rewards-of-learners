#! /usr/bin/env bash

#already ran seaquest.. Add back in later
for i in beamrider breakout enduro hero pong qbert spaceinvaders seaquest; do 
    job="$i""_jobsubmit"
    echo $job
    condor_submit $job
done
