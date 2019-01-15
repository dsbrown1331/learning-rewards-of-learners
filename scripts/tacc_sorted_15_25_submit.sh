#! /usr/bin/env bash

for i in beamrider breakout enduro hero mspacman pong qbert seaquest spaceinvaders videopinball; do 
    echo "Submitting job for $i"
    sbatch sorted_$i.slurm
done
