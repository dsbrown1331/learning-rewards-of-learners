## Clone appropriately

```
git submodule update --init --recursive
```

You should see the contents of `baselines`

## Generate Trajectory and Videos

```
./run.py --env_id BreakoutNoFrameskip-v4 --model_path ./models/breakout/checkpoints/03600 --record_video
```

Replace the arguments as you want. Currently models for each 100 learning steps (upto 3600 learning steps) are uploaded.

You can omit the last flag `--record_video`. When it is turned on, then the videos will be recorded under the current directory.
