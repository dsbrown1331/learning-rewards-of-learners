## Clone appropriately

```
git submodule update --init --recursive
```

You should see the contents of `baselines`

## Train

```
OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=/home/wonjoon/workspace/LfL_new/learner/models/log/reacher python -m baselines.run --alg=ppo2 --env=Reacher-v2 --save_interval=20
```

## Generate Trajectory and Videos

First, download the pretrained models [link](), and extract under the `models` directory.

```
./run_test.py --env_id BreakoutNoFrameskip-v4 --env_type atari --model_path ./models/breakout/checkpoints/03600 --record_video
```
or,
```
./run_test.py --env_id Reacher-v2 --env_type mujoco --model_path ./models/reacher/checkpoints/01000 --record_video
```


Replace the arguments as you want. Currently models for each 100 learning steps (upto 3600 learning steps) are uploaded.

You can omit the last flag `--record_video`. When it is turned on, then the videos will be recorded under the current directory.
