# learning-rewards-of-learners

## Preference Learning

### Train

```
python preference_learning.py --env_id Swimmer-v2 --env_type mujoco --learners_path ./learner/models/swimmer/checkpoints/
```

### Eval

```
python preference_learning.py --env_id Swimmer-v2 --env_type mujoco --learners_path ./learner/models/swimmer/checkpoints/ --eval --D 10000
```
