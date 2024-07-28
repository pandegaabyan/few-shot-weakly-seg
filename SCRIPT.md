# Run Fit Test

```bash
python run.py --help
python run.py -d
python run.py -d -c learn/ref_ckpt_path {exp/run/last.ckpt}
python run.py -d -r -c learn/exp_name {exp} -c learn/run {run} -c learn/num_epochs 10
python run.py -d -m fit
python run.py -d -m fit -c learn/ref_ckpt_path {exp/run/last.ckpt}
python run.py -d -m fit -r -c learn/exp_name {exp} -c learn/run {run} -c learn/num_epochs 10
python run.py -d -m test -c learn/exp_name {exp} -c learn/run_name {run}
python run.py -d -m test -c learn/exp_name {exp} -c learn/run_name {run} -c data/batch_size 1
python run.py -d -m test -c learn/ref_ckpt_path {exp/run/last.ckpt}
```

# Run Study

```bash
python run.py -d -m study -oc num_trials 3
python run.py -d -m study -oc num_trials 2 -oc num_folds 2
python run.py -d -m study -c log/configuration True -c log/table True -oc num_trials 2
python run.py -d -m study -r -oc num_trials 1 -oc study_name {study}
```

# Clean

```bash

```

# Optuna

```bash
optuna-dashboard sqlite:///logs/optuna_dummy.sqlite3
```
