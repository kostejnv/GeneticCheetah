# GeneticCheetah

Final project for the *Genetic Algorithms in Robotics* course at university.

A genetic algorithm evolves a small feed-forward neural network that controls the
[MuJoCo HalfCheetah](https://gymnasium.farama.org/environments/mujoco/half_cheetah/)
agent — no gradients, no backprop. The genome is the flat vector of network weights
and biases; fitness is the cumulative reward the network earns when its outputs are
fed to the simulator as joint torques.

## How it works

- **Arena** (`arena.py`) — wraps the `HalfCheetah-v4` Gymnasium environment and runs
  an episode for a given behaviour function, returning the average reward over
  several repetitions. Optionally records video to `export/`.
- **CheetahLab** (`cheetah_laboratory.py`) — defines the genome layout. The
  `ClassicNNCheetahLab` takes a list describing the network architecture
  (e.g. `[17, 8, 6]` for a single hidden layer of width 8) and converts a flat
  genome into a stack of weight matrices. Hidden layers use a clipped logistic
  sigmoid; the output layer uses `tanh` so actions land in `[-1, 1]`.
- **Evolution** (`evolution.py`) — drives the GA via [PyGAD](https://pygad.readthedocs.io/),
  with parallel fitness evaluation and an optional Gaussian mutation operator.

## Files

| File | Purpose |
|---|---|
| `arena.py` | HalfCheetah simulation wrapper |
| `cheetah_laboratory.py` | Genome ↔ neural network mapping |
| `evolution.py` | PyGAD-based genetic algorithm |
| `experiments.py` | Single training run with fixed hyperparameters |
| `optimize_hyperparams.py` | Optuna sweep over GA + architecture hyperparameters |
| `experiments.ipynb` | Result analysis: heatmaps, boxplots, best-run inspection |

## Running

```bash
pip install gymnasium[mujoco] pygad optuna numpy pandas plotly
python experiments.py            # one training run
python optimize_hyperparams.py   # 150-trial Optuna sweep (slow)
```

Optuna trial results are cached at `data/cheetah_hyperparams.csv`; subsequent runs
reuse the cached frame instead of re-running the sweep.
