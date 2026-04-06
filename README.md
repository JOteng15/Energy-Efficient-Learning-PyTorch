# Energy-Aware Training Framework
Currently working this project which is inspired by these two papers (Pache & van Rossum (2023): Metabolic cost of synaptic plasticity in biological systems and Progressive Data Dropout: An Embarrassingly Simple Approach to Train Faster), which is essentially a unified framework for energy-efficient deep learning that jointly optimises **which data** and **which weights** are updated during training, using metabolic energy as the governing design constraint.

## Core Components

### 1. Error-Driven Updates
Skip backward passes for samples the model already classifies correctly. Inspired by biological learning where synaptic updates only occur when prediction errors signal the need for change.

### 2. Progressive Data Dropout
Progressively remove "easy" samples from training across epochs. Samples that the model consistently classifies correctly are dropped first, following a configurable schedule (linear, exponential, or cosine).

### 3. Selective Synaptic Updates
After backpropagation, only update the most informative weights. Three strategies are available:
- **Magnitude-based**: Keep gradients with the largest absolute values
- **Relative**: Keep gradients that are large relative to current weight magnitude
- **Layer-adaptive**: Allocate more updates to layers with higher gradient variance

### 4. Metabolic Energy Model
A principled energy accounting system that assigns costs to:
- Forward passes (baseline cost)
- Backward passes (~3x forward cost)
- Weight updates (proportional to number of parameters updated)



## Architecture

```
energy_aware_framework.py
├── SimpleMLP / SimpleCNN          # Network architectures
├── MetabolicEnergyModel           # Energy cost accounting
├── ProgressiveDataDropout         # Adaptive dataset reduction
├── SelectiveSynapticUpdater       # Sparse gradient updates
├── EnergyAwareTrainer             # Unified training loop
├── run_experiments()              # Comparative experiments
└── plot_results()                 # Visualisation
```
