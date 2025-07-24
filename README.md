# RLtoolbox Components

A Python package providing modular components for reinforcement learning experiments, designed to work with the RLtoolbox framework.

## Overview

RLtoolbox Components offers a collection of pre-built, configurable modules for reinforcement learning research and experimentation. The package follows a component-based architecture that allows for easy composition and configuration of RL experiments through JSON configuration files.

## Features

- **Neural Networks**: Multi-layer perceptron actor-critic networks
- **Algorithms**: PPO (Proximal Policy Optimization) implementation
- **Environments**: Gymnasium environment wrappers
- **Loggers**: Console logging utilities
- **Configuration-driven**: JSON-based experiment configuration
- **Modular Design**: Easy to extend and customize components

## Installation

### From Source

```bash
git clone https://github.com/username/RLtoolbox-components.git
cd RLtoolbox-components
pip install -e .
```

### Dependencies

This package requires:
- Python 3.7+
- PyTorch
- NumPy
- RLtoolbox (base framework)
- Gymnasium (for environment components)

## Quick Start

### Basic Usage

The package is designed to be used through JSON configuration files that define the components and their relationships:

```bash
python -m rltoolbox.cli train example.json
```
