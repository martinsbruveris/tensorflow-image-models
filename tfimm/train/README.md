# Training framework

## Overview

The training framework consists of the following parts
- Trainer class, defined in `trainer.py`. The role of the trainer class is to perform
  high-level orchestration of the training process: it contains the main training
  loop, keep track of epochs, checkpoint saving, validation and logging to W&B. It
  relies on the problem class to implement the logic for all these operations.
- Problem class, derived from `ProblemBase`. See `ClassificationProblem` for an example.
  The problem class contains all the problem-specific code: how to create the model,
  the optimizer, compute the loss function, perform one training step, and perform
  validation.
- Dataset classes should provide the ability to iterate over data. See `TFDSWrapper`
  for a simple wrapper around `tensorflow_datasets` datasets, integrating them into
  the configuration system of the framework.
- A configuration containing all the information needed to reproduce an experiment. See
  documentation below.
- An orchestrator, which combines all the pieces and calls `trainer.train()`. See 
  `run(cfg)` in `train.py`. It creates the trainer, problem and dataset objects as
  specified by the configuration and starts the training loop.

## Configuration

One of the strengths of the framework is the flexible configuration system. We build
upon the following principles

- Each experiment is fully defined by the configuration. After creating the 
  configuration, either directly in the code, via a config file or via command line
  parameters, the training is launched via `tfimm.train.run(cfg)`.
- A configuration object is a dataclass, whose fields are either `int`, `float`, `bool`,
  `str`, `tuple` or another dataclass satifying the same format.
- All configuration parameters can be set either in python code, via a config file or
  via command line parameters. The latter makes all values accessible for W&B 
  hyperparameter sweeps. Configuration parameters are logged to W&B at the start of the 
  training and stored as a file in the checkpoint directory.

There are four sources of configuration values. These are in order of increasing
priority:
- Default values defined in the dataclass
- Values defined when creating the config in python code
- Values stored in the config file
- Values passed as command line parameters

### Class-specific configurations

To deal with different problem classes, dataset classes, etc., we use a type registry
system. For example, the `ClassificationProblem` has the following structure

```python
@dataclass
class ClassificationConfig:
    nb_classes: int

@cfg_serializable
class ClassificationProblem:
    cfg_class = ClassificationConfig

    def __init__(self, cfg: ClassificationConfig):
        ...
```

This is combined with the following `ExperimentConfig`

```python
@dataclass
class ExperimentConfig:
    problem: Any
    problem_class: str

def main():
    cfg = ExperimentConfig(
      problem=ClassificationConfig(nb_classes=10),
      problem_class="ClassificationProblem"
    )
    run(cfg)
```

The function `run(cfg)` uses the information in `problem_class` to construct the 
correct problem class, in this case `ClassificationProblem` using the configuration
provided by the `problem` field. The `@cfg_serializable` decorator registers the
`ClassificationProblem` class to the type registry and makes it available for 
constuction

```python
from tfimm.train.registry import get_class

def run(cfg: ExperimentConfig):
    problem = get_class(cfg.problem_class)(cfg.problem)
```

Configuration classes can be arbitrarily nested. E.g., the `ClassificationConfig` could
contain a `ModelConfig` sub-config with information, which model to construct for 
training.

```python
@dataclass
class ClassificationConfig:
    nb_classes: int
    model: Any
    model_class: str
```

Note that configuration classes are not stand-alone. By this we mean that, e.g, 
`ClassificationConfig` is the configuration for the `ClassificationProblem` class.
That is why `problem_class="ClassificationProblem"`. We specify, which problem class
to create for the experiment and then the `cfg_class` field in `ClassificationProblem`
provides the link to `ClassificationConfig`.

### Configurations as dictionaries

Each nested configuration can be converted to either a nested dictionary structure (for
saving as yaml) or a flat dictionary (for argument parsing). Because of the type 
registry (`@cfg_serializable` decorator) we can also reconstruct the python dataclasses.

```python
# Python structure
cfg = ExperimentConfig(
    problem=ClassificationConfig(nb_classes=10),
    problem_class="ClassificationProblem",
)

# Nested dictionaries
nested_cfg = {
  "problem": {"nb_classes": 10},
  "problem_class": "ClassificationProblem",
}

# Flat dictionaries
flat_cfg = {
  "problem.nb_classes": 10,
  "problem_class": "ClassificationProblem",
}
```

The conversion happens using the functions `to_dict_format()`, `to_cls_format()`,
`flat_to_deep()` and `deep_to_flat()`.

```python
assert nested_cfg == to_dict_format(cfg)

# Note that we cannot automatically reconstruct the class `ExperimentConfig`, because
# the information is not present in the nested dictionary.
assert cfg == ExperimentConfig(**to_cls_format(nested_cfg))

assert flat_cfg == deep_to_flat(nested_cfg)
assert nested_cfg == flat_to_deep(flat_cfg)
```

### Configuration discovery and default arguments

When parsing command line arguments we start with an expected structure, e.g.

```python
def main():
    cfg = ExperimentConfig(problem=None, problem_class="")
    parsed_cfg = parse_args(cfg)
```

We tell `parse_args()` that we expect the user to provide an argument `--problem_class`,
and the value of that argument will determine via registry lookup, which further 
parameters to expect. E.g., if the user invokes the script via

```shell
python script.py --problem_class=ClassificationProblem ...
```

this will tell the parser to look up the config for `ClassificationProblem` and hence
to expect the parameter `--problem.nb_classes`. If default values for any of the
discovered parameters are present either in the definition of `ClassificationConfig`
or the `cfg` object passed to `parse_args()`, these will be applied.