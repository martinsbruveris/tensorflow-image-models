from tfimm.train import (
    ClassificationConfig,
    ExperimentConfig,
    ModelConfig,
    TFDSConfig,
    TrainerConfig,
    run,
)


def main_with_python_config():
    """Start experiment with a config defined by python code."""
    cfg = ExperimentConfig(
        trainer=TrainerConfig(
            nb_epochs=3,
            nb_samples_per_epoch=640,
            display_loss_every_it=5,
            ckpt_dir="/tmp/exp_cifar10",
        ),
        trainer_class="SingleGPUTrainer",
        problem=ClassificationConfig(
            model=ModelConfig(
                model_name="resnet18",
                pretrained="",
                input_size=(64, 64),
                nb_channels=3,
                nb_classes=10,
            ),
            model_class="ModelFactory",
            binary_loss=False,
            weight_decay=0.01,
            lr=0.01,
            mixed_precision=False,
        ),
        problem_class="ClassificationProblem",
        train_dataset=TFDSConfig(
            dataset_name="cifar10",
            split="train",
            input_size=(64, 64),
            batch_size=32,
            repeat=True,
            shuffle=True,
            nb_samples=-1,
            dtype="float32",
        ),
        train_dataset_class="TFDSWrapper",
        val_dataset=TFDSConfig(
            dataset_name="cifar10",
            split="test",
            input_size=(64, 64),
            batch_size=32,
            repeat=False,
            shuffle=False,
            nb_samples=320,
            dtype="float32",
        ),
        val_dataset_class="TFDSWrapper",
        log_wandb=False,
    )

    run(cfg, parse_args=False)


def main_with_cfg_file():
    """Start experiment with a config defined by a config file."""
    cfg = {"cfg_file": "tfimm/train/examples/config.yaml"}
    run(cfg, parse_args=False)


if __name__ == "__main__":
    main_with_python_config()
    main_with_cfg_file()
