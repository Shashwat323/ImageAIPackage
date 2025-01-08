import argparse

import torch
from ray import tune, train
from ray.tune.search.optuna import OptunaSearch

import run
from models import adjustibleresnet
import loader
import torch.nn as nn

class Trainer:
    """
    A trainer for a given dataset using adjustable ResNet and hyperparameter tuning.
    """

    def __init__(self, root="D:\\Other\\Repos\\ImageAIPackage", batch_size=64, fraction=1.0, use_progress_bar=True):
        """
        Initialize the trainer with default or user-provided configurations.
        """
        self.root = root
        self.batch_size = batch_size
        self.fraction = fraction
        self.use_progress_bar = use_progress_bar

    def get_device(self):
        """
        Detect the appropriate device (GPU or CPU) for training.
        """
        return (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    def objective(self, config):
        """
        Objective function for training and optimizing the model. This function will be called by Tune.
        """
        device = self.get_device()
        train_loader, test_loader, val_loader = loader.get_dataloaders(
            batch_size=self.batch_size,
            root=self.root,
            dataset_type="cifar10",
            augmentations=config["augmentations"],
            fraction=self.fraction
        )
        model = adjustibleresnet.ResNet50(
            image_channels=3, num_classes=10,
            dropout=config["dropout"], initial_out=config["initial_out"]
        )
        model.to(device)
        for param in model.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        loss_fn = nn.CrossEntropyLoss()

        while True:
            # Training phase
            run.train(train_loader, model, loss_fn, optimizer, use_progress_bar=self.use_progress_bar)

            # Validation/Test phase
            acc = run.test(test_loader, model, loss_fn)

            # Report to the tuning framework
            train.report({"mean_loss": acc})

    def tune_model(self, num_samples=100, iterations=5, search_space=None):
        """
        Tune the model using Ray Tune with the given hyperparameter search space.
        """
        if search_space is None:
            search_space = {
                "initial_out": tune.randint(32, 128),
                "dropout": tune.uniform(0.2, 0.5),
                "augmentations": tune.randint(5, 20),
                "lr": tune.loguniform(1e-5, 1e-2),
            }

        algo = OptunaSearch(metric="mean_loss", mode="min")

        trainable_with_gpu = tune.with_resources(self.objective, {"gpu": 1})

        tuner = tune.Tuner(
            trainable_with_gpu,
            tune_config=tune.TuneConfig(
                search_alg=algo,
                num_samples=num_samples,
            ),
            run_config=train.RunConfig(stop={"training_iteration": iterations}),
            param_space=search_space,
        )

        return tuner.fit()


def main(root, batch_size, fraction, use_progress_bar='True', search_space=None, num_samples = 20, iterations=3):

    # Initialize trainer with parsed arguments
    trainer = Trainer(
        root=root,
        batch_size=batch_size,
        fraction=fraction,
        use_progress_bar=(use_progress_bar == 'True')
    )

    # Define the search space and start tuning
    if search_space is None:
        search_space = {
            "initial_out": tune.randint(32, 128),
            "dropout": tune.uniform(0.2, 0.5),
            "augmentations": tune.randint(5, 20),
            "lr": tune.loguniform(1e-5, 1e-2),
        }

    results = trainer.tune_model(search_space=search_space, num_samples=num_samples, iterations=iterations)
    return results.get_best_result(metric="mean_loss", mode="min").config


if __name__ == "__main__":
    # Parse arguments for user customization
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="D:\\Other\\Repos\\ImageAIPackage",
                        help='Root directory (where ImageAIPackage is located)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for loading data')
    parser.add_argument('--fraction', type=float, default=1.0, help='Fraction of the dataset to use')
    parser.add_argument('--use_progress_bar', type=str, default='True', help='Set to False to disable progress bar')
    args = parser.parse_args()
    main(args.root, args.batch_size, args.fraction, args.use_progress_bar)