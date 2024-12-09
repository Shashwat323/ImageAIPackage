import argparse

import torch
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
import run
import adjustibleresnet
import loader
import torch.nn as nn

root = "D:\\Other\\Repos\\ImageAIPackage"

def objective(config):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    train_loader, test_loader, val_loader = loader.get_dataloaders(batch_size=64, root=root,
                                                       dataset_type="cifar10", augmentations=config["augmentations"])  # Load some data
    model = adjustibleresnet.ResNet50(image_channels=3, num_classes=10, dropout=config["dropout"],
                                      linear_neurons=config["linear_neurons"], initial_out=config["initial_out"])
    model.to(device)
    for param in model.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.CrossEntropyLoss()
    while True:
        run.train(train_loader, model, loss_fn, optimizer, use_progress_bar=True)  # Train the model
        acc = run.test(test_loader, model, loss_fn)  # Compute test accuracy
        train.report({"mean_accuracy": acc})  # Report to Tune



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="D:\\Other\\Repos\\ImageAIPackage", help='set to root directory (where ImageAIPackage is located)')
    args = parser.parse_args()
    root = args.root

    search_space = {"linear_neurons": tune.randint(1024, 4096), "initial_out": tune.randint(32, 128),
                    "dropout": tune.uniform(0.2, 0.5), "augmentations": tune.randint(5,20),
                    "lr": tune.loguniform(1e-5, 1e-2)}
    algo = OptunaSearch()  # ②

    trainable_with_gpu = tune.with_resources(objective, {"gpu": 1})

    tuner = tune.Tuner(  # ③
        trainable_with_gpu,
        tune_config=tune.TuneConfig(
            metric="mean_accuracy",
            mode="max",
            search_alg=algo,
            num_samples=100,
        ),
        run_config=train.RunConfig(
            stop={"training_iteration": 5},
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)