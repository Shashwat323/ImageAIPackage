import argparse

import torch
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
import run
import vit
import loader
import torch.nn as nn

root = "D:\\Other\\Repos\\ImageAIPackage"

def objective(config):  # ①
    train_loader, test_loader, val_loader = loader.get_dataloaders(batch_size=64, root=root,
                                                       dataset_type="cifar10")  # Load some data
    model = vit.get_model(hidden_neurons=config["hidden_neurons"], hidden_layers=config["hidden_layers"],
                              in_features=1280, dropout=config["dropout"]) # Create a PyTorch conv net
    for param in model.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.CrossEntropyLoss()
    while True:
        run.train(train_loader, model, loss_fn, optimizer, progress_bar=False)  # Train the model
        acc = run.test(test_loader, model, loss_fn)  # Compute test accuracy
        train.report({"mean_accuracy": acc})  # Report to Tune



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="D:\\Other\\Repos\\ImageAIPackage", help='set to root directory (where ImageAIPackage is located)')
    args = parser.parse_args()
    root = args.root

    search_space = {"hidden_neurons": tune.randint(640, 2560), "hidden_layers": tune.randint(1, 4),
                    "dropout": tune.uniform(0.2, 0.5),
                    "lr": tune.uniform(1e-5, 1e-2)}
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