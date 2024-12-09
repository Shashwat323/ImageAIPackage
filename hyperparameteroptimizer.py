import torch
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
import run
import models
import loader
import torch.nn as nn


def objective(config):  # ①
    train_loader, test_loader, val_loader = loader.get_dataloaders(batch_size=64, root="D:\\Other\\Repos\\ImageAIPackage",
                                                       dataset_type="cifar10")  # Load some data
    model = models.ModularSimpleCNN(num_classes=10, downsample=8, in_channels=3, hidden_neurons=config["hidden_neurons"],
                                    num_conv_layers=config["num_conv_layers"], expansion=config["expansion"],
                                    dropout=config["dropout"]) # Create a PyTorch conv net
    for param in model.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.CrossEntropyLoss()
    while True:
        run.train(train_loader, model, loss_fn, optimizer)  # Train the model
        acc = run.test(test_loader, model, loss_fn)  # Compute test accuracy
        train.report({"mean_accuracy": acc})  # Report to Tune



if __name__ == "__main__":
    search_space = {"hidden_neurons": tune.randint(128, 2048), "num_conv_layers": tune.randint(1, 4),
                    "expansion": tune.randint(2,4), "dropout": tune.uniform(0.0, 0.5),
                    "lr": tune.uniform(1e-5, 1e-2)}
    algo = OptunaSearch()  # ②

    tuner = tune.Tuner(  # ③
        objective,
        tune_config=tune.TuneConfig(
            metric="mean_accuracy",
            mode="max",
            search_alg=algo,
        ),
        run_config=train.RunConfig(
            stop={"training_iteration": 5},
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)