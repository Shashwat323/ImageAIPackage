from datetime import datetime

import torch
import torch.nn as nn
from loader import get_dataloaders
from models.models import get_model
from tqdm import tqdm

import numpy as np
import argparse

# train one epoch
def train(train_loader, model, loss_fn, optimizer, use_progress_bar = True):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    total = 0
    correct = 0
    if use_progress_bar:
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    else:
        progress_bar = enumerate(train_loader)
    # Train
    model.train()
    for batch, (X, y) in progress_bar:
        X = X.to(device)
        y = torch.tensor(y, dtype=torch.long).to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accuracy calculation
        _, predicted = torch.max(pred.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

        if use_progress_bar:
            progress_bar.set_postfix(loss=loss.item(), accuracy=100 * correct / total)


# validate and return mae loss
def validate(val_loader, model, loss_fn, use_progress_bar=False):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    total = 0
    correct = 0
    if use_progress_bar:
        progress_bar = tqdm(enumerate(val_loader), total=len(val_loader))
    else:
        progress_bar = enumerate(val_loader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (X, y) in progress_bar:
            X = X.to(device)
            y = torch.tensor(y, dtype=torch.long).to(device)

            pred = model(X)

            loss = loss_fn(pred, y)
            val_loss += loss.item()

            # Accuracy calculation
            _, predicted = torch.max(pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            if use_progress_bar:
                progress_bar.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

    val_loss /= len(val_loader)

    print(f"val loss: {val_loss}")
    return val_loss



# Test and return loss
def test(test_loader, model, loss_fn, use_progress_bar=False):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    total = 0
    correct = 0
    if use_progress_bar:
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
    else:
        progress_bar = enumerate(test_loader)

    # Test
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (X, y) in progress_bar:
            X = X.to(device)
            y = torch.tensor(y, dtype=torch.long).to(device)

            pred = model(X)

            loss = loss_fn(pred, y)
            test_loss += loss.item()

            # Accuracy calculation
            _, predicted = torch.max(pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            if use_progress_bar:
                progress_bar.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

    test_loss /= len(test_loader)

    print(f"test loss: {test_loss:>7f}")
    return test_loss

def generate_timestamp_string() -> str:
    """
    Generate a string based on the current date and time.

    Returns:
    - str : A string representing the current date and time in the format 'YYYYMMDD_HHMMSS'.
    """
    # Get the current date and time
    now = datetime.now()

    # Format the datetime object into a string
    timestamp_string = now.strftime('%Y%m%d_%H%M%S')

    return timestamp_string

# helper class for early stopping
class EarlyStopping:
    def __init__(self, patience=3, verbose=False, delta=0, root = ""):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.root = root

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.root + f'/weights/{generate_timestamp_string()}.pt')  # save checkpoint
        self.val_loss_min = val_loss


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="", help='set to root directory (where ImageAIPackage is located)')
    parser.add_argument('--dataset', type=str, default="")
    parser.add_argument('--model', type=str, default="")
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for (default: 10)')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--model_path', type=str, default="", help='path to existing model, else leave blank to train new one')
    args = parser.parse_args()

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=args.batch_size, root=args.root, dataset_type=args.dataset,
                                                            augmentations=20)
    model = get_model(model_type=args.model).float().to(device)
    if args.model_path != "":
        model.load_state_dict(torch.load(args.model_path))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    epochs = args.epochs
    early_stopping = EarlyStopping(patience=3, verbose=True, delta=0, root=args.root)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)
        val_loss = validate(test_loader, model, loss_fn)
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    #model.load_state_dict(torch.load(args.root + f'/weights/{generate_timestamp_string()}.pt'))
    #test(test_loader, model, loss_fn)

    print("Done!")