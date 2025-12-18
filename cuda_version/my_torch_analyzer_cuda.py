#!/usr/bin/python3

import argparse
from typing import Optional
import cupy as cp
from Tensor import Tensor
from myModel import Model
import random
import time
from parsing import parse_line
import warnings

# @warnings.deprecated("Unused old function needed to be removed.")
def load_dataset_test(chessfile: str, batch_size: int) -> list[str]:
    random.seed(time.time())
    data = []
    with open(chessfile, 'r') as f:
        lines = f.readlines()
        for i in range(batch_size):
            data.append(random.choice(lines).strip())
    return data

def load_dataset(chessfile: str, model: Model) -> tuple[int, list[str]]:
    with open(chessfile, 'r') as f:
        lines: list[str] = []
        for line in f:
            lines.append(line.strip())
    dataset_size: int = len(lines)
    num_batch:    int = int(cp.ceil(dataset_size / model.batch_size))
    return num_batch, lines

def train(loadfile: str, chessfile: str, savefile: Optional[str]) -> None:
    print(f"Training with loadfile={loadfile}, chessfile={chessfile}, savefile={savefile}")

    model = Model.from_file(loadfile)
    params = model.parameters()

    print(f"Model loaded: layers={len(model.layers)}, lr={model.lr}, batch_size={model.batch_size}, epochs={model.epochs}")

    def mse_batch(pred: Tensor, target: Tensor) -> Tensor:
        """Compute MSE for entire batch at once"""
        diff: Tensor = pred - target
        squared_diff: Tensor = diff * diff
        # Mean over all elements
        total: Tensor = squared_diff.sum()
        scale: Tensor = Tensor(0.5 / pred.data.size)
        return total * scale
    
    num_batches:   int
    dataset_lines: list[str]
    dataset_info:  tuple[int, list[str]] = load_dataset(chessfile, model)
    num_batches, dataset_lines           = dataset_info

    if not dataset_lines:
        print("Empty dataset\n")
        return
    
    for epoch in range(model.epochs):
        epoch_loss = 0.0
        random.shuffle(dataset_lines)
        
        # Process multiple batches per epoch
        for batch_idx in range(num_batches):
            start: int = batch_idx * model.batch_size
            end: int = start + model.batch_size
            batch_lines: list[str] = dataset_lines[start:end]
            
            # Prepare batch data
            X_batch = []
            y_batch = []
            for single in batch_lines:
                X, y = parse_line(single)
                X_batch.append(X)
                y_batch.append(y)
            
            # Convert to Tensors - batch processing
            X_tensor = Tensor(cp.array(X_batch))
            y_tensor = Tensor(cp.array(y_batch))
            
            # Zero gradients
            for p in params:
                p.grad = cp.zeros_like(p.data)
            
            # Single forward and backward pass for entire batch
            y_pred = model(X_tensor)
            loss = mse_batch(y_pred, y_tensor)
            loss.backward()
            
            # Update parameters
            for p in params:
                p.data -= model.lr * p.grad
            
            epoch_loss += loss.data
        
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{model.epochs}, Loss: {avg_epoch_loss:.6f}")
        
        # Save model after each epoch
        if savefile:
            model.put_in_file(savefile)
        else:
            model.put_in_file(loadfile)


def predict(loadfile: str, chessfile: str) -> None:
    print(f"Predicting with loadfile={loadfile}, chessfile={chessfile}")
    model = Model([], 0, 0, 0)
    model.load_from_file(loadfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="My Torch",
        usage="%(prog)s [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE",
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true', help="Train the model")
    mode_group.add_argument('--predict', action='store_true', help="Run prediction")
    parser.add_argument('--save', metavar='SAVEFILE', help="Destination to save the trained model")
    parser.add_argument('LOADFILE', help="Path to the model file to load")
    parser.add_argument('CHESSFILE', help="Path to the chess data file")
    # parser.print_help()

    args = parser.parse_args()

    if args.save and not args.train:
        parser.error("--save can only be used with --train")

    if args.train:
        train(args.LOADFILE, args.CHESSFILE, args.save)
    else:
        predict(args.LOADFILE, args.CHESSFILE)
