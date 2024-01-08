import click
import torch
from models.model import myawesomemodel

import os
import matplotlib.pyplot as plt


@click.group()
def cli():
    """Command line interface."""
    pass


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths:
current_dir = os.path.dirname(os.path.realpath(__file__))
raw_data_dir = os.path.join(current_dir, "../data/raw/corruptmnist")
processed_data_dir = os.path.join(current_dir, "../data/processed")
models_dir = os.path.join(current_dir, "../models")
visualization_dir = os.path.join(current_dir, "../reports/figures")

# Create directories for outputs if they don't exist
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(visualization_dir):
    os.makedirs(visualization_dir)


## TRAINING
@click.command()
# Adding command line options
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--epochs", default=10, help="number of epochs to train for")
@click.option("--batch_size", default=256, help="batch size to use for training")
@click.option("--model_name", default="model", help="name of model to save")
# Defining the function
def train(lr, epochs, batch_size, model_name):
    """Train a model on MNIST."""
    print("Training day and night")

    # Import model
    model = myawesomemodel.to(device)

    # Import data
    train_data = torch.load(processed_data_dir + "/train_images.pt")
    train_labels = torch.load(processed_data_dir + "/train_target.pt")
    # Convert to dataloader (to convert to batches)
    train_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_data, train_labels),
        batch_size=batch_size,
        shuffle=True,
    )

    # Train model hyperparameters
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_epochs = epochs

    # For visualization
    train_loss = []

    # Training loop
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()

            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)

            loss = criterion(output, labels)

            loss.backward()

            optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
        train_loss.append(loss.item())

    # Prepare plot
    plt.plot(train_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training loss")

    # Save model (and plot)
    # If model_name exists, make new name (add _1, _2, etc.)
    if os.path.exists(models_dir + f"/{model_name}.pt"):
        i = 1
        while os.path.exists(models_dir + f"/{model_name}_{i}.pt"):
            i += 1
        torch.save(model, models_dir + f"/{model_name}_{i}.pt")  # save model
        plt.savefig(visualization_dir + f"/train_loss_{i}.png")  # save plot
    else:
        torch.save(model, models_dir + f"/{model_name}.pt")
        plt.savefig(visualization_dir + "/train_loss.png")

    return loss.item()


## EVALUATION
@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # Import model
    model = torch.load(os.path.join(models_dir, model_checkpoint))

    # Import data
    test_data = torch.load(processed_data_dir + "/test_images.pt")
    test_labels = torch.load(processed_data_dir + "/test_target.pt")
    # Convert to dataloader (to convert to batches)
    test_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_data, test_labels),
        batch_size=64,
        shuffle=False,
    )

    test_predictions = []
    test_labels = []

    # Evaluation loop
    with torch.no_grad():
        for batch in test_dataloader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)

            test_predictions.append(output.argmax(dim=1).cpu())
            test_labels.append(labels.cpu())

        test_predictions = torch.cat(test_predictions, dim=0)
        test_labels = torch.cat(test_labels, dim=0)

    print("Accuracy: ", (test_predictions == test_labels).float().mean().item())


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
