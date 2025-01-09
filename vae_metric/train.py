import argparse
import time
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torchvision import datasets, transforms

from models.vqvae import VQVAE

def load_data(data_path, batch_size):

    transform = transforms.ToTensor()
    dataset = datasets.ImageFolder(root=data_path, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    training_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_train_images = torch.cat([img[0].unsqueeze(0) for img in train_dataset], dim=0)  # Stack images
    x_train_var = all_train_images.var().item()

    return training_loader, validation_loader, x_train_var

def save_model_and_results(model, results, hyperparameters, filename):
    """
    Example of a more modern PyTorch save approach.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "results": results,
        "hyperparameters": hyperparameters
    }
    if not os.path.exists("./results"):
        os.makedirs("./results")
    torch.save(checkpoint, f"./results/vqvae_{filename}.pth")


def main(args):

    if args.filename is None:
        args.filename = time.strftime("%Y%m%d_%H%M%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.save:
        print(f"Results will be saved in ./results/vqvae_{args.filename}.pth")

    # Load data (in reality, adapt to your actual dataset)
    training_loader, validation_loader, x_train_var = load_data(
        args.dataset, args.batch_size
    )

    # Build model
    model = VQVAE(
        h_dim=args.n_hiddens,
        res_h_dim=args.n_residual_hiddens,
        n_res_layers=args.n_residual_layers,
        n_embeddings=args.n_embeddings,
        embedding_dim=args.embedding_dim,
        beta=args.beta
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

    results = {
        "n_updates": 0,
        "recon_errors": [],
        "loss_vals": [],
        "perplexities": []
    }

    model.train()

    update_count = 0
    loader_iter = iter(training_loader)

    while update_count < args.n_updates:
        try:
            x, _ = next(loader_iter)
        except StopIteration:
            loader_iter = iter(training_loader)
            x, _ = next(loader_iter)

        x = x.to(device)
        optimizer.zero_grad()

        embedding_loss, x_hat, perplexity = model(x)
        # If x_train_var is 0 or very small, you may want to clamp or handle it carefully
        recon_loss = torch.mean((x_hat - x) ** 2) / x_train_var
        loss = recon_loss + embedding_loss

        loss.backward()
        optimizer.step()

        results["recon_errors"].append(recon_loss.item())
        results["perplexities"].append(perplexity.item())
        results["loss_vals"].append(loss.item())
        results["n_updates"] = update_count

        if update_count % args.log_interval == 0 and update_count > 0:
            interval_slice = slice(-args.log_interval, None)
            avg_recon = np.mean(results["recon_errors"][interval_slice])
            avg_loss = np.mean(results["loss_vals"][interval_slice])
            avg_perp = np.mean(results["perplexities"][interval_slice])

            print(f"Update #{update_count} | "
                  f"Recon Error: {avg_recon:.4f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Perplexity: {avg_perp:.4f}")

            if args.save:
                hyperparams = vars(args)
                save_model_and_results(model, results, hyperparams, args.filename + f"_{update_count}")

        update_count += 1

    if args.save:
        hyperparams = vars(args)
        save_model_and_results(model, results, hyperparams, args.filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_updates", type=int, default=50000)
    parser.add_argument("--n_hiddens", type=int, default=128)
    parser.add_argument("--n_residual_hiddens", type=int, default=32)
    parser.add_argument("--n_residual_layers", type=int, default=2)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--n_embeddings", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--dataset", type=str)

    parser.add_argument("--save", action="store_true")
    parser.add_argument("--filename", type=str, default=None)

    args = parser.parse_args()
    main(args)
