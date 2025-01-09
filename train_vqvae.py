import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
from torch import optim
from torch.utils.data import Dataset

# NOTE: otherwise vqvae modules dont see each other
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "vqvae")))
from vqvae.models.vqvae import VQVAE
IMAGE_DIM = 128

def train_vqvae(
    data_path,
    output_checkpoint="vqvae_checkpoint.pt",
    batch_size=16,
    epochs=10,
    h_dim=IMAGE_DIM,
    res_h_dim=32,
    n_res_layers=2,
    n_embeddings=512,
    embedding_dim=64,
    beta=0.25,
    device='cuda',      
):
    transform = transforms.ToTensor()
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    print("Loading the data..")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2,pin_memory=True)

    model = VQVAE(
        h_dim=h_dim,
        res_h_dim=res_h_dim,
        n_res_layers=n_res_layers,
        n_embeddings=n_embeddings,
        embedding_dim=embedding_dim,
        beta=beta,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    global_step = 0
    print("Starting the training..")
    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(dataloader):
            # blocker for the demo, i want to see if it works on a tiny batch
            if i == 1000:
                break
            imgs = imgs.to(device)
            optimizer.zero_grad()

            embedding_loss, img_hat, perplexity = model(imgs)
            recon_loss = F.mse_loss(img_hat, imgs)
            total_loss = embedding_loss + recon_loss

            total_loss.backward()
            optimizer.step()
            global_step += 1

            if i % 50 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs}, "
                    f"Step {i}, "
                    f"Recon Loss: {recon_loss.item():.4f}, "
                    f"VQ Loss: {embedding_loss.item():.4f}, "
                    f"Total: {total_loss.item():.4f}, "
                    f"Perplexity: {perplexity.item():.2f}, "
                )
    print("Training is complete, sabing the model")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),  
        "h_dim": h_dim,
        "res_h_dim": res_h_dim,
        "n_res_layers": n_res_layers,
        "n_embeddings": n_embeddings,
        "embedding_dim": embedding_dim,
        "beta": beta,
    }, output_checkpoint)

    print("Saved")

if __name__ == '__main__':
    data_path = os.path.join("dataset")
    train_vqvae(data_path)
