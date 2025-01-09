from vqvae.models.vqvae import VQVAE
import torch.nn.functional as F
import torch

class Metric:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.vqvae = VQVAE(
            h_dim=checkpoint["h_dim"],
            res_h_dim=["res_h_dim"],
            n_res_layers=["n_res_layers"],
            n_embeddings=["n_embeddings"],
            embedding_dim=["embedding_dim"],
            beta=checkpoint["beta"],
        ).to(self.device)

        self.vqvae.load_state_dict(checkpoint['model_state_dict'])
        self.vqvae.eval()

    @torch.no_grad()
    def reconstruction_loss(self, image):
        image = image.to(self.device)
        _, image_hat, _ = self.vqvae(image)
        
        recon_loss = F.mse_loss(image_hat, image)
        return recon_loss.item()

    def evo_metric(self, image):
        return -self.reconstruction_loss(image)
    
    def rl_reward(self, image):
        loss = self.reconstruction_loss(image)
        return 1.0 / 1.0 + loss