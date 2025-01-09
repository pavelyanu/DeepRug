from vqvae.models.vqvae import VQVAE
import torch.nn.functional as F
import torch

class Metric:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.vqvae = VQVAE(
            h_dim=checkpoint['hyperparameters']["n_hiddens"],
            res_h_dim=checkpoint['hyperparameters']["n_residual_hiddens"],
            n_res_layers=checkpoint['hyperparameters']["n_residual_layers"],
            n_embeddings=checkpoint['hyperparameters']["n_embeddings"],
            embedding_dim=checkpoint['hyperparameters']["embedding_dim"],
            beta=checkpoint['hyperparameters']["beta"],
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