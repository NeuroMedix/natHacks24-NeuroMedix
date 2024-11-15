import torch
import torch.nn as nn
import torch.optim as optim

# Define the VAE architecture
class MoleculeVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(MoleculeVAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # Mean and log-variance
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        enc_out = self.encoder(x)
        mean, log_var = torch.chunk(enc_out, 2, dim=1)
        return mean, log_var
    
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        return recon_x, mean, log_var

# Training function
def train_vae(vae, data_loader, epochs, lr):
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        vae.train()
        epoch_loss = 0
        kl_weight = min(1.0, epoch / (epochs * 0.5))  # Gradually increase up to 1.0
        for batch in data_loader:
            fingerprints = batch[0]  # Input data

            # Forward pass
            recon_batch, mean, log_var = vae(fingerprints)

            # Compute loss
            recon_loss = loss_fn(recon_batch, fingerprints)
            kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            loss = recon_loss + kl_divergence * kl_weight

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(data_loader)}, KL Weight: {kl_weight}")

# Save the VAE model
def save_vae(vae, path):
    torch.save(vae.state_dict(), path)

# Load the VAE model
def load_vae(vae, path):
    # state_dict = torch.load(path)
    state_dict = torch.load(path, weights_only=True)
    vae.load_state_dict(state_dict)
    return vae

