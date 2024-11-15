import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

teacher_forcing_ratio = 0.5

# Decoder RNN Model
class SMILESDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, max_length):
        super(SMILESDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_length = max_length

        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.embedding = nn.Embedding(output_dim, hidden_dim)  # Map vocab indices to hidden_dim
        # self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)  # Replace GRU with LSTM
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, fingerprints, hidden=None):
        latent = self.fc1(fingerprints).unsqueeze(1)
        outputs = []
        for _ in range(self.max_length):
            latent, hidden = self.rnn(latent, hidden)
            out = self.fc2(latent)
            outputs.append(out)
        return torch.cat(outputs, dim=1)

# Dataset Class
class SMILESDataset(Dataset):
    def __init__(self, fingerprints, sequences):
        self.fingerprints = fingerprints
        self.sequences = sequences

    def __len__(self):
        return len(self.fingerprints)

    def __getitem__(self, idx):
        return self.fingerprints[idx], self.sequences[idx]

# Training function
def train_decoder(decoder, vae, data_loader, epochs, lr, criterion):
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    
    for epoch in range(epochs):
        decoder.train()
        epoch_loss = 0
        for fingerprints, sequences in data_loader:
            # Obtain latent vectors from VAE
            with torch.no_grad():  # No need to compute gradients for VAE during decoder training
                mean, log_var = vae.encode(fingerprints)
                latent_vectors = vae.reparameterize(mean, log_var)

            # Forward pass through decoder
            output_sequences = decoder(latent_vectors)
            # Target sequences (shifted by one)
            target_sequences = sequences[:, 1:]

            # Apply teacher forcing
            # use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            # if use_teacher_forcing:
            #     decoder_input = sequences[:, :-1]  # Use ground truth as input
            # else:
            #     decoder_input = latent_vectors
            
            # Compute loss
            # loss = criterion(output_sequences.reshape(-1, output_sequences.size(-1)), target_sequences.reshape(-1))

            # Ensure output_sequences is reshaped correctly
            loss = criterion(
                output_sequences.reshape(-1, output_sequences.size(-1)),  # Shape: (batch_size * seq_len, num_classes)
                target_sequences.reshape(-1)[:output_sequences.numel() // output_sequences.size(-1)]  # Shape: (batch_size * seq_len,)
            )

            # # Flatten the tensors
            # output_sequences_flat = output_sequences.reshape(-1, output_sequences.size(-1))  # Shape: (batch_size * seq_len, num_classes)
            # target_sequences_flat = target_sequences.reshape(-1)  # Shape: (batch_size * seq_len)

            # # Create mask for non-padding tokens
            # mask = target_sequences_flat != criterion.ignore_index  # Mask of valid positions

            # # Use the mask to select only valid elements
            # masked_output = output_sequences_flat[mask, :]  # Selects valid rows from logits
            # masked_target = target_sequences_flat[mask]      # Selects valid target elements

            # # Check if there's any valid data left after masking
            # if masked_output.size(0) == 0:
            #     continue  # Skip if the batch only contained padding

            # # Compute loss on masked data
            # loss = criterion(masked_output, masked_target)

            # try:
            #     loss = criterion(
            #         output_sequences.reshape(-1, output_sequences.size(-1)),  # (batch_size * seq_len, num_classes)
            #         target_sequences.reshape(-1)  # (batch_size * seq_len)
            #     )
            # except ValueError as e:
            #     print(f"Shape mismatch error: {e}")
            #     print(f"Output sequence shape after reshape: {output_sequences.reshape(-1, output_sequences.size(-1)).shape}")
            #     print(f"Target sequence shape after reshape: {target_sequences.reshape(-1).shape}")
            #     continue  # Skip this batch if there’s a shape mismatch error

            # loss = criterion(
            #     output_sequences.reshape(-1, output_sequences.size(-1)),
            #     target_sequences.reshape(-1)
            # )

            epoch_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(data_loader)}")

# Generate SMILES
def generate_smiles_from_fingerprint(decoder, fingerprint, max_length, vocab):
    decoder.eval()
    with torch.no_grad():
        # Embed the latent vector through fc1 to get initial input for the RNN
        latent = decoder.fc1(fingerprint.unsqueeze(0))  # Shape: (1, hidden_dim)
        hidden = None  # Initial hidden state is None
        sequence = []

        for _ in range(max_length):
            # Pass through the RNN
            latent, hidden = decoder.rnn(latent, hidden)  # latent shape: (1, 1, hidden_dim)
            output_logits = decoder.fc2(latent.squeeze(1))  # Shape: (1, vocab_size)

            # Use sampling from the probability distribution for diversity
            char_probs = torch.softmax(output_logits, dim=-1)
            char_idx = torch.multinomial(char_probs, num_samples=1).item()

            # Stop generation if EOS token is produced
            if vocab[char_idx] == '<EOS>':
                break

            sequence.append(char_idx)

            # Prepare the next input by transforming the chosen char index back to the RNN input size
            latent = decoder.embedding(torch.tensor([[char_idx]], dtype=torch.long))  # Use embedding layer

    return "".join([vocab[idx] for idx in sequence])

# Save and Load Decoder
def save_decoder(decoder, path):
    torch.save(decoder.state_dict(), path)

def load_decoder(decoder, path):
    # state_dict = torch.load(path)
    state_dict = torch.load(path, weights_only=True)
    decoder.load_state_dict(state_dict)
    return decoder
