from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import numpy as np
from rdkit.Chem import DataStructs
import rdkit.rdBase as rkrd
from vae_model import MoleculeVAE, train_vae, save_vae, load_vae
from smiles_decoder import SMILESDecoder, SMILESDataset, train_decoder, save_decoder, load_decoder, generate_smiles_from_fingerprint
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Suppress RDKit warnings
rkrd.DisableLog('rdApp.warning')

# Load your molecular dataset
def load_smiles_dataset(file_path):
    data = pd.read_csv(file_path)
    return data['SMILES'].tolist()

# Preprocess the SMILES strings (e.g., remove NaN or non-strings)
def preprocess_smiles(smiles_list):
    valid_smiles = []
    for smile in smiles_list:
        if isinstance(smile, str) and smile.strip():
            try:
                mol = Chem.MolFromSmiles(smile)
                if mol:
                    valid_smiles.append(smile)
            except:
                pass
    return valid_smiles

# Convert SMILES to Morgan fingerprints
def smiles_to_fp(smiles, fp_size=1024):
    generator = AllChem.GetMorganGenerator(radius=2, fpSize=fp_size)
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        Chem.SanitizeMol(mol)  # Preprocess the molecule
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    return None

# Validate SMILES strings
def validate_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return mol is not None
    else:
        return False

# def clean_smiles(smiles):
#     if validate_smiles(smiles):
#         mol = Chem.MolFromSmiles(smiles)
#         Chem.SanitizeMol(mol)
#         return Chem.MolToSmiles(mol, canonical=True)  # Return canonical SMILES

#     return None

# def fix_common_smiles_issues(smiles):
#     valid_chars = set("CNOFPSIHBrcnopfsi0123456789-=#()[]+/\\")
#     smiles = "".join([char for char in smiles if char in valid_chars])  # Remove invalid characters

#     # Balance brackets
#     while smiles.count("(") > smiles.count(")"):
#         smiles += ")"
#     while smiles.count("[") > smiles.count("]"):
#         smiles += "]"

#     # Fix unmatched ring closures
#     ring_numbers = set(char for char in smiles if char.isdigit())
#     for num in ring_numbers:
#         if smiles.count(num) % 2 != 0:  # Remove unmatched digits
#             smiles = smiles.replace(num, "", 1)

#     if validate_smiles(smiles):
#         return smiles
    
#     return None

# def refine_smiles(smiles):
#     for i in range(len(smiles), 0, -1):
#         truncated_smiles = smiles[:i]  # Try shorter versions of the string
#         if validate_smiles(smiles):
#             mol = Chem.MolFromSmiles(truncated_smiles)
#             if mol:
#                 return Chem.MolToSmiles(mol, canonical=True)  # Return canonical SMILES
#         else:
#             continue
#     return None

# def fix_smiles_pipeline(smiles):
#     # Step 1: Clean with RDKit sanitization
#     fixed_smiles = clean_smiles(smiles)
#     if fixed_smiles:
#         return fixed_smiles

#     # Step 2: Apply basic heuristics
#     smiles = fix_common_smiles_issues(smiles)

#     # Step 3: Refine iteratively
#     fixed_smiles = refine_smiles(smiles)
#     if fixed_smiles:
#         return fixed_smiles

#     return None

# Function to calculate SA score with fixes
# def calculate_sa_score_with_fix(smiles):
#     fixed_smiles = fix_smiles_pipeline(smiles)  # Attempt to fix the SMILES
#     if fixed_smiles:
#         mol = Chem.MolFromSmiles(fixed_smiles)
#         if mol:
#             sa_score = calculateScore(mol)
#             return sa_score
#     return None
    
# Load and preprocess dataset
smiles_data = load_smiles_dataset('FDA_approved_drugs_with_smiles.csv')
# print(smiles_data)
cleaned_smiles_data = preprocess_smiles(smiles_data)
fingerprints = np.array([smiles_to_fp(smile) for smile in cleaned_smiles_data if smile])
fingerprints_tensor = torch.tensor(fingerprints, dtype=torch.float32)
# fingerprints_tensor = fingerprints.clone().detach().float()
# dataset = TensorDataset(fingerprints_tensor)
# data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# print(fingerprints.shape)

# Vocabulary
vocab = sorted(list(set("".join(cleaned_smiles_data))))
vocab.append('<EOS>')
char_to_idx = {char: idx for idx, char in enumerate(vocab)}  # Map characters to indices
idx_to_char = {idx: char for char, idx in char_to_idx.items()}  # Map indices back to characters

# print("Vocabulary:", vocab)
# print("Character-to-Index Mapping:", char_to_idx)

# Encode SMILES and pad sequences
smiles_sequences = [[char_to_idx[char] for char in smile] for smile in cleaned_smiles_data]
padding_idx = len(vocab)
# print("Encoded SMILES Sequences:", smiles_sequences)

# Convert sequences to tensors and pad them
padded_sequences = pad_sequence([torch.tensor(seq) for seq in smiles_sequences], batch_first=True, padding_value=padding_idx)
# print("Padded Sequences Shape:", padded_sequences.shape)

# Prepare DataLoader
dataset = SMILESDataset(fingerprints_tensor, padded_sequences)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Train VAE
vae = MoleculeVAE(input_dim=1024, hidden_dim=512, latent_dim=512)
# train_vae(vae, DataLoader(TensorDataset(fingerprints_tensor), batch_size=64), epochs=100, lr=1e-3)
# save_vae(vae, "vae_model.pth")

# Train Decoder
decoder = SMILESDecoder(latent_dim=512, hidden_dim=256, output_dim=len(vocab), max_length=120)
criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)
# train_decoder(decoder, vae, data_loader, epochs=100, lr=1e-3, criterion=criterion)
# save_decoder(decoder, "smiles_decoder.pth")

# Load VAE and decoder
vae = load_vae(vae, "vae_model.pth")
decoder = load_decoder(decoder, "smiles_decoder.pth")

# Generate latent vector and SMILES
# Generate SMILES with VAE + Decoder
vae.eval()
decoder.eval()

with torch.no_grad():
    for batch in data_loader:
        fingerprints = batch[0]
        mean, _ = vae.encode(fingerprints)
        latent_vector = mean[0].unsqueeze(0)# + 0.1 * torch.randn_like(mean[0].unsqueeze(0))  # Perturbation
        break

# def generate_and_validate_smiles(decoder, latent_vector, max_length, vocab):
#     for _ in range(5):  # Retry up to 5 times
#         smiles = generate_smiles_from_fingerprint(decoder, latent_vector, max_length, vocab)
#         if validate_smiles(smiles):
#             return smiles
#     return "Invalid SMILES generated."

# generated_smiles = generate_and_validate_smiles(decoder, latent_vector, max_length=120, vocab=idx_to_char)
# Example latent vector
latent_vector = torch.randn(1, 512)  # Random latent vector

# Generate SMILES
generated_smiles = decoder.decode(latent_vector, max_length=120, vocab=idx_to_char)
# fixed = calculate_sa_score_with_fix(generated_smiles)
print()
print("Generated SMILES:")
print()
print(generated_smiles)
print()

# generated_smiles = generate_smiles_from_fingerprint(decoder, latent_vector, max_length=120, vocab=idx_to_char)

# if validate_smiles(generated_smiles):
#     print("Generated SMILES (VAE):", generated_smiles)
# else:
#     print("Invalid SMILES generated.")

# Visualize Latent Space
# latent_vectors = []
# for batch in data_loader:
#     fingerprints = batch[0]
#     mean, _ = vae.encode(fingerprints)
#     latent_vectors.append(mean.detach().numpy())
# latent_vectors = np.concatenate(latent_vectors, axis=0)
# pca = PCA(n_components=2)
# latent_2d = pca.fit_transform(latent_vectors)
# plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.7)
# plt.title("Latent Space Visualization")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.show()

# latent_vector = torch.randn(1, 128)  # Example latent vector
# # generated_smiles = generate_smiles(decoder, latent_vector, max_length=120, vocab=idx_to_char)
# generated_smiles = generate_smiles_from_fingerprint(decoder, latent_vector, max_length=120, vocab=idx_to_char)
# print("Generated SMILES:", generated_smiles[1])




# MOLBERT

# # from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import torch

# # Load the tokenizer and the pre-trained MegaMolBART model
# tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
# model = AutoModelForSeq2SeqLM.from_pretrained("seyonec/MegaMolBART")

# # Function to generate SMILES
# def generate_smiles(num_samples=10, latent_size=256):
#     smiles_list = []
    
#     for _ in range(num_samples):
#         # Generate a random latent vector
#         latent_vector = torch.randn(1, latent_size)
        
#         # Decode the latent vector into a SMILES sequence
#         outputs = model.generate(latent_vector)
        
#         # Convert tokenized output to SMILES string
#         smiles = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         smiles_list.append(smiles)
    
#     return smiles_list

# # Generate SMILES
# num_samples = 10  # Specify how many SMILES you want to generate
# generated_smiles = generate_smiles(num_samples)

# # Output the generated SMILES
# for idx, smile in enumerate(generated_smiles):
#     print(f"{idx + 1}: {smile}")


# CHEMVAE
# import numpy as np
# from deepchem.models import ChemVAE

# # Load the pre-trained ChemVAE model
# model = ChemVAE.load_pretrained()

# # Function to generate SMILES strings using ChemVAE
# def generate_smiles_chemvae(model, num_samples=100):
#     generated_smiles = []
#     for _ in range(num_samples):
#         latent_vector = np.random.normal(size=(1, model.latent_size))  # Sample a latent vector
#         smile = model.decode(latent_vector)  # Decode to generate SMILES
#         generated_smiles.append(smile[0])  # Decode returns a list; get the first element
#     return generated_smiles

# # Generate 100 SMILES strings
# generated_smiles = generate_smiles_chemvae(model, num_samples=100)

# # Output generated SMILES
# for i, smile in enumerate(generated_smiles):
#     print(f"{i + 1}: {smile}")
