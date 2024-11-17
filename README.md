# SMILES-based Molecular Generation Repository

## **Overview**
This repository is designed to develop and experiment with machine learning models for generating and validating molecular structures using SMILES (Simplified Molecular Input Line Entry System). The project employs a Variational Autoencoder (VAE) and a SMILES Decoder to create new molecules, focusing on Alzheimer's Disease drug discovery.

---

## **Repository Structure**

### **Code Files**
- **`main.py`**: Main script for loading data, training models, and generating SMILES.
- **`vae_model.py`**: Defines the VAE architecture for encoding and decoding molecules.
- **`smiles_decoder.py`**: Implements the SMILES Decoder to reconstruct SMILES from latent space.
- **`smile_to_img.py`**: Converts SMILES strings into molecular images.

### **Dataset Files**
- **`FDA_approved_drugs_with_smiles.csv`**: FDA-approved drugs with SMILES data.
- **`ad_drugs.csv`**: Drugs related to Alzheimer's Disease.
- **`zinc_database.csv`**: Additional dataset for training.

### **Model Files**
- **`vae_model.pth`**: Pre-trained VAE model weights.
- **`smiles_decoder.pth`**: Pre-trained SMILES Decoder model weights.

### **Utility Files**
- **`requirements.txt`**: Lists required Python libraries.

---

## **Workflow**

### **Step 1: Preprocess Data**
- Load and clean SMILES datasets using RDKit.

### **Step 2: Convert SMILES**
- Transform SMILES into numerical morgan fingerprints using RDKit.

### **Step 3: Train Models**
- Train the VAE to encode and reconstruct molecules.
- Train the SMILES Decoder to generate SMILES from latent representations.

### **Step 4: Generate New Molecules**
- Use trained models to generate new SMILES from the latent space.

### **Step 5: Validate and Visualize**
- Validate the generated SMILES.
- Visualize molecular structures using `smile_to_img.py`.

---

## **How to Run**

1. **Install Dependencies**
   - Install all required libraries:
     ```bash
     pip install -r requirements.txt
     ```

2. **Run the Main Script**
   - Train models, preprocess data, and generate new SMILES:
     ```bash
     python main.py
     ```

3. **Use Pre-trained Models**
   - Skip training by loading the pre-trained weights:
     - `vae_model.pth`
     - `smiles_decoder.pth`

4. **Generate New Molecules**
   - Modify `main.py` to sample latent space and generate new SMILES.

---

## **Limitations and Future Work**

### **Current Limitations**
- Generated SMILES may not always represent valid molecules.
- Limited by the quality and size of the training dataset.
- Basic model architecture may struggle with highly complex molecular structures.

### **Future Improvements**
- Incorporate larger, more diverse datasets for training.
- Use advanced model architectures like MegaMolBART or ChemBERTa.
- Implement stricter validation techniques to ensure the generation of chemically meaningful molecules.

---

## **Applications**
While this project demonstrates drug discovery for Alzheimer's Disease, the same framework can be extended to:
- Discover treatments for other diseases.
- Explore new materials or biologically active compounds.

By refining this workflow, we aim to accelerate drug development and reduce costs, making it accessible for small pharmaceutical companies.
