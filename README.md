# SMILES-based Molecular Generation Repository

## **Overview**
This repository is designed to develop and experiment with machine learning models for generating and validating molecular structures using SMILES (Simplified Molecular Input Line Entry System). The project employs a Variational Autoencoder (VAE) and a SMILES Decoder to create new molecules, focusing on Alzheimer's Disease drug discovery.

---

## **Repository Structure**

### **1. Code Files**

- **`main.py`**
  - **Purpose**: Main script for the entire workflow, including:
    - Loading and preprocessing molecular datasets.
    - Converting SMILES strings to numerical representations (Morgan fingerprints).
    - Training the VAE and SMILES decoder models.
    - Generating and validating new SMILES.
  
- **`vae_model.py`**
  - **Purpose**: Defines the architecture and training process for the VAE.
  - **Key Features**:
    - Encoder to compress molecular data into a latent space.
    - Decoder to reconstruct molecular data from the latent space.
    - Training function to optimize the VAE model.

- **`smiles_decoder.py`**
  - **Purpose**: Implements the SMILES Decoder model to convert latent representations into SMILES strings.
  - **Key Features**:
    - RNN-based decoder architecture.
    - Training function for the decoder.
    - Functionality to generate new SMILES from latent space representations.

- **`smile_to_img.py`**
  - **Purpose**: Converts SMILES strings into molecular images for visualization.

### **2. Dataset Files**

- **`FDA_approved_drugs_with_smiles.csv`**
  - **Description**: Contains FDA-approved drugs and their SMILES representations.
  - **Usage**: Primary dataset for training the VAE and decoder models.

- **`ad_drugs.csv`**
  - **Description**: Dataset focusing on drugs related to Alzheimer's Disease.
  - **Usage**: Used for validation or further model refinement.

- **`zinc_database.csv`**
  - **Description**: Supplementary dataset containing additional molecular data.
  - **Usage**: Augments the training of the VAE for a broader chemical space.

### **3. Model Files**

- **`vae_model.pth`**
  - **Description**: Pre-trained weights for the VAE model.
  - **Usage**: Allows direct inference without retraining.

- **`smiles_decoder.pth`**
  - **Description**: Pre-trained weights for the SMILES Decoder model.
  - **Usage**: Enables generation of SMILES strings without retraining.

### **4. Utility Files**

- **`requirements.txt`**
  - **Description**: Specifies the Python libraries required to run the project.
  - **Installation Command**:
    ```bash
    pip install -r requirements.txt
    ```

---

## **Workflow**

### **Step 1: Data Preprocessing**
- Load SMILES datasets (e.g., `FDA_approved_drugs_with_smiles.csv`).
- Validate and clean SMILES strings using RDKit to ensure they represent valid molecules.

### **Step 2: Convert SMILES to Numerical Representations**
- Use RDKit's Morgan fingerprints to convert SMILES into fixed-length numerical arrays.

### **Step 3: Train the Models**
- Train the Variational Autoencoder (VAE) to:
  - Encode molecular data into a latent space.
  - Reconstruct molecules from the latent space.
- Train the SMILES Decoder to:
  - Generate valid SMILES strings from latent representations.

### **Step 4: Generate New Molecules**
- Use the trained VAE and Decoder to sample the latent space and generate new molecular structures in SMILES format.

### **Step 5: Validate and Visualize**
- Validate the generated SMILES strings to ensure they represent valid molecules.
- Convert SMILES strings to images using `smile_to_img.py` for easier visualization.

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
- Generated SMILES - not always represent valid molecules.
- Limited by the quality and size of the training dataset.
- Basic model architecture may struggle with highly complex molecular structures.

### **Future Improvements**
- Incorporate larger, more diverse datasets for training.
- Use advanced model architectures like MegaMolBART or ChemBERTa.
- Implement stricter validation techniques to ensure the generation of chemically meaningful molecules.

---

## **Applications**
While this project demonstrates drug discovery for Alzheimer’s Disease, the same framework can be extended to:
- Discover treatments for other diseases.
- Explore new materials or biologically active compounds.

By refining this workflow, we aim to accelerate drug development and reduce costs, making it accessible for small pharmaceutical companies.
