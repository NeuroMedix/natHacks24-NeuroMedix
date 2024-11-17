from rdkit import Chem
from rdkit.Chem import Draw
import json
import os

def create_molecular_structure(smiles: str, output_file: str = "molecule.png"):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            Draw.MolToFile(mol, output_file)
            print(f"Molecular structure saved as {output_file}")
        else:
            print("Invalid SMILES string.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
#     smiles_input = input("Enter a valid SMILES string: ")
#     create_molecular_structure(smiles_input)

    # Load the JSON data from a file
    output_parent_folder = "images"
    os.makedirs(output_parent_folder, exist_ok=True)

    with open("database.json", "r") as file:
        data = json.load(file)

    # Process each item in the JSON
    for item in data:
        lead_name = item.get("name")
        lead_folder = os.path.join(output_parent_folder, lead_name)
        os.makedirs(lead_folder, exist_ok=True)

        analogs = item.get("analogs", [])
        
        # Sort by tanimotoSimilarity and take the top 3
        top_analogs = sorted(analogs, key=lambda x: x["tanimotoSimilarity"], reverse=True)[:3]
        
        for analog in top_analogs:
            smiles = analog["smiles"]
            sanitized_filename = ''.join(c for c in smiles if c not in "/\\*") + ".png"
            output_path = os.path.join(lead_folder, sanitized_filename)
            create_molecular_structure(smiles, output_path)

    print(f"All images saved in the '{output_parent_folder}' folder.")