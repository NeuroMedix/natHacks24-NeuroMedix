from rdkit import Chem
from rdkit.Chem import Draw

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
    smiles_input = input("Enter a valid SMILES string: ")
    create_molecular_structure(smiles_input)
