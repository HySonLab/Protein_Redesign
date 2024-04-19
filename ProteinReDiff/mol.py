"""
Adapted from Nakata, S., Mori, Y. & Tanaka, S. 
End-to-end protein–ligand complex structure generation with diffusion-based generative models.
BMC Bioinformatics 24, 233 (2023).
https://doi.org/10.1186/s12859-023-05354-5

Repository: https://github.com/shuyana/DiffusionProteinLigand

ProteinReDiff includes significant innovations including:
- Stochastically masking & featurization of protein sequences
- Adaptations of Single Representation Attention and Outer Product Mean from AF2
- Parameterization of \beta_T diffusion (instead of using variational lower bound in DPL)
- Denoising through both sequence and structure spaces
- Flexible generation output (sequences only, sequence-structures)

"""

import copy
from pathlib import Path
from typing import Union

import numpy as np
from rdkit import Chem


def standardize_mol(mol: Chem.Mol) -> Chem.Mol:
    mol = copy.deepcopy(mol)

    Chem.SanitizeMol(mol)
    mol = Chem.RemoveHs(mol, sanitize=True)
    if mol is None:
        raise ValueError("Failed to standardize molecule.")

    return mol


def mol_from_file(path: Union[Path, str]) -> Chem.Mol:
    if isinstance(path, str):
        path = Path(path)

    if path.suffix == ".sdf":
        mol = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)[0]
    elif path.suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(path), sanitize=False, removeHs=False)
    else:
        raise ValueError(f"Unrecognized file format: {path.suffix}.")

    if mol is None:
        raise ValueError(f"Failed to construct a molecule from: {str(path)}.")

    mol = standardize_mol(mol)

    return mol


def get_mol_positions(mol: Chem.Mol) -> np.ndarray:
    assert mol.GetNumConformers() == 1

    conformer = mol.GetConformer(0)
    pos = conformer.GetPositions().astype(np.float32)

    return pos


def update_mol_positions(mol: Chem.Mol, pos: np.ndarray) -> Chem.Mol:
    mol = copy.deepcopy(mol)
    if mol.GetNumConformers() == 0:
        mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()))
    assert mol.GetNumConformers() == 1

    pos = np.asarray(pos, dtype=np.float64)
    assert pos.shape == (mol.GetNumAtoms(), 3)

    conformer = mol.GetConformer(0)
    for i, p in enumerate(pos):
        conformer.SetAtomPosition(i, p)

    return mol
