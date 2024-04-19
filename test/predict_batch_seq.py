import dataclasses
import itertools
import warnings
import os
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from typing import Iterable, List, Union, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from rdkit import Chem
from torch.utils.data import DataLoader

from ProteinReDiff.data import InferenceDataset, collate_fn, ligand_to_data, protein_to_data
from ProteinReDiff.model import ProteinReDiffModel ## (NN)
from ProteinReDiff.mol import get_mol_positions, mol_from_file, update_mol_positions
from ProteinReDiff.protein import (
    RESIDUE_TYPES,
    RESIDUE_TYPE_INDEX,
    Protein,
    protein_from_pdb_file,
    protein_from_sequence,
    proteins_to_pdb_file,
)
from ProteinReDiff.tmalign import run_tmalign

RESIDUE_TYPES_MASK = RESIDUE_TYPES + ["<mask>"]
def compute_residue_esm(protein: Protein) -> torch.Tensor:
    esm_model, esm_alphabet = torch.hub.load(
        "facebookresearch/esm:main", "esm2_t33_650M_UR50D"
    )
    esm_model.cuda().eval()
    esm_batch_converter = esm_alphabet.get_batch_converter()

    data = []
    for chain, _ in itertools.groupby(protein.chain_index):
        sequence = "".join(
            [RESIDUE_TYPES_MASK[aa] for aa in protein.aatype[protein.chain_index == chain]]
        )
        data.append(("", sequence))
    batch_tokens = esm_batch_converter(data)[2].cuda()
    with torch.inference_mode():
        results = esm_model(batch_tokens, repr_layers=[esm_model.num_layers])
    token_representations = results["representations"][esm_model.num_layers].cpu()
    residue_representations = []
    for i, (_, sequence) in enumerate(data):
        residue_representations.append(token_representations[i, 1 : len(protein.aatype) + 1])
    residue_esm = torch.cat(residue_representations, dim=0)
    assert residue_esm.size(0) == len(protein.aatype)
    return residue_esm

def proteins_from_fasta(fasta_file: Union[str, Path]):
    names = []
    proteins = []
    with open(fasta_file, "r") as f:
        for line in f:
            if line.startswith(">"):
                name = line.lstrip(">").rstrip("\n").replace(" ","_")
                names.append(name)
            elif not line in ['\n', '\r\n']:
                sequence = line.rstrip("\n")
                protein = protein_from_sequence(sequence)
                proteins.append(protein)

    return proteins, names

def parse_ligands(ligand_input: Union[str, Path, list]):
    ligands = []
    if isinstance(ligand_input, list):
        for lig in ligand_input:
            ligand = Chem.MolFromSmiles(lig)
            ligand = update_mol_positions(ligand, np.zeros((ligand.GetNumAtoms(), 3)))
            ligands.append(ligand)
    else:
        with open(ligand_input, "r") as f:
            for line in f:
                ligand = Chem.MolFromSmiles(line.rstrip("\n"))
                ligand = update_mol_positions(ligand, np.zeros((ligand.GetNumAtoms(), 3)))
                ligands.append(ligand)

    return ligands

def update_pos(
    protein: Protein, ligand: Chem.Mol, pos: np.ndarray
) -> Tuple[Protein, Chem.Mol]:
    atom_pos = np.zeros_like(protein.atom_pos)
    atom_pos[:, 1] = pos[ligand.GetNumAtoms() :]
    atom_mask = np.zeros_like(protein.atom_mask)
    atom_mask[:, 1] = 1.0
    protein = dataclasses.replace(protein, atom_pos=atom_pos, atom_mask=atom_mask)
    ligand = update_mol_positions(ligand, pos[: ligand.GetNumAtoms()])
    return protein, ligand

def predict_seq(
    proba: torch.Tensor
) -> list :
    tokens = torch.argmax(torch.softmax((torch.tensor(proba)), dim = -1), dim = -1)
    RESIDUE_TYPES_NEW = ["X"] + RESIDUE_TYPES
    return "".join(map(lambda i : RESIDUE_TYPES_NEW[i], tokens)).lstrip("X").rstrip("X")

def update_seq(
    protein: Protein, proba: torch.Tensor
) -> Protein:
    tokens = torch.argmax(torch.softmax((torch.tensor(proba)), dim = -1), dim = -1)
    RESIDUE_TYPES_NEW = ["X"] + RESIDUE_TYPES
    sequence = "".join(map(lambda i : RESIDUE_TYPES_NEW[i], tokens)).lstrip("X").rstrip("X")
    aatype = np.array([RESIDUE_TYPES.index(s) for s in sequence], dtype=np.int64)
    protein = dataclasses.replace(protein, aatype = aatype)
    return protein


def main(args):
    pl.seed_everything(args.seed, workers=True)
    # Check if the directory exists
    if os.path.exists(args.output_dir):
        # Remove the existing directory
        os.rmdir(args.output_dir)
    args.output_dir.mkdir(parents=True)

    # Model
    model = ProteinReDiffModel.load_from_checkpoint(
        args.ckpt_path, num_steps=args.num_steps
    )
    ## (NN)
    model.training_mode = False
    args.num_gpus = 1
    model.mask_prob = args.mask_prob
    ## (NN)

    # Inputs
    proteins, names = proteins_from_fasta(args.fasta)
    
    if args.ligand_file is None:
        ligand_input = ["*"]*args.num_samples*len(names)
        
        ligands = parse_ligands(ligand_input)
    else:
        ligands = parse_ligands(args.ligand_file)
    
    # total_num_atoms = len(protein.aatype) + ligand.GetNumAtoms()
    # print(f"Total number of atoms: {total_num_atoms}")
    # if total_num_atoms > 400:
    #     warnings.warn("Too many atoms (> 400). May take a long time for sample generation.")
    
    datas = []
    for protein, ligand in zip(proteins, ligands):
        data = {
            **ligand_to_data(ligand),
            **protein_to_data(protein, residue_esm=compute_residue_esm(protein)),
        }
        datas.extend([data]*args.num_samples)

    

    # Generate samples
    trainer = pl.Trainer.from_argparse_args(
        args,
        accelerator="auto",
        gpus = args.num_gpus,
        default_root_dir=args.output_dir,
        max_epochs=-1,
    )
    results = trainer.predict(    
        model,
        dataloaders=DataLoader(
            InferenceDataset(datas, args.num_samples * len(names)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        ),
    )

    torch.save(results,"results.pt")
    positions = [p[0] for p in results] ## (NN)
    probabilities = [s[1] for s in results] ## (NN)
    
    # positions = torch.cat(positions, dim=0).detach().cpu().numpy()
    # probabilities = torch.cat(probabilities, dim=0).detach().cpu().numpy() #(NN)
    names = [n for n in names for _ in range(args.num_samples)]
    
        
    with open(args.output_dir / "sample_sequences.fasta", "w") as f:
        for i, (name, seq_prob) in enumerate(zip(names, probabilities)):
            f.write(">{}_sample_{}\n".format(name,i%args.num_samples))
            sequence = predict_seq(seq_prob.squeeze())
            f.write("{}\n".format(sequence))
            
    # Save samples
    # sample_proteins, sample_ligands = [], []
    # tmscores = []
    # for pos, seq_prob in zip(positions, probabilities):
        # sample_protein, sample_ligand = update_pos(protein, ligand, pos)
        # sample_protein = update_seq(sample_protein, seq_prob)
    #     if ref_protein is None:
    #         warnings.warn(
    #             "Using the first sample as a reference. The resulting structures may be mirror images."
    #         )
    #         ref_protein = sample_protein
    #     tmscore, t, R = max(
    #         run_tmalign(sample_protein, ref_protein),
    #         run_tmalign(sample_protein, ref_protein, mirror=True),
    #         key=itemgetter(0),
    #     )
    #     sample_proteins.append(
    #         dataclasses.replace(
    #             sample_protein, atom_pos=t + sample_protein.atom_pos @ R
    #         )
    #     )
    #     sample_ligands.append(
    #         update_mol_positions(
    #             sample_ligand, t + get_mol_positions(sample_ligand) @ R
    #         )
    #     )
    #     tmscores.append(tmscore)
    # proteins_to_pdb_file(sample_proteins, args.output_dir / "sample_protein.pdb")
    # with Chem.SDWriter(str(args.output_dir / "sample_ligand.sdf")) as w:
    #     for sample_ligand in sample_ligands:
    #         w.write(sample_ligand)
    # with open(args.output_dir / "sample_tmscores.txt", "w") as f:
    #     for tmscore in tmscores:
    #         f.write(str(tmscore) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=torch.get_num_threads())
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--mask_prob", type=float, default=0.3)
    parser.add_argument("--training_mode", action="store_true")
    parser.add_argument("-c", "--ckpt_path", type=Path, required=True)
    parser.add_argument("-o", "--output_dir", type=Path, required=True)
    parser.add_argument("-p", "--fasta", type=str, required=True)
    parser.add_argument("-l", "--ligand_file", type=str, default=None)
    parser.add_argument("-n", "--num_samples", type=int, default = 1)
    args = parser.parse_args()

    main(args)
