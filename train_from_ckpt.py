"""
Adapted from Nakata, S., Mori, Y. & Tanaka, S. 
End-to-end protein–ligand complex structure generation with diffusion-based generative models.
BMC Bioinformatics 24, 233 (2023).
https://doi.org/10.1186/s12859-023-05354-5

Repository: https://github.com/shuyana/DiffusionProteinLigand
"""
import os
import warnings
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from ProteinReDiff.data import PDBDataModule
from ProteinReDiff.model import ProteinReDiffModel



def main(args):
    pl.seed_everything(args.seed, workers=True)
    args.save_dir.mkdir(parents=True)

    datamodule = PDBDataModule.from_argparse_args(args)
    model = ProteinReDiffModel(args)
    trainer = pl.Trainer.from_argparse_args(
        args,
        accelerator="auto",
        precision=16,
        strategy="ddp_find_unused_parameters_false",
        resume_from_checkpoint=args.trained_ckpt,
        callbacks=[
            ModelCheckpoint(
                filename="{epoch:03d}-{val_loss:.2f}",
                monitor="val_loss",
                save_top_k=3,
                save_last=True,
            )
        ],
        default_root_dir=args.save_dir,
        max_epochs=-1,
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = PDBDataModule.add_argparse_args(parser)
    parser = ProteinReDiffModel.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--trained_ckpt", type=Path, required=True)
    args = parser.parse_args()

    # https://github.com/Lightning-AI/lightning/issues/5558#issuecomment-1199306489
    warnings.filterwarnings("ignore", "Detected call of", UserWarning)

    main(args)
