# Complex-based Ligand-Binding Proteins Redesign by Equivariant Diffusion-based Generative Models

![Equivariant-Diffusion](Equivariant-Diffusion.png)

Contributors:
* Nhan Nguyen
* Duy Nguyen
* Truong Son Hy (Correspondent / PI)

## Setup Environment
Clone this repository and install dependencies:
```bash
git clone https://github.com/HySonLab/Protein_Redesign.git
cd Protein_Redesign
conda env create -f environment.yml
conda activate ProteinReDiff
```

Download model parameters:
```bash
gdown --fuzzy --folder https://drive.google.com/drive/u/1/folders/1AAJ4P5EmQtwle9_eSeNMcF-KMWObksxZ
```

Additionally, TMalign is required to align generated structures.
You can install it as follows:
```bash
wget https://zhanggroup.org/TM-align/TMalign.cpp
g++ -static -O3 -ffast-math -lm -o TMalign TMalign.cpp
chmod +x TMalign
export PATH="/path/to/TMalign:$PATH"
```

## Sample generation
Generate complex structures with the protein structure-free model (ProteinReDiff):
```bash
python generate.py \
    --ckpt_path "checkpoints/ProteinReDiff_v1.ckpt" \
    --output_dir "workdir/generate/example_ProteinReDiff" \
    --protein "LSEQLKHCNGILKELLSKKHAAYAWPFYKPVDASALGLHDYHDIIKHPMDLSTVKRKMENRDYRDAQEFAADVRLMFSNCYKYNPPDHDVVAMARKLQDVFEFRYAKMPD" \
    --ligand "Cc1ccc2c(c1c3cc(cc4c3nc([nH]4)C5CC5)c6c(noc6C)C)cccn2" \
    --num_samples 8
```

Alternatively, the protein structure-dependent model (ProteinReDiff+S) can be used:
```bash
wget https://files.rcsb.org/download/6MOA.pdb
python generate.py \
    --ckpt_path "checkpoints/ProteinReDiffS_v1.ckpt" \
    --output_dir "workdir/generate/example_ProteinReDiffS" \
    --protein "6MOA.pdb" \
    --ligand "Cc1ccc2c(c1c3cc(cc4c3nc([nH]4)C5CC5)c6c(noc6C)C)cccn2" \
    --num_samples 8
```
Note that an input protein structure must be given as a PDB file in this case.

Besides, you can specify a reference protein structure to be used for the alignment of results:
```bash
python generate.py \
    --ckpt_path "checkpoints/ProteinReDiff_v1.ckpt" \
    --output_dir "workdir/generate/example_ProteinReDiff_ref" \
    --protein "LSEQLKHCNGILKELLSKKHAAYAWPFYKPVDASALGLHDYHDIIKHPMDLSTVKRKMENRDYRDAQEFAADVRLMFSNCYKYNPPDHDVVAMARKLQDVFEFRYAKMPD" \
    --ligand "Cc1ccc2c(c1c3cc(cc4c3nc([nH]4)C5CC5)c6c(noc6C)C)cccn2" \
    --num_samples 8 \
    --ref_path "6MOA.pdb"
```
This is used only for alignment and does not affect the generation process itself.

The argument num_steps can be modified from the default of 64 to reduce execution time:
```bash
python generate.py \
    --ckpt_path "checkpoints/ProteinReDiff_v1.ckpt" \
    --output_dir "workdir/generate/example_ProteinReDiff_fast" \
    --protein "LSEQLKHCNGILKELLSKKHAAYAWPFYKPVDASALGLHDYHDIIKHPMDLSTVKRKMENRDYRDAQEFAADVRLMFSNCYKYNPPDHDVVAMARKLQDVFEFRYAKMPD" \
    --ligand "Cc1ccc2c(c1c3cc(cc4c3nc([nH]4)C5CC5)c6c(noc6C)C)cccn2" \
    --num_samples 8 \
    --ref_path "6MOA.pdb" \
    --num_steps 24
```

## Training

Download the PDBbind dataset from https://zenodo.org/record/6408497 and unzip it.

Move the resulting PDBBind_processed directory to data/.

Preprocess the dataset:
```bash
python preprocess_pdbbind.py
```

Finally, run the training script:
```
python train.py \
    --num_workers 8 \
    --batch_size 1 \
    --accumulate_grad_batches 8 \
    --no_cb_distogram \
    --save_dir "workdir/train/example_ProteinReDiff" \
    --single_dim 256 \
    --pair_dim 32 \
    --num_blocks 4
```
where the no_cb_distogram argument makes the model protein structure-free.

Please modify the batch_size and accumulate_grad_batches arguments according to your machine(s).

Default values can be used to reproduce the settings used in our paper:
```
python train.py \
    --num_workers 8 \
    --batch_size 3 \
    --accumulate_grad_batches 8 \
    --no_cb_distogram \
    --save_dir "workdir/train/reproduce_ProteinReDiff"
```
