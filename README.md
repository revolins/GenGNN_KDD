# Review Repo for "Are Expressive Encoders Necessary for Discrete Graph Generation?"

### Installation

1. Install [Conda](https://www.anaconda.com/docs/getting-started/anaconda/install) (our version = 23.7.2) 
2. Create GenGNN's environment:  
   ```bash
   # manually set PATH `prefix:` in gnn.yaml (or remove if using default)
   conda env create -f ggn.yaml
   conda activate ggn
   ```
3. Run the following commands to check if the installation of the main packages was successful:  
   ```bash
   python -c "import sys; print('Python version:', sys.version)"
   python -c "import rdkit; print('RDKit version:', rdkit.__version__)"
   python -c "import graph_tool as gt; print('Graph-Tool version:', gt.__version__)"
   python -c "import torch; print(f'PyTorch version: {torch.__version__}, CUDA version (via PyTorch): {torch.version.cuda}')"
   python -c "import torch_geometric as tg; print('PyTorch Geometric version:', tg.__version__)"
   ```
   If you see no errors, the installation was successful and you can proceed to the next step.
4. Compile the ORCA evaluator:  
   ```bash
   cd src/analysis/orca
   g++ -O2 -std=c++11 -o orca orca.cpp
   ```
5. Compile magnipy:
   ```
   cd magnipy
   poetry install
   ```
---

## Usage

* All commands use `python main.py` with custom global environment overrides. Note that `main.py` is inside the `src` directory.
* Experimental configs are under `configs/experiment`. 
    * ablation_study.components within config is used to specify GenGNN components (LayerNorm, FFN, Node+Edge Gating, Residual, RRWP Node/Edge Features).
    * model.type specifices GenGNN: `conv`, PPGN: `ppgn`, GraphTransformer: `gt`. Note `ppgn` and `gt` will not override RRWP Node/Edge Features.
    * model.name specifies `digress` or `defog`
* Experimental config and model checkpoint files use the following naming convention:
    * DeFoG: `{backbone}_{dataset}`
    * DiGress: `{backbone}_{dataset}_digress`
* `general.gpus` utilizes a list in order to enable direct specification of GPU number (i.e. `general.gpus=[0]`), multi-gpu support is untested.

### Training GenGNN

```bash
python main.py experiment=<backbone_dataset_model> general.gpus=[gpu number]
```

### Evaluating GenGNN
```bash
python main.py experiment=<backbone_dataset_model> general.gpus=[gpu number] general.test_only=<checkpoint>
```

### Example GenGNN Evaluation
```bash
python main.py experiment=gnn_comm20 general.gpus=[0] general.test_only=gnn_comm20.ckpt
```

## Checkpoints

Checkpoints available here: [Google Drive](https://drive.google.com/drive/folders/16GxPMxZI7YNLZ7UVAuiHUyfLLilMciQ5?usp=sharing).

### Checkpoint Folder Structure:

- GenGNN
    - DeFoG/
        - Comm20/
        - Planar/
        - QM9/
        - QM9(H)/
        - SBM/
        - Tree/
        - ZINC250K/
        - guacamol + moses checkpoint files
    - DiGress/
        - Comm20/
        - Planar/
        - QM9/
        - QM9(H)/
        - SBM/
        - Tree/
        - ZINC250K/