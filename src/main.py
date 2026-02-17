import os
import pathlib
import warnings
from pathlib import Path

import graph_tool
import torch

torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig, ListConfig
from hydra.core.hydra_config import HydraConfig
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from src import utils
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
from graph_discrete_flow_model import GraphDiscreteFlowModel
from discrete_diffusion_model import DiscreteDenoisingDiffusion
from models.extra_features import DummyExtraFeatures, ExtraFeatures
from omegaconf import OmegaConf

warnings.filterwarnings("ignore", category=PossibleUserWarning)

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print("cfg.general.num_sample_fold: ", cfg.general.num_sample_fold)
    if cfg.general.test_only is not None and any(substring in cfg.general.name for substring in ['tls', 'qm9', 'tree', 'comm20', 'sbm', 'planar', 'zinc']): #, 'comm20' -- for testing MagDiff  
            cfg.general.num_sample_fold = 5
    print("cfg.general.num_sample_fold: ", cfg.general.num_sample_fold, flush=True)
    if not hasattr(cfg, 'hier_dir') or cfg.hier_dir is None:
        OmegaConf.set_struct(cfg, False)
        hc = HydraConfig.get()
        original_dir = hc.runtime.output_dir
        cfg.hier_dir = original_dir
        print(f"Set hier directory to: {cfg.hier_dir}", flush=True)
    if "experiment" in cfg:
        OmegaConf.set_struct(cfg, False)
        cfg = OmegaConf.merge(cfg, cfg.experiment)
    pl.seed_everything(cfg.train.seed)
    print("ROOT TYPE:", type(cfg))              # must be DictConfig
    print("HAS experiment:", "experiment" in cfg)
    if "experiment" in cfg:
        print("EXPERIMENT TYPE:", type(cfg.experiment))
    print("MODEL TYPE:", type(cfg.get("model", None)))
    dataset_config = cfg["dataset"]

    if dataset_config["name"] in [
        "sbm",
        "comm20",
        "planar",
        "tree",
    ]:
        from analysis.visualization import NonMolecularVisualization
        from datasets.spectre_dataset import (
            SpectreGraphDataModule,
            SpectreDatasetInfos,
        )
        from analysis.spectre_utils import (
            PlanarSamplingMetrics,
            SBMSamplingMetrics,
            Comm20SamplingMetrics,
            TreeSamplingMetrics,
        )

        datamodule = SpectreGraphDataModule(cfg)
        if dataset_config["name"] == "sbm":
            sampling_metrics = SBMSamplingMetrics(datamodule)
        elif dataset_config["name"] == "comm20":
            sampling_metrics = Comm20SamplingMetrics(datamodule)
        elif dataset_config["name"] == "planar":
            sampling_metrics = PlanarSamplingMetrics(datamodule)
        elif dataset_config["name"] == "tree":
            sampling_metrics = TreeSamplingMetrics(datamodule)
        else:
            raise NotImplementedError(
                f"Dataset {dataset_config['name']} not implemented"
            )

        dataset_infos = SpectreDatasetInfos(datamodule, dataset_config)

        train_metrics = TrainAbstractMetricsDiscrete()
        visualization_tools = NonMolecularVisualization(dataset_name=cfg.dataset.name)

        # Get ablation flags from config if available
        ablation_flags = None
        if hasattr(cfg.model, 'ablation_study') and cfg.model.ablation_study.enabled:
            ablation_flags = cfg.model.ablation_study.components

        extra_features = ExtraFeatures(
            cfg.model.extra_features,
            cfg.model.rrwp_steps,
            dataset_info=dataset_infos,
            ablation_flags=ablation_flags,
        )
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )

    elif dataset_config["name"] in ["qm9", "guacamol", "moses", "zinc"]:
        from metrics.molecular_metrics import (
            TrainMolecularMetrics,
            SamplingMolecularMetrics,
        )
        from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
        from models.extra_features_molecular import ExtraMolecularFeatures
        from analysis.visualization import MolecularVisualization

        if "qm9" in dataset_config["name"]:
            from datasets import qm9_dataset

            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
            dataset_smiles = qm9_dataset.get_smiles(
                cfg=cfg,
                datamodule=datamodule,
                dataset_infos=dataset_infos,
                evaluate_datasets=False,
            )
        elif dataset_config["name"] == "guacamol":
            from datasets import guacamol_dataset

            datamodule = guacamol_dataset.GuacamolDataModule(cfg)
            dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)
            dataset_smiles = guacamol_dataset.get_smiles(
                raw_dir=datamodule.train_dataset.raw_dir,
                filter_dataset=cfg.dataset.filter,
            )

        elif dataset_config.name == "moses":
            from datasets import moses_dataset

            datamodule = moses_dataset.MosesDataModule(cfg)
            dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
            dataset_smiles = moses_dataset.get_smiles(
                raw_dir=datamodule.train_dataset.raw_dir,
                filter_dataset=cfg.dataset.filter,
            )
        elif "zinc" in dataset_config["name"]:
            from datasets import zinc_dataset

            datamodule = zinc_dataset.ZINCDataModule(cfg)
            dataset_infos = zinc_dataset.ZINCinfos(datamodule=datamodule, cfg=cfg)
            dataset_smiles = zinc_dataset.get_smiles(
                cfg=cfg,
                datamodule=datamodule,
                dataset_infos=dataset_infos,
                evaluate_datasets=False,
            )
        else:
            raise ValueError("Dataset not implemented")

        # Get ablation flags from config if available
        ablation_flags = None
        if hasattr(cfg.model, 'ablation_study') and cfg.model.ablation_study.enabled:
            ablation_flags = cfg.model.ablation_study.components

        extra_features = (
            ExtraFeatures(
            cfg.model.extra_features,
            cfg.model.rrwp_steps,
            dataset_info=dataset_infos,
            ablation_flags=ablation_flags,
            )
            if cfg.model.extra_features is not None
            else DummyExtraFeatures()
        )
        
        domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)

        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )

        train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)

        # We do not evaluate novelty during training
        add_virtual_states = "absorbing" == cfg.model.transition
        sampling_metrics = SamplingMolecularMetrics(
            dataset_infos, dataset_smiles, cfg, add_virtual_states=add_virtual_states
        )
        visualization_tools = MolecularVisualization(
            cfg.dataset.remove_h, dataset_infos=dataset_infos
        )

    elif dataset_config["name"] == "tls":
        from datasets import tls_dataset
        from metrics.tls_metrics import TLSSamplingMetrics
        from analysis.visualization import NonMolecularVisualization

        datamodule = tls_dataset.TLSDataModule(cfg)
        dataset_infos = tls_dataset.TLSInfos(datamodule=datamodule)

        train_metrics = TrainAbstractMetricsDiscrete()

        # Get ablation flags from config if available
        ablation_flags = None
        if hasattr(cfg.model, 'ablation_study') and cfg.model.ablation_study.enabled:
            ablation_flags = cfg.model.ablation_study.components

        extra_features = (
            ExtraFeatures(
                cfg.model.extra_features,
                cfg.model.rrwp_steps,
                dataset_info=dataset_infos,
                ablation_flags=ablation_flags,
            )
            if cfg.model.extra_features is not None
            else DummyExtraFeatures()
        )
        domain_features = DummyExtraFeatures()

        sampling_metrics = TLSSamplingMetrics(datamodule)

        visualization_tools = NonMolecularVisualization(dataset_name=cfg.dataset.name)

        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )

    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    dataset_infos.compute_reference_metrics(
        datamodule=datamodule,
        sampling_metrics=sampling_metrics,
    )

    visualization_tools = None # I hard commented this to save storage, uncomment for standard visualization

    model_kwargs = {
        "dataset_infos": dataset_infos,
        "train_metrics": train_metrics,
        "sampling_metrics": sampling_metrics,
        "visualization_tools": visualization_tools,
        "extra_features": extra_features,
        "domain_features": domain_features,
        "test_labels": (
            datamodule.test_labels
            if ("qm9" in cfg.dataset.name and cfg.general.conditional)
            else None
        ),
    }

    utils.create_folders(cfg)
    if cfg.model.name == 'defog':
        model = GraphDiscreteFlowModel(cfg=cfg, **model_kwargs)
    elif cfg.model.name == 'digress':
        model = DiscreteDenoisingDiffusion(
            cfg=cfg,
            dataset_infos=dataset_infos,
            train_metrics=train_metrics,
            sampling_metrics=sampling_metrics,
            visualization_tools=visualization_tools,
            extra_features=extra_features,
            domain_features=domain_features
        )

    # # Initialize lazy modules if any

    print("Model: ", model)
    print("Cx: ", dataset_infos.input_dims["X"])
    print("Ce: ", dataset_infos.input_dims["E"])

    callbacks = []
    if cfg.train.save_model:
        #checkpoint_dir = getattr(cfg.general, 'checkpoint_path', f"checkpoints/{cfg.general.name}")
        checkpoint_dir = str(cfg.hier_dir) + "/checkpoints/"
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch}",
            save_top_k=-1,
            every_n_epochs=cfg.general.sample_every_val
            * cfg.general.check_val_every_n_epochs,
        )
        callbacks.append(checkpoint_callback)

    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    name = cfg.general.name
    if name == "debug":
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    #use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    trainer = Trainer(
        gradient_clip_val=cfg.train.clip_grad,
        strategy="ddp_find_unused_parameters_true",  # Needed to load old DeFoG checkpoints
        accelerator="gpu",
        devices=cfg.general.gpus,
        max_epochs=cfg.train.n_epochs,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
        fast_dev_run=name == "debug",
        enable_progress_bar=False,
        callbacks=callbacks,
        log_every_n_steps=50 if name != "debug" else 1,
        logger=[],
    )

    print("="*50)
    print("Model parameter summary:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("="*50)

    if not cfg.general.test_only and cfg.general.generated_path is None:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
    else:
        # Start by evaluating test_only_path
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            print("Directory:", directory)
            files_list = os.listdir(directory)
            for file in files_list:
                if ".ckpt" in file:
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == cfg.general.test_only:
                        continue
                    print("Loading checkpoint", ckpt_path)
                    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
