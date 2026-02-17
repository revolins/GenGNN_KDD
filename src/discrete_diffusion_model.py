import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time
#import wandb
import os
import numpy as np
import hashlib
import random

try:
    from magnipy import Magnipy
except ImportError:
    print("Magnipy failed to import -- (re)install?", flush=True)
    Magnipy = None

from collections import Counter
from models.transformer_model import GraphTransformer
from models.gnn_model import GraphConvolution
from models.ppgn_model import GraphPPGN
from diffusion.noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete,\
    MarginalUniformTransition
from diffusion import diffusion_utils
from metrics.train_metrics import TrainLossDiscrete
from metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL
from src import utils
from tqdm import tqdm


class DiscreteDenoisingDiffusion(pl.LightningModule):
    def __init__(
        self,
        cfg,
        dataset_infos,
        train_metrics,
        sampling_metrics,
        visualization_tools,
        extra_features,
        domain_features,
        test_labels=None
    ):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg

        # Auto-configure magdiff and feature_extractor for test_only mode
        if hasattr(cfg.general, 'test_only') and cfg.general.test_only is not None:
            # Set default magdiff config if not present
            #'integration': 'trapz','absolute_area': True, 'scale': False
            if not hasattr(cfg, 'magdiff'):
                cfg.magdiff = {
                    'batch_size': 256,
                    'enabled': True,
                    'subset_enabled': True,
                    'subset_min_nodes': None,
                    'subset_max_nodes': None,
                    'subset_sample_size': 512,
                    'magnipy': {
                        'n_ts': 30,
                        'log_scale': False,
                        'scale_finding': 'convergence',
                        'target_prop': 0.95,
                        'metric': 'euclidean',
                        'method': 'pinv'
                    }
                }

            # Set default feature_extractor config if not present
            if not hasattr(cfg, 'feature_extractor'):
                cfg.feature_extractor = {
                    'enabled': True,
                    'checkpoint_path': None,  # Use random initialization
                    'n_layers': 7,
                    # Keep feature extractor dims aligned with dataset feature space
                    # so embeddings match reference metrics computed from dataset graphs.
                    'hidden_dims': 128, #dataset_infos.output_dims['X'],
                    'dropout': 0.0,
                    'num_classes': dataset_infos.output_dims['X'],
                    'input_dim': dataset_infos.output_dims['X'],
                    'edge_dim': dataset_infos.output_dims['E'],
                    'freeze_weights': True,
                    'orthogonal_init': True
                }

        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps
        self.test_labels = test_labels

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train)

        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        if "gt" in cfg.model.type:
            self.model = GraphTransformer(n_layers=cfg.model.n_layers,
                                        input_dims=input_dims,
                                        hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                        hidden_dims=cfg.model.hidden_dims,
                                        output_dims=output_dims,
                                        act_fn_in=nn.ReLU(),
                                        act_fn_out=nn.ReLU(),
                                        gen_model=str(cfg.model.name))
        elif "conv" in cfg.model.type:
            ablation_flags = None
            if hasattr(cfg.model, 'ablation_study') and cfg.model.ablation_study.enabled:
                ablation_flags = cfg.model.ablation_study.components
            self.model = GraphConvolution(n_layers=cfg.model.n_layers,
                                    input_dims=input_dims,
                                    hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                    hidden_dims=cfg.model.hidden_dims,
                                    output_dims=output_dims,
                                    act_fn_in=nn.ReLU(),
                                    act_fn_out=nn.ReLU(),
                                    dropout=cfg.model.dropout,
                                    ablation_flags=ablation_flags,
                                )
        elif "ppgn" in cfg.model.type:
            self.model = GraphPPGN(n_layers=cfg.model.n_layers,
                                    input_dims=input_dims,
                                    hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                    hidden_dims=cfg.model.hidden_dims,
                                    output_dims=output_dims,
                                    act_fn_in=nn.ReLU(),
                                    act_fn_out=nn.ReLU(),
                                    gen_model=str(cfg.model.name),
                                ) 
        else: raise AssertionError("No layer type selected, choose cfg.model.type from 'gt', 'conv', 'ppgn' ")

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps)

        if cfg.model.transition == 'uniform':
            self.transition_model = DiscreteUniformTransition(x_classes=self.Xdim_output, e_classes=self.Edim_output,
                                                              y_classes=self.ydim_output)
            x_limit = torch.ones(self.Xdim_output) / self.Xdim_output
            e_limit = torch.ones(self.Edim_output) / self.Edim_output
            y_limit = torch.ones(self.ydim_output) / self.ydim_output
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)
        elif cfg.model.transition == 'marginal':

            node_types = self.dataset_info.node_types.float()
            x_marginals = node_types / torch.sum(node_types)

            edge_types = self.dataset_info.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)
            print(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")
            self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                              y_classes=self.ydim_output)
            self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                                y=torch.ones(self.ydim_output) / self.ydim_output)

        self.save_hyperparameters(ignore=['train_metrics', 'sampling_metrics'])
        self.get_output_base_path = lambda: getattr(cfg, 'hier_dir', cfg.general.name)
        self.start_epoch_time = None
        self.val_start_time = None
        self.test_start_time = None
        self.sampling_start_time = None
        self.train_epoch_times = []
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0

    def load_state_dict(self, state_dict, strict=True):
        """Sanity Check strictness for model state dict loading"""
        return super().load_state_dict(state_dict, strict=strict)

    def training_step(self, data, i):
        if data.edge_index.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return

        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        loss = self.train_loss(masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
                               true_X=X, true_E=E, true_y=data.y,
                               log=i % self.log_every_steps == 0)

        self.train_metrics(masked_pred_X=pred.X, masked_pred_E=pred.E, true_X=X, true_E=E,
                           log=i % self.log_every_steps == 0)

        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                 weight_decay=self.cfg.train.weight_decay)

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        self.print("Size of the input features", self.Xdim, self.Edim, self.ydim)
        # if self.local_rank == 0:
        #     utils.setup_wandb(self.cfg)

    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        if self.start_epoch_time == None: self.start_epoch_time = 0.0
        training_epoch_time = time.time() - self.start_epoch_time
        self.train_epoch_times.append(training_epoch_time)
        to_log = self.train_loss.log_epoch_metrics()
        if self.start_epoch_time == None: self.start_epoch_time = time.time()
        self.print(f"Epoch {self.current_epoch}: X_CE: {to_log['train_epoch/x_CE'] :.3f}"
                      f" -- E_CE: {to_log['train_epoch/E_CE'] :.3f} --"
                      f" y_CE: {to_log['train_epoch/y_CE'] :.3f}"
                      f" -- {training_epoch_time:.1f}s ")
        epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: {epoch_at_metrics} -- {epoch_bond_metrics}")
        # DEBUG
        # if torch.cuda.is_available():
        #     print(torch.cuda.memory_summary())
        # else:
        #     print("CUDA is not available. Skipping memory summary.")

    def on_validation_epoch_start(self) -> None:
        self.val_start_time = time.time()
        self.val_nll.reset()
        self.val_X_kl.reset()
        self.val_E_kl.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()
        self.sampling_metrics.reset()

    def validation_step(self, data, i):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y,  node_mask, test=False)
        return {'loss': nll}

    def on_validation_epoch_end(self) -> None:
        metrics = [self.val_nll.compute(), self.val_X_kl.compute() * self.T, self.val_E_kl.compute() * self.T,
                   self.val_X_logp.compute(), self.val_E_logp.compute()]
        # if wandb.run:
        #     wandb.log({"val/epoch_NLL": metrics[0],
        #                "val/X_kl": metrics[1],
        #                "val/E_kl": metrics[2],
        #                "val/X_logp": metrics[3],
        #                "val/E_logp": metrics[4]}, commit=False)

        self.print(f"Epoch {self.current_epoch}: Val NLL {metrics[0] :.2f} -- Val Atom type KL {metrics[1] :.2f} -- ",
                   f"Val Edge type KL: {metrics[2] :.2f}")

        # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        val_nll = metrics[0]
        self.log("val/epoch_NLL", val_nll, sync_dist=True)

        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
        self.print('Val loss: %.4f \t Best val loss:  %.4f\n' % (val_nll, self.best_val_nll))

        # Store results
        to_log = {
            "val_NLL": metrics[0],
            "val_X_kl": metrics[1],
            "val_E_kl": metrics[2],
            "val_X_logp": metrics[3],
            "val_E_logp": metrics[4],
        }
        filename = os.path.join(
            self.get_output_base_path(),
            f"val_epoch{self.current_epoch}_res_0.0_general.txt",
        )
        
        self.val_counter += 1
        if self.val_counter % self.cfg.general.sample_every_val == 0:
            start = time.time()
            samples_left_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_save = self.cfg.general.samples_to_save
            chains_left_to_save = self.cfg.general.chains_to_save

            samples = []
            labels = []

            ident = 0
            while samples_left_to_generate > 0:
                bs = 2 * self.cfg.train.batch_size
                to_generate = min(samples_left_to_generate, bs)
                to_save = min(samples_left_to_save, bs)
                chains_save = min(chains_left_to_save, bs)
                cur_samples, cur_labels = self.sample_batch(batch_id=ident, batch_size=to_generate, num_nodes=None,
                                                 save_final=to_save,
                                                 keep_chain=chains_save,
                                                 number_chain_steps=self.number_chain_steps)
                samples.extend(cur_samples)
                labels.extend(cur_labels)
                ident += to_generate

                samples_left_to_save -= to_save
                samples_left_to_generate -= to_generate
                chains_left_to_save -= chains_save
            self.print("Computing sampling metrics...")
            to_log_sampling = self.evaluate_samples(samples=samples, labels=None, is_test=False)
            sampling_time = time.time() - start
            self.print(f'Done. Sampling took {sampling_time:.2f} seconds\n')

            print("Validation epoch end ends...")

        if self.val_counter % self.cfg.general.sample_every_val == 0:
            combined_to_log = {**to_log, **to_log_sampling}
            combined_to_log["sampling_time"] = f"{sampling_time:.2f}"
            # Add training time statistics
            if self.train_epoch_times:
                combined_to_log["train_time_max"] = f"{max(self.train_epoch_times):.2f}"
                combined_to_log["train_time_min"] = f"{min(self.train_epoch_times):.2f}"
                combined_to_log["train_time_avg"] = f"{sum(self.train_epoch_times)/len(self.train_epoch_times):.2f}"
            with open(filename, "w") as file:
                for key, value in combined_to_log.items():
                    file.write(f"{key}: {value}\n")

        val_epoch_time = time.time() - self.val_start_time
        if self.val_counter % self.cfg.general.sample_every_val == 0:
            # Re-open file to add epoch time
            filename = os.path.join(
                self.get_output_base_path(),
                f"val_epoch{self.current_epoch}_res_0.0_general.txt",
            )
            with open(filename, "a") as file:
                file.write(f"validation_epoch_time: {val_epoch_time:.2f}\n")

    def on_test_epoch_start(self) -> None:
        self.print("Starting test...")
        self.test_start_time = time.time()
        self.test_nll.reset()
        self.test_X_kl.reset()
        self.test_E_kl.reset()
        self.test_X_logp.reset()
        self.test_E_logp.reset()
        # if self.local_rank == 0:
        #     utils.setup_wandb(self.cfg)

    def test_step(self, data, i):
        # Standard log-likelihood and MMD testing
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y, node_mask, test=True)
        return {'loss': nll}

    def on_test_epoch_end(self) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        metrics = [self.test_nll.compute(), self.test_X_kl.compute(), self.test_E_kl.compute(),
                   self.test_X_logp.compute(), self.test_E_logp.compute()]
        # if wandb.run:
        #     wandb.log({"test/epoch_NLL": metrics[0],
        #                "test/X_kl": metrics[1],
        #                "test/E_kl": metrics[2],
        #                "test/X_logp": metrics[3],
        #                "test/E_logp": metrics[4]}, commit=False)

        self.print(f"Epoch {self.current_epoch}: Test NLL {metrics[0] :.2f} -- Test Atom type KL {metrics[1] :.2f} -- ",
                   f"Test Edge type KL: {metrics[2] :.2f} -- Test Node log prob {metrics[3] :.2f}",
                   f"Test Edge log prob {metrics[4] :.2f}")

        test_nll = metrics[0]
        # if wandb.run:
        #     wandb.log({"test/epoch_NLL": test_nll}, commit=False)

        self.print(f'Test loss: {test_nll :.4f}')

        to_log = {
            "test_NLL": metrics[0],
            "test_X_kl": metrics[1],
            "test_E_kl": metrics[2],
            "test_X_logp": metrics[3],
            "test_E_logp": metrics[4],
        }
        filename = os.path.join(
            self.get_output_base_path(),
            f"test_epoch{self.current_epoch}_res_0.0_general.txt",
        )

        self.sampling_start_time = time.time()
        samples_left_to_generate = self.cfg.general.final_model_samples_to_generate * self.cfg.general.num_sample_fold
        samples_left_to_save = self.cfg.general.final_model_samples_to_save
        chains_left_to_save = self.cfg.general.final_model_chains_to_save

        samples = []
        labels = []
        id = 0
        while samples_left_to_generate > 0:
            self.print(f'Samples left to generate: {samples_left_to_generate}/'
                       f'{self.cfg.general.final_model_samples_to_generate}', end='', flush=True)
            bs = 2 * self.cfg.train.batch_size
            to_generate = min(samples_left_to_generate, bs)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            cur_samples, cur_labels = self.sample_batch(id, to_generate, num_nodes=None, save_final=to_save,
                                             keep_chain=chains_save, number_chain_steps=self.number_chain_steps)
            samples.extend(cur_samples)
            labels.extend(cur_labels)
            id += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save
        self.print("Saving the generated graphs")
        filename = f'generated_samples1.txt'
        for i in range(2, 10):
            if os.path.exists(filename):
                filename = f'generated_samples{i}.txt'
            else:
                break
        with open(filename, 'w') as f:
            for item in samples:
                f.write(f"N={item[0].shape[0]}\n")
                atoms = item[0].tolist()
                f.write("X: \n")
                for at in atoms:
                    f.write(f"{at} ")
                f.write("\n")
                f.write("E: \n")
                for bond_list in item[1]:
                    for bond in bond_list:
                        f.write(f"{bond} ")
                    f.write("\n")
                f.write("\n")
        self.print("Generated graphs Saved. Computing sampling metrics...")
        to_log_sampling = self.evaluate_samples(samples=samples, labels=None, is_test=True)
        sampling_time = time.time() - self.sampling_start_time
        combined_to_log = {**to_log, **to_log_sampling}
        combined_to_log["sampling_time"] = f"{sampling_time:.2f}"

        filename = os.path.join(
            self.get_output_base_path(),
            f"test_epoch{self.current_epoch}_res_0.0_general.txt",
        )
        with open(filename, "w") as file:
            for key, value in combined_to_log.items():
                file.write(f"{key}: {value}\n")

        test_epoch_time = time.time() - self.test_start_time
        # Re-open file to add epoch time
        filename = os.path.join(
            self.get_output_base_path(),
            f"test_epoch{self.current_epoch}_res_0.0_general.txt",
        )
        with open(filename, "a") as file:
            file.write(f"test_epoch_time: {test_epoch_time:.2f}\n")

        self.print("Done testing.")

    def evaluate_samples(
        self,
        samples,
        labels,
        is_test,
        save_filename="",
    ):
        print("Computing sampling metrics...")

        to_log = {}
        samples_to_evaluate = self.cfg.general.final_model_samples_to_generate
        if is_test:
            for i in range(self.cfg.general.num_sample_fold):
                cur_samples = samples[
                    i * samples_to_evaluate : (i + 1) * samples_to_evaluate
                ]

                cur_to_log = self.sampling_metrics.forward(
                    cur_samples,
                    ref_metrics=self.dataset_info.ref_metrics,
                    name=f"self.name_{i}",
                    current_epoch=self.current_epoch,
                    val_counter=-1,
                    test=is_test,
                    local_rank=self.local_rank,
                    labels=labels,
                )

                if i == 0:
                    to_log = {i: [cur_to_log[i]] for i in cur_to_log}
                else:
                    to_log = {i: to_log[i] + [cur_to_log[i]] for i in cur_to_log}

                filename = os.path.join(
                    self.cfg.hier_dir,
                    f"epoch{self.current_epoch}_res_fold{i}_{save_filename}.txt",
                )
                with open(filename, "w") as file:
                    for key, value in cur_to_log.items():
                        file.write(f"{key}: {value}\n")

            to_log = {
                i: (np.array(to_log[i]).mean(), np.array(to_log[i]).std())
                for i in to_log
            }

            magdiff_enabled = (
                (hasattr(self.cfg.general, 'test_only') and self.cfg.general.test_only is not None) or
                (hasattr(self.cfg, 'magdiff') and getattr(self.cfg.magdiff, 'enabled', False))
            )

            #NOTE: MagDiff only works for unique Magnitude embeddings, meaning Tree and Planar will fail to run due to fixed node-size
            # We advise avoiding ZINC250K, GuacaMol, and MOSES due to their large number of samples leading to considerable CPU overhead
            magdiff_enabled = True if any(sub in self.cfg.dataset.name for sub in ['comm20', 'sbm', 'qm9']) else False
            print(f"Magnipy import: {Magnipy}")
            print(f"magdiff_enabled for {self.cfg.dataset.name}: {magdiff_enabled}", flush=True)
            if magdiff_enabled and Magnipy is not None:
                print("Computing MagDiff metrics...")
                test_graphs = self._get_test_graphs_from_datamodule()

                predicted_embeddings = self.extract_graph_embeddings(
                    samples, batch_size=self.cfg.magdiff.batch_size
                )

                reference_embeddings = self.extract_graph_embeddings(
                    test_graphs, batch_size=self.cfg.magdiff.batch_size
                )

                pred_shape = getattr(predicted_embeddings, 'shape', None)
                ref_shape = getattr(reference_embeddings, 'shape', None)
                print(f"MagDiff embeddings shapes: pred={pred_shape}, ref={ref_shape}", flush=True)
                if pred_shape and ref_shape:
                    print(f"Pred dims: {pred_shape[0]} x {pred_shape[1]}, Ref dims: {ref_shape[0]} x {ref_shape[1]}")

                if self._is_constant_embeddings(predicted_embeddings) or self._is_constant_embeddings(reference_embeddings):
                    print("Skipping MagDiff because embeddings are constant/degenerate.")
                    to_log["magdiff"] = 0
                    to_log["magnitude_convergence_scale"] = -1
                    to_log["magnitude_dimension"] = -1
                    return to_log

                if not self._valid_magdiff_inputs(predicted_embeddings, reference_embeddings):
                    print("Skipping MagDiff because embeddings are empty or malformed.")
                    to_log["magdiff"] = -100
                    to_log["magnitude_convergence_scale"] = -100
                    to_log["magnitude_dimension"] = -100
                    return to_log

                pred_magnipy = Magnipy(
                    predicted_embeddings,
                    **self.cfg.magdiff.magnipy
                )
                ref_magnipy = Magnipy(
                    reference_embeddings,
                    **self.cfg.magdiff.magnipy
                )

                try:
                    magdiff_value = pred_magnipy.MagDiff(ref_magnipy, scale=True)
                    magarea_pred = pred_magnipy.MagArea(scale=True)
                    magarea_ref = ref_magnipy.MagArea(scale=True)
                    to_log["magarea_pred"] = magarea_pred
                    to_log["magarea_ref"] = magarea_ref
                    to_log["magdiff"] = magdiff_value
                    ref_cardinality = getattr(reference_embeddings, "shape", [0])[0]
                    safe_ref_card = max(int(ref_cardinality), 1)
                    to_log["magdiff_norm_cardinality_ref"] = magdiff_value / safe_ref_card
                    to_log["magnitude_convergence_scale"] = ref_magnipy.get_t_conv()
                    to_log["magnitude_dimension"] = ref_magnipy.get_magnitude_dimension()
                except ValueError as magdiff_error:
                    print(f"MagDiff convergence failed: {magdiff_error}")
                    to_log["magdiff"] = -5
                    to_log["magnitude_convergence_scale"] = -5
                    to_log["magnitude_dimension"] = -5
                    return to_log

                print(f"MagDiff: {magdiff_value:.4f}")
                print(f"MagDiff per-ref-cardinality: {to_log['magdiff_norm_cardinality_ref']:.4f}")
                print(f"MagArea Pred: {magarea_pred:.4f}")
                print(f"MagArea Ref: {magarea_ref:.4f}")
                print(f"Magnitude convergence scale: {ref_magnipy.get_t_conv():.4f}")
                print(f"Magnitude dimension: {ref_magnipy.get_magnitude_dimension():.4f}")
        else:
            to_log = self.sampling_metrics.forward(
                samples,
                ref_metrics=self.dataset_info.ref_metrics,
                name=self.cfg.general.name,
                current_epoch=self.current_epoch,
                val_counter=-1,
                test=is_test,
                local_rank=self.local_rank,
                labels=labels,
            )

        return to_log

    def kl_prior(self, X, E, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)
        assert probX.shape == X.shape

        bs, n, _ = probX.shape

        limit_X = self.limit_dist.X[None, None, :].expand(bs, n, -1).type_as(probX)
        limit_E = self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)

        # Make sure that masked rows do not contribute to the loss
        limit_dist_X, limit_dist_E, probX, probE = diffusion_utils.mask_distributions(true_X=limit_X.clone(),
                                                                                      true_E=limit_E.clone(),
                                                                                      pred_X=probX,
                                                                                      pred_E=probE,
                                                                                      node_mask=node_mask)

        kl_distance_X = F.kl_div(input=probX.log(), target=limit_dist_X, reduction='none')
        kl_distance_E = F.kl_div(input=probE.log(), target=limit_dist_E, reduction='none')

        return diffusion_utils.sum_except_batch(kl_distance_X) + \
               diffusion_utils.sum_except_batch(kl_distance_E)

    def compute_Lt(self, X, E, y, pred, noisy_data, node_mask, test):
        pred_probs_X = F.softmax(pred.X, dim=-1)
        pred_probs_E = F.softmax(pred.E, dim=-1)
        pred_probs_y = F.softmax(pred.y, dim=-1)

        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], self.device)
        Qt = self.transition_model.get_Qt(noisy_data['beta_t'], self.device)

        # Compute distributions to compare with KL
        bs, n, d = X.shape
        prob_true = diffusion_utils.posterior_distributions(X=X, E=E, y=y, X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        prob_pred = diffusion_utils.posterior_distributions(X=pred_probs_X, E=pred_probs_E, y=pred_probs_y,
                                                            X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))

        # Reshape and filter masked rows
        prob_true_X, prob_true_E, prob_pred.X, prob_pred.E = diffusion_utils.mask_distributions(true_X=prob_true.X,
                                                                                                true_E=prob_true.E,
                                                                                                pred_X=prob_pred.X,
                                                                                                pred_E=prob_pred.E,
                                                                                                node_mask=node_mask)
        kl_x = (self.test_X_kl if test else self.val_X_kl)(prob_true.X, torch.log(prob_pred.X))
        kl_e = (self.test_E_kl if test else self.val_E_kl)(prob_true.E, torch.log(prob_pred.E))
        return self.T * (kl_x + kl_e)

    def reconstruction_logp(self, t, X, E, y, node_mask):
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)

        probX0 = X @ Q0.X  # (bs, n, dx_out)
        probE0 = E @ Q0.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled0 = diffusion_utils.sample_discrete_features(probX=probX0, probE=probE0, node_mask=node_mask)

        X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output).float()
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).float()
        y0 = y # originally sampled0.y and no y in definition
        assert (X.shape == X0.shape) and (E.shape == E0.shape)

        sampled_0 = utils.PlaceHolder(X=X0, E=E0, y=y0).mask(node_mask)

        # Predictions
        noisy_data = {'X_t': sampled_0.X, 'E_t': sampled_0.E, 'y_t': sampled_0.y, 'node_mask': node_mask,
                      't': torch.zeros(X0.shape[0], 1).type_as(y0)}
        extra_data = self.compute_extra_data(noisy_data)
        pred0 = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        probX0 = F.softmax(pred0.X, dim=-1)
        probE0 = F.softmax(pred0.E, dim=-1)
        proby0 = F.softmax(pred0.y, dim=-1)

        # Set masked rows to arbitrary values that don't contribute to loss
        probX0[~node_mask] = torch.ones(self.Xdim_output).type_as(probX0)
        probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = torch.ones(self.Edim_output).type_as(probE0)

        diag_mask = torch.eye(probE0.size(1)).type_as(probE0).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(probE0.size(0), -1, -1)
        probE0[diag_mask] = torch.ones(self.Edim_output).type_as(probE0)

        return utils.PlaceHolder(X=probX0, E=probE0, y=proby0)

    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data

    def compute_val_loss(self, pred, noisy_data, X, E, y, node_mask, test=False):
        """Computes an estimator for the variational lower bound.
           pred: (batch_size, n, total_features)
           noisy_data: dict
           X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
           node_mask : (bs, n)
           Output: nll (size 1)
       """
        t = noisy_data['t']

        # 1.
        N = node_mask.sum(1).long()
        log_pN = self.node_dist.log_prob(N)

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        kl_prior = self.kl_prior(X, E, node_mask)

        # 3. Diffusion loss
        loss_all_t = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test)

        # 4. Reconstruction loss
        # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        prob0 = self.reconstruction_logp(t, X, E, y, node_mask)

        loss_term_0 = self.val_X_logp(X * prob0.X.log()) + self.val_E_logp(E * prob0.E.log())

        # Combine terms
        nlls = - log_pN + kl_prior + loss_all_t - loss_term_0
        assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'

        # Update NLL metric object and return batch nll
        nll = (self.test_nll if test else self.val_nll)(nlls)        # Average over the batch

        # if wandb.run:
        #     wandb.log({"kl prior": kl_prior.mean(),
        #                "Estimator loss terms": loss_all_t.mean(),
        #                "log_pn": log_pN.mean(),
        #                "loss_term_0": loss_term_0,
        #                'batch_test_nll' if test else 'val_nll': nll}, commit=False)

        self.print({"kl prior": kl_prior.mean(),
                       "Estimator loss terms": loss_all_t.mean(),
                       "log_pn": log_pN.mean(),
                       "loss_term_0": loss_term_0,
                       'batch_test_nll' if test else 'val_nll': nll})

        return nll

    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        return self.model(X, E, y, node_mask)

    @torch.no_grad()
    def sample_batch(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int,
                     save_final: int, num_nodes=None, input_properties=None):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X, E, y = z_T.X, z_T.E, z_T.y

        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps < self.T
        chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
        chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2)))

        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)
        chain_times = torch.zeros((number_chain_steps + 1, keep_chain))

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask, input_properties)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            # Save the first keep_chain graphs
            write_index = (s_int * number_chain_steps) // self.T
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]
            chain_times[write_index] = s_norm.flatten()[:keep_chain]

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y



        # Prepare the chain for saving
        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain                  # Overwrite last frame with the resulting X, E
            chain_E[0] = final_E_chain
            chain_times[0] = torch.zeros((keep_chain))

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            # Repeat last frame to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            chain_times = torch.cat(
                [chain_times, chain_times[-1:].repeat(10, 1)], dim=0
            )
            assert chain_X.size(0) == (number_chain_steps + 10)

        molecule_list = []
        label_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])
            label_list.append(None) # naive for now

        # Visualize chains
        if self.visualization_tools is not None:
            self.print('Visualizing chains...')
            current_path = os.getcwd()
            num_molecules = chain_X.size(1)       # number of molecules
            for i in range(num_molecules):
                result_path = os.path.join(current_path, f'chains/{self.cfg.general.name}/'
                                                         f'epoch{self.current_epoch}/'
                                                         f'chains/molecule_{batch_id + i}')
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                    _ = self.visualization_tools.visualize_chain(result_path,
                                                                 chain_X[:, i, :].numpy(),
                                                                 chain_E[:, i, :].numpy(),
                                                                 chain_times[:, i].numpy())
                self.print('\r{}/{} complete'.format(i+1, num_molecules), end='', flush=True)
            self.print('\nVisualizing molecules...')

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(current_path,
                                       f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.print("Done.")

        return molecule_list, label_list

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask, input_properties=None):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0

        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=X_t,
                                                                                           Qt=Qt.X,
                                                                                           Qsb=Qsb.X,
                                                                                           Qtb=Qtb.X)

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)
        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)
    
    def _valid_magdiff_inputs(self, predicted_embeddings, reference_embeddings) -> bool:
        """Return True only when the embeddings are non-empty 2D arrays."""
        if predicted_embeddings is None or reference_embeddings is None:
            return False

        if not (hasattr(predicted_embeddings, 'shape') and hasattr(reference_embeddings, 'shape')):
            return False

        if len(predicted_embeddings.shape) != 2 or len(reference_embeddings.shape) != 2:
            return False

        if predicted_embeddings.shape[0] == 0 or reference_embeddings.shape[0] == 0:
            return False

        if predicted_embeddings.shape[1] != reference_embeddings.shape[1]:
            return False

        if predicted_embeddings.shape[1] == 0:
            return False

        return True

    def _is_constant_embeddings(self, embeddings, atol=1e-6) -> bool:
        """Return True if embeddings contain effectively identical rows."""
        if embeddings is None or embeddings.size == 0:
            return False
        if embeddings.ndim != 2:
            return False
        if embeddings.shape[0] == 1:
            return True
        diffs = np.abs(embeddings - embeddings[0:1])
        return np.max(diffs) < atol
    
    @torch.no_grad()
    def extract_graph_embeddings(self, graphs, batch_size=32):
        """
        Extract embeddings from a list of graphs using the GenGNN feature extractor.

        Parameters:
        -----------
        graphs : List[Tuple[torch.Tensor, torch.Tensor]]
            List of (X, E) tuples where X is [n_nodes] and E is [n_nodes, n_nodes]
        batch_size : int
            Batch size for processing

        Returns:
        --------
        np.ndarray
            Embeddings of shape [n_graphs, embedding_dim]
        """

        return self._extract_with_dedicated_extractor(graphs, batch_size)

    def _get_test_graphs_from_datamodule(self):
        """Load test graphs directly from the active datamodule when available."""
        datamodule = getattr(self.trainer, "datamodule", None)

        test_graphs = []
        for data_batch in tqdm(datamodule.test_dataloader()):
            dense_data, node_mask = utils.to_dense(
                data_batch.x,
                data_batch.edge_index,
                data_batch.edge_attr,
                data_batch.batch,
            )
            dense_data = dense_data.mask(node_mask, collapse=True).split(node_mask)
            for graph in dense_data:
                if graph.X.numel() == 0:
                    continue
                test_graphs.append([graph.X, graph.E])

        return test_graphs

    def _filter_magdiff_graphs(self, graphs):
        if not graphs:
            return graphs
        magdiff_cfg = getattr(self.cfg, 'magdiff', None)
        if magdiff_cfg is None or not getattr(magdiff_cfg, 'subset_enabled', False):
            return graphs

        min_nodes = getattr(magdiff_cfg, 'subset_min_nodes', None)
        max_nodes = getattr(magdiff_cfg, 'subset_max_nodes', None)
        sample_size = getattr(magdiff_cfg, 'subset_sample_size', None)

        filtered = []
        for graph in graphs:
            count = self._graph_node_count(graph)
            if min_nodes is not None and count < min_nodes:
                continue
            if max_nodes is not None and count > max_nodes:
                continue
            filtered.append(graph)

        if not filtered:
            return graphs

        if sample_size is not None and sample_size < len(filtered):
            filtered = random.sample(filtered, sample_size)

        return filtered

    def _extract_with_dedicated_extractor(self, graphs, batch_size=32):
        """Extract embeddings using dedicated GenGNNFeatureExtractor."""
        if not hasattr(self, 'feature_extractor') or self.feature_extractor is None:
            self._load_feature_extractor()

        embeddings = []

        for i in range(0, len(graphs), batch_size):
            batch_graphs = graphs[i:i + batch_size]

            # Convert to PyG format
            pyg_batch = self._convert_graphs_to_pyg_batch(batch_graphs)
            pyg_batch = pyg_batch.to(self.device)

            # Extract features
            batch_embeddings = self.feature_extractor.extract_features(pyg_batch)
            batch_numpy = batch_embeddings.cpu().numpy()

            # unique_rows = np.unique(batch_numpy, axis=0)
            # print(
            #     f"Feature extractor batch {i // batch_size}: embeddings shape {batch_numpy.shape}, "
            #     f"unique rows {unique_rows.shape[0]}",
            #     flush=True,
            # )

            embeddings.append(batch_numpy)

        result = np.vstack(embeddings) if embeddings else np.array([])
        print(
            f"Extracted total embeddings: {result.shape}, unique total {np.unique(result, axis=0).shape[0]}",
            flush=True,
        )
        return result

    def _convert_graphs_to_pyg_batch(self, graphs):
        """Convert list of (X, E) tuples to PyG Batch."""
        from torch_geometric.data import Data, Batch

        pyg_data = []
        for X, E in graphs:
            # Convert to PyG Data format
            n_nodes = len(X)

            # Create node features (one-hot if needed)
            if X.dim() == 1:
                # Assume class indices, convert to one-hot
                num_classes = self.output_dims['X']
                x = F.one_hot(X.long(), num_classes=num_classes).float()
            else:
                x = X.float()

            # Create edge features
            if E.dim() == 2:
                # Convert adjacency matrix to edge_index and edge_attr
                edge_index = E.nonzero().t()
                edge_attr = E[edge_index[0], edge_index[1]]

                # Convert edge_attr to one-hot if needed
                if edge_attr.dim() == 1:
                    num_edge_classes = self.output_dims['E']
                    edge_attr = F.one_hot(edge_attr.long(), num_classes=num_edge_classes).float()
            elif E.dim() == 3:
                # Dense adjacency with edge feature channels
                dense_edges = (E.sum(dim=-1) != 0)
                if dense_edges.any():
                    edge_index = dense_edges.nonzero().t()
                    edge_attr = E[edge_index[0], edge_index[1]]
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long)
                    edge_attr = torch.zeros((0, E.size(-1)), dtype=E.dtype)
            else:
                # Assume already in correct format
                edge_index = E
                edge_attr = torch.ones(edge_index.size(1), dtype=torch.long)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            pyg_data.append(data)

        return Batch.from_data_list(pyg_data)

    def _load_feature_extractor(self):
        """Load the dedicated feature extractor."""
        from models.gen_gnn_feature_extractor import GenGNNFeatureExtractor

        # Use randomly initialized weights
        model = GenGNNFeatureExtractor(self.cfg)
        # Weights are already orthogonally initialized in __init__ if orthogonal_init=True

        model.eval()
        self.feature_extractor = model.to(self.device)
        print(f"Loaded randomly-initialized feature extractor")


    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data['t']
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

    @torch.no_grad()
    def extract_graph_embeddings(self, graphs, batch_size=32):
        """
        Extract embeddings from a list of graphs using the GenGNN feature extractor.

        Parameters:
        -----------
        graphs : List[Tuple[torch.Tensor, torch.Tensor]]
            List of (X, E) tuples where X is [n_nodes] and E is [n_nodes, n_nodes]
        batch_size : int
            Batch size for processing

        Returns:
        --------
        np.ndarray
            Embeddings of shape [n_graphs, embedding_dim]
        """
        return self._extract_with_dedicated_extractor(graphs, batch_size)

    def _convert_graphs_to_pyg_batch(self, graphs):
        """Convert list of (X, E) tuples to PyG Batch."""
        from torch_geometric.data import Data, Batch

        pyg_data = []
        for X, E in graphs:
            # Convert to PyG Data format
            n_nodes = len(X)

            # Create node features (one-hot if needed)
            if X.dim() == 1:
                # Assume class indices, convert to one-hot
                num_classes = self.Xdim_output
                x = F.one_hot(X.long(), num_classes=num_classes).float()
            else:
                x = X.float()

            # Create edge features
            if E.dim() == 2:
                # Convert adjacency matrix to edge_index and edge_attr
                edge_index = E.nonzero().t()
                edge_attr = E[edge_index[0], edge_index[1]]

                # Convert edge_attr to one-hot if needed
                if edge_attr.dim() == 1:
                    num_edge_classes = self.Edim_output
                    edge_attr = F.one_hot(edge_attr.long(), num_classes=num_edge_classes).float()
            else:
                # Assume already in correct format
                edge_index = E
                edge_attr = torch.ones(edge_index.size(1), dtype=torch.long)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            pyg_data.append(data)

        return Batch.from_data_list(pyg_data)

    def _load_feature_extractor(self):
        """Load the dedicated feature extractor."""
        from models.gen_gnn_feature_extractor import GenGNNFeatureExtractor

        # Use randomly initialized weights
        model = GenGNNFeatureExtractor(self.cfg)
        # Weights are already orthogonally initialized in __init__ if orthogonal_init=True
        init_type = "randomly initialized"

        model.eval()
        self.feature_extractor = model.to(self.device)
        print(f"Loaded feature extractor ({init_type})")

    def _safe_softmax(self, logits_or_probs, dim=-1):
        # If already looks like probs (nonnegative, rowsum≈1), skip softmax
        x = logits_or_probs
        if (x.min() >= 0) and torch.allclose(x.sum(dim=dim), torch.ones_like(x.sum(dim=dim)), atol=1e-3):
            return x
        return F.softmax(x, dim=dim)

    def _masked_mean(self, x, mask, dim, keepdim=False, eps=1e-12):
        # mask is broadcastable to x along dim; mask in {0,1}
        w = mask.to(x.dtype)
        num = (x * w).sum(dim=dim, keepdim=keepdim)
        den = w.sum(dim=dim, keepdim=keepdim).clamp_min(eps)
        return num / den

    def _masked_std(self, x, mask, dim, keepdim=False, eps=1e-12):
        mu = self._masked_mean(x, mask, dim, keepdim=True, eps=eps)
        var = self._masked_mean((x - mu)**2, mask, dim, keepdim=keepdim, eps=eps)
        return torch.sqrt(var + 1e-12)
