import torch.nn as nn
import numpy as np
import torch
import wandb
from pathlib import Path
import torch.nn.functional as F

from .loss import kl_divergence, orthogonal_loss, ZINBLoss, MMDLoss
from .util import MetricTracker, log_gradients_to_wandb

class Trainer:
    def __init__(self, config, model, optimizer, dataloader, scheduler, device):
        self.config = config

        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device
        self.scheduler = scheduler

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period'] 
        self.cluster_epoch = cfg_trainer['cluster_period']
        self.use_vamp = cfg_trainer['use_vamp']   
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.imc_criterion_recon = nn.MSELoss()
        self.gene_criterion_recon = ZINBLoss().cuda()
        self.criterion_classification = nn.CrossEntropyLoss()
        self.mmd_loss = MMDLoss()

        wandb.init(project=config['name'], config=config)
        self.metric_tracker = MetricTracker(
            'total_loss', 'recon_loss', 'kl_loss_1', 'kl_loss_2', 'ortho_loss',
            'batch_loss_z1', 'batch_loss_z2'
        )

        self.checkpoint_dir =  config.save_dir

    def _imc_epoch(self, epoch):
        self.metric_tracker.reset()
        self.model.train()

        total_correct_z1 = 0
        total_correct_z2 = 0
        total_samples = 0

        for data, batch_id, _ in self.dataloader:
            data, batch_id = data.to(self.device), batch_id.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            if self.use_vamp:
                bio_z, z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar, batch_z, batch_mu, batch_logvar, bio_batch_pred, batch_batch_pred, reconstruction = self.model(data)
            else:
                bio_z, bio_mu, bio_logvar, batch_z, batch_mu, batch_logvar, bio_batch_pred, batch_batch_pred, reconstruction = self.model(data)

            # reconstruction loss
            recon_loss = self.imc_criterion_recon(data, reconstruction)

            # kl loss 
            if self.use_vamp:
                kl_loss_1 = self.model.bio_encoder.Vamp_KL_loss(
                                                        z1_q=z1_q, 
                                                        z1_q_mean=z1_q_mean, 
                                                        z1_q_logvar=z1_q_logvar, 
                                                        z2_q=z2_q, 
                                                        z2_q_mean=z2_q_mean, 
                                                        z2_q_logvar=z2_q_logvar, 
                                                        z1_p_mean=z1_p_mean, 
                                                        z1_p_logvar=z1_p_logvar
                                                    )
            else:
                kl_loss_1 = kl_divergence(bio_mu, bio_logvar).mean()
                bio_z_prior = torch.randn_like(bio_z, device=self.device)
                mmd_loss = self.mmd_loss(bio_z, bio_z_prior)

            kl_loss_2 = kl_divergence(batch_mu, batch_logvar).mean()

            # classifier loss
            batch_loss_z1 = self.criterion_classification(bio_batch_pred, batch_id)
            batch_loss_z2 = self.criterion_classification(batch_batch_pred, batch_id)
            
            # Orthogonal loss
            ortho_loss_value = orthogonal_loss(bio_z, batch_z)

            # Total loss
            loss = 10*recon_loss + 0.3*batch_loss_z1 + 1*batch_loss_z2 + 0.03*mmd_loss + 0.1*kl_loss_2 + 0.01*ortho_loss_value
   
            loss.backward()
            self.optimizer.step()

            # Update losses
            losses = {
                'total_loss': loss.item(),
                'recon_loss': recon_loss.item(),
                'batch_loss_z1': batch_loss_z1.item(),
                'batch_loss_z2': batch_loss_z2.item(),
                'kl_loss_1': kl_loss_1.item(),
                'kl_loss_2': kl_loss_2.item(),
                'ortho_loss': ortho_loss_value.item(),
            }
            self.metric_tracker.update_batch(losses, count=data.size(0))

            # Accuracy calculation
            z1_pred = torch.argmax(bio_batch_pred, dim=1)
            z2_pred = torch.argmax(batch_batch_pred, dim=1)

            total_correct_z1 += (z1_pred == batch_id).sum().item()
            total_correct_z2 += (z2_pred == batch_id).sum().item()

            total_samples += batch_id.size(0)
        self.scheduler.step()

        # Avg accuracy for epoch
        z1_accuracy = total_correct_z1 / total_samples * 100
        z2_accuracy = total_correct_z2 / total_samples * 100

        # log to wandb
        self.metric_tracker.log_to_wandb({
            'Z1 Accuracy': z1_accuracy,
            'Z2 Accuracy': z2_accuracy
        })

        self.logger.info(f"Epoch {epoch}: Loss = {self.metric_tracker.avg('total_loss'):.2f}, kl_loss = {self.metric_tracker.avg('kl_loss_1'):.2f}, Z1 Accuracy = {z1_accuracy:.2f}, Z2 Accuracy = {z2_accuracy:.2f}")

    def _gene_train(self, epoch):
        self.metric_tracker.reset()
        self.model.train()

        total_correct_z1 = 0
        total_correct_z2 = 0
        total_samples = 0

        for data, batch_id, _ in self.dataloader:
            data, batch_id = data.to(self.device), batch_id.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            if self.use_vamp:
                bio_z, z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar, batch_z, batch_mu, batch_logvar, bio_batch_pred, batch_batch_pred, _mean, _disp, _pi = self.model(data)
            else:
                bio_z, bio_mu, bio_logvar, batch_z, batch_mu, batch_logvar, bio_batch_pred, batch_batch_pred, _mean, _disp, _pi, size_factor, size_mu, size_logvar = self.model(data)

            # zinb loss loss
            recon_loss = self.gene_criterion_recon(data, _mean, _disp, _pi)

            # kl loss
            if self.use_vamp:
                kl_loss_1 = self.model.bio_encoder.Vamp_KL_loss(
                                                        z1_q=z1_q, 
                                                        z1_q_mean=z1_q_mean, 
                                                        z1_q_logvar=z1_q_logvar, 
                                                        z2_q=z2_q, 
                                                        z2_q_mean=z2_q_mean, 
                                                        z2_q_logvar=z2_q_logvar, 
                                                        z1_p_mean=z1_p_mean, 
                                                        z1_p_logvar=z1_p_logvar
                                                     )
            else:
                 kl_loss_1 = kl_divergence(bio_mu, bio_logvar).mean()
            kl_loss_2 = kl_divergence(batch_mu, batch_logvar).mean()
            kl_loss_size = kl_divergence(size_mu, size_logvar).mean()

            bio_z_prior = torch.randn_like(bio_z, device=self.device)
            mmd_loss = self.mmd_loss(bio_z, bio_z_prior)

            # classifier loss
            batch_loss_z1 = self.criterion_classification(bio_batch_pred, batch_id)
            batch_loss_z2 = self.criterion_classification(batch_batch_pred, batch_id)
            
            # Orthogonal loss
            ortho_loss_value = orthogonal_loss(bio_z, batch_z)

            # Total loss
            loss = 6*recon_loss + 0.05*batch_loss_z1 + 0.01*batch_loss_z2 + 0.005*kl_loss_1 + 0.001*kl_loss_2 + 0.001*ortho_loss_value + 0.01*kl_loss_size

            loss.backward()
            self.optimizer.step()

            # Update losses
            losses = {
                'total_loss': loss.item(),
                'recon_loss': recon_loss.item(),
                'batch_loss_z1': batch_loss_z1.item(),
                'batch_loss_z2': batch_loss_z2.item(),
                'kl_loss_1': kl_loss_1.item(),
                'kl_loss_2': kl_loss_2.item(),
                'ortho_loss': ortho_loss_value.item(),
            }
            self.metric_tracker.update_batch(losses, count=data.size(0))

            # Accuracy calculation
            z1_pred = torch.argmax(bio_batch_pred, dim=1)
            z2_pred = torch.argmax(batch_batch_pred, dim=1)

            total_correct_z1 += (z1_pred == batch_id).sum().item()
            total_correct_z2 += (z2_pred == batch_id).sum().item()

            total_samples += batch_id.size(0)
        self.scheduler.step()

        # Avg accuracy for epoch
        z1_accuracy = total_correct_z1 / total_samples * 100
        z2_accuracy = total_correct_z2 / total_samples * 100

        # log to wandb
        self.metric_tracker.log_to_wandb({
            'Z1 Accuracy': z1_accuracy,
            'Z2 Accuracy': z2_accuracy
        })

        self.logger.info(f"Epoch {epoch}: Loss = {self.metric_tracker.avg('total_loss'):.2f}, kl_loss = {self.metric_tracker.avg('kl_loss_1'):.2f}, Z1 Accuracy = {z1_accuracy:.2f}, Z2 Accuracy = {z2_accuracy:.2f}")

    def train(self, gene=False):
        for epoch in range(1, self.epochs):
            if gene:
                self._gene_train(epoch)
            else:
                self._imc_epoch(epoch)
            
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))


    