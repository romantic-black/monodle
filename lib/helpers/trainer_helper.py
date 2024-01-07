import os
import tqdm
import datetime
import torch
import numpy as np
import torch.nn as nn

from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.save_helper import save_checkpoint
from lib.losses.centernet_loss import compute_centernet3d_loss
from torch.utils.tensorboard import SummaryWriter


class Trainer(object):
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 train_loader,
                 test_loader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.writer = SummaryWriter(log_dir=os.path.join(self.cfg['log_dir'],
                                                         datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # loading pretrain/resume model
        if cfg.get('pretrain_model'):
            assert os.path.exists(cfg['pretrain_model'])
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=cfg['pretrain_model'],
                            map_location=self.device,
                            logger=self.logger)

        if cfg.get('resume_model', None):
            assert os.path.exists(cfg['resume_model'])
            self.epoch = load_checkpoint(model=self.model.to(self.device),
                                         optimizer=self.optimizer,
                                         filename=cfg['resume_model'],
                                         map_location=self.device,
                                         logger=self.logger)
            self.lr_scheduler.last_epoch = self.epoch - 1

        self.gpu_ids = list(map(int, cfg['gpu_ids'].split(',')))
        self.model = torch.nn.DataParallel(model, device_ids=self.gpu_ids).to(self.device)



    def train(self):
        start_epoch = self.epoch

        progress_bar = tqdm.tqdm(range(start_epoch, self.cfg['max_epoch']), dynamic_ncols=True, leave=True, desc='epochs')
        for epoch in range(start_epoch, self.cfg['max_epoch']):
            # reset random seed
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            # train one epoch
            self.train_one_epoch()
            self.epoch += 1

            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()


            # save trained model
            if (self.epoch % self.cfg['save_frequency']) == 0:
                os.makedirs('checkpoints', exist_ok=True)
                ckpt_name = os.path.join('checkpoints', 'checkpoint_epoch_%d' % self.epoch)
                save_checkpoint(get_checkpoint_state(self.model, self.optimizer, self.epoch), ckpt_name)

            progress_bar.update()

        return None


    def train_one_epoch(self):
        self.model.train()
        stat_dict = {}
        disp_dict = {}
        progress_bar = tqdm.tqdm(total=len(self.train_loader), leave=(self.epoch+1 == self.cfg['max_epoch']), desc='iters')
        for batch_idx, (inputs, targets, _) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            for key in targets.keys():
                targets[key] = targets[key].to(self.device)

            # train one batch
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            total_loss, stats_batch = compute_centernet3d_loss(outputs, targets)
            total_loss.backward()
            self.optimizer.step()

            for key in stats_batch.keys():
                if key not in stat_dict.keys():
                    stat_dict[key] = 0

                if isinstance(stats_batch[key], int):
                    stat_dict[key] += (stats_batch[key])
                else:
                    stat_dict[key] += (stats_batch[key]).detach()

            for key in stats_batch.keys():
                if key not in disp_dict.keys():
                    disp_dict[key] = 0
                # disp_dict[key] += loss_terms[key]
                if isinstance(stats_batch[key], int):
                    disp_dict[key] += (stats_batch[key])
                else:
                    disp_dict[key] += (stats_batch[key]).detach()
            if trained_batch % self.cfg['disp_frequency'] == 0:
                log_str = 'BATCH[%04d/%04d]' % (trained_batch, len(self.train_loader))
                for key in sorted(disp_dict.keys()):
                    disp_dict[key] = disp_dict[key] / self.cfg_train['disp_frequency']
                    log_str += ' %s:%.4f,' % (key, disp_dict[key])
                    disp_dict[key] = 0  # reset statistics
                self.logger.info(log_str)
            progress_bar.update()
            trained_batch = batch_idx + 1
        progress_bar.close()
        for key in stat_dict.keys():
            stat_dict[key] /= trained_batch
            self.writer.add_scalar(f'train/{key}', stat_dict[key], self.epoch)

    def record_val_loss(self):

        self.model.eval()
        stat_dict = {}
        progress_bar = tqdm.tqdm(total=len(self.test_loader), leave=True, desc='Val Progress')
        with torch.no_grad():
            for batch_idx, (inputs, calibs, coord_ranges, targets, info) in enumerate(self.test_loader):
                # load evaluation data and move data to current device.
                if type(inputs) != dict:
                    inputs = inputs.to(self.device)
                else:
                    for key in inputs.keys(): inputs[key] = inputs[key].to(self.device)
                calibs = calibs.to(self.device)
                coord_ranges = coord_ranges.to(self.device)
                for key in targets.keys(): targets[key] = targets[key].to(self.device)
                # the outputs of centernet
                outputs = self.model(inputs, coord_ranges, calibs, targets, K=50)
                total_loss, loss_terms = compute_centernet3d_loss(outputs, targets)
                for key in loss_terms.keys():
                    if key not in stat_dict.keys():
                        stat_dict[key] = 0

                    if isinstance(loss_terms[key], int):
                        stat_dict[key] += (loss_terms[key])
                    else:
                        stat_dict[key] += (loss_terms[key]).detach()
                trained_batch = batch_idx + 1
                progress_bar.update()
            progress_bar.close()
            for key in stat_dict.keys():
                stat_dict[key] /= trained_batch
                self.writer.add_scalar(f'val/{key}', stat_dict[key], self.epoch)
        return stat_dict
