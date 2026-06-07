"""Trainer for paired on/off coded-flash restoration."""

from __future__ import annotations

import platform

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torchvision.utils import make_grid

from .base import BaseTrainer
from srcs.logger import BatchMetrics
from srcs.model.coded_flash_model import coded_flash_reblur
from srcs.utils.util import collect, get_logger, instantiate


class Trainer(BaseTrainer):
    """Training loop for dataloaders that return coded-flash batch dicts."""

    def __init__(
        self,
        model,
        criterion,
        metric_ftns,
        optimizer,
        config,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
    ):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.limit_train_iters = config["trainer"].get("limit_train_iters", len(self.data_loader))
        if not self.limit_train_iters or self.limit_train_iters > len(self.data_loader):
            self.limit_train_iters = len(self.data_loader)
        self.limit_valid_iters = config["trainer"].get("limit_valid_iters", len(self.valid_data_loader))
        if not self.limit_valid_iters or self.limit_valid_iters > len(self.valid_data_loader):
            self.limit_valid_iters = len(self.valid_data_loader)
        args = ["loss", *[m.__name__ for m in self.metric_ftns]]
        self.train_metrics = BatchMetrics(*args, postfix="/train", writer=self.writer)
        self.valid_metrics = BatchMetrics(*args, postfix="/valid", writer=self.writer)
        self.losses = self.config["loss"]

    def _compute_loss_and_metrics(self, batch, phase: str):
        coded = batch["coded_image"]
        target = batch["target"]
        off_frames = batch["off_frames"]
        code = batch["code"]

        output = self.model(coded)
        output_flat = torch.flatten(output, end_dim=1)
        target_flat = torch.flatten(target, end_dim=1)

        loss = self.losses["main_loss"] * self.criterion["main_loss"](output_flat, target_flat)
        if "reblur_loss" in self.losses:
            reblurred = coded_flash_reblur(output, off_frames, code)
            loss = loss + self.losses["reblur_loss"] * self.criterion["reblur_loss"](reblurred, coded)

        metrics = {}
        for metric in self.metric_ftns:
            metric_value = metric(output_flat, target_flat)
            if self.config.n_gpu > 1:
                metric_value = collect(metric_value)
            metrics[metric.__name__] = metric_value
        return loss, metrics, output

    def _after_iter(self, epoch, batch_idx, phase, loss, metrics):
        self.writer.set_step(
            (epoch - 1) * getattr(self, f"limit_{phase}_iters") + batch_idx,
            speed_chk=phase,
        )
        loss_v = loss.item() if self.config.n_gpu == 1 else collect(loss)
        getattr(self, f"{phase}_metrics").update("loss", loss_v)
        for key, value in metrics.items():
            value = value.item() if isinstance(value, torch.Tensor) else value
            getattr(self, f"{phase}_metrics").update(key, value)

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        last_images = None

        for batch_idx, batch in enumerate(self.data_loader):
            batch = _move_batch(batch, self.device)
            loss, metrics, output = self._compute_loss_and_metrics(batch, "train")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_idx % self.logging_step == 0 or (batch_idx + 1) == self.limit_train_iters:
                self._after_iter(epoch, batch_idx, "train", loss, metrics)
                self.logger.info(
                    f"Train Epoch: {epoch} {self._progress(batch_idx)} "
                    f"Loss: {loss:.6f} Lr: {self.optimizer.param_groups[0]['lr']:.3e}"
                )

            last_images = (batch, output.detach())
            if (batch_idx + 1) == self.limit_train_iters:
                break

        if last_images is not None:
            self._write_epoch_images(epoch, "train", *last_images)

        log = self.train_metrics.result()
        if self.valid_data_loader is not None:
            log.update(**self._valid_epoch(epoch))
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.writer.set_step(epoch)
        for key, value in log.items():
            self.writer.add_scalar(key + "/epoch", value)
        return log

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()
        last_images = None
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_data_loader):
                batch = _move_batch(batch, self.device)
                loss, metrics, output = self._compute_loss_and_metrics(batch, "valid")
                self._after_iter(epoch, batch_idx, "valid", loss, metrics)
                last_images = (batch, output.detach())
                if (batch_idx + 1) == self.limit_valid_iters:
                    break
        if last_images is not None:
            self._write_epoch_images(epoch, "valid", *last_images)
        return self.valid_metrics.result()

    def _write_epoch_images(self, epoch, phase, batch, output):
        frame_num = output.shape[1]
        stride = max(1, frame_num // 4)
        self.writer.set_step(epoch)
        self.writer.add_image(
            f"{phase}/input",
            make_grid(batch["coded_image"][0:1].cpu(), nrow=1, normalize=True),
        )
        self.writer.add_image(
            f"{phase}/output",
            make_grid(output[0, 0::stride].cpu(), nrow=2, normalize=True),
        )
        self.writer.add_image(
            f"{phase}/target",
            make_grid(batch["target"][0, 0::stride].cpu(), nrow=2, normalize=True),
        )

    def _progress(self, batch_idx):
        total = self.data_loader.batch_size * self.limit_train_iters
        current = batch_idx * self.data_loader.batch_size
        if dist.is_initialized():
            current *= dist.get_world_size()
        return "[{}/{} ({:.0f}%)]".format(current, total, 100.0 * current / total)


def trainning(gpus, config):
    OmegaConf.set_struct(config, False)
    config.n_gpu = len(gpus)
    OmegaConf.set_struct(config, True)
    if len(gpus) > 1:
        torch.multiprocessing.spawn(multi_gpu_train_worker, nprocs=len(gpus), args=(gpus, config))
    else:
        train_worker(config)


def train_worker(config):
    OmegaConf.set_struct(config, True)
    logger = get_logger("train")
    data_loader, valid_data_loader = instantiate(config.data_loader)
    model = instantiate(config.arch)
    logger.info(model)

    criterion = {}
    if "main_loss" in config.loss:
        criterion["main_loss"] = instantiate(config.main_loss)
    if "reblur_loss" in config.loss:
        criterion["reblur_loss"] = instantiate(config.reblur_loss)
    metrics = [instantiate(metric) for metric in config["metrics"]]

    optimizer = instantiate(config.optimizer, model.parameters())
    lr_scheduler = instantiate(config.lr_scheduler, optimizer)
    trainer = Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        config=config,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
    )
    trainer.train()


def multi_gpu_train_worker(rank, gpus, config):
    config.local_rank = rank
    if platform.system() == "Windows":
        backend = "gloo"
    elif platform.system() == "Linux":
        backend = "nccl"
    else:
        raise RuntimeError("Unknown platform")
    dist.init_process_group(backend=backend, init_method="tcp://127.0.0.1:34567", world_size=len(gpus), rank=rank)
    train_worker(config)


def _move_batch(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved
