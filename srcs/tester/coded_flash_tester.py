"""Tester for paired on/off coded-flash restoration."""

from __future__ import annotations

import logging
import os
import time

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from srcs.utils.util import instantiate
from srcs.utils.utils_image_kair import imsave, tensor2uint


def testing(gpus, config):
    test_worker(gpus, config)


def test_worker(gpus, config):
    OmegaConf.set_struct(config, True)
    logger = logging.getLogger("test")
    os.makedirs(config.outputs_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading checkpoint: {config.checkpoint} ...")
    checkpoint = torch.load(config.checkpoint, map_location=device, weights_only=False)
    logger.info("Checkpoint loaded")

    loaded_config = OmegaConf.create(checkpoint["config"]) if "config" in checkpoint else config
    model = instantiate(loaded_config.arch)
    logger.info(model)
    if len(gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpus)
    model.load_state_dict(checkpoint["state_dict"])

    metrics = [instantiate(metric) for metric in config.metrics]
    data_loader = instantiate(config.test_data_loader)
    log = test(data_loader, model, device, metrics, config)
    logger.info(log)


def test(data_loader, model, device, metrics, config):
    model = model.to(device)
    model.eval()
    if config.get("save_img", False):
        for name in ("input", "output", "target"):
            os.makedirs(os.path.join(config.outputs_dir, name), exist_ok=True)

    total_metrics = torch.zeros(len(metrics), device=device)
    time_start = time.time()
    n_samples = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Testing")):
            batch = _move_batch(batch, device)
            coded = batch["coded_image"]
            target = batch["target"]
            output = torch.clamp(model(coded), 0, 1)

            if config.get("save_img", False):
                _save_batch_images(config.outputs_dir, batch_idx, coded, output, target)

            output_flat = torch.flatten(output, end_dim=1)
            target_flat = torch.flatten(target, end_dim=1)
            batch_size = coded.shape[0]
            for metric_idx, metric in enumerate(metrics):
                total_metrics[metric_idx] += metric(output_flat, target_flat) * batch_size
            n_samples += batch_size

    time_cost = time.time() - time_start
    code = getattr(model.module if hasattr(model, "module") else model, "ce_code", None)
    log = {"time/sample": time_cost / max(n_samples, 1)}
    if code is not None:
        log["ce_code"] = code.detach().cpu().view(-1).int().tolist()
    log.update({metric.__name__: total_metrics[idx].item() / max(n_samples, 1) for idx, metric in enumerate(metrics)})
    return log


def _save_batch_images(outputs_dir, batch_idx, coded, output, target):
    batch_size, frame_n = output.shape[:2]
    for item_idx in range(batch_size):
        sample_id = batch_idx * batch_size + item_idx + 1
        imsave(tensor2uint(coded[item_idx]), os.path.join(outputs_dir, "input", f"coded#{sample_id:04d}.jpg"))
        for frame_idx in range(frame_n):
            imsave(
                tensor2uint(output[item_idx, frame_idx]),
                os.path.join(outputs_dir, "output", f"out-frame#{sample_id:04d}-{frame_idx + 1:04d}.jpg"),
            )
            imsave(
                tensor2uint(target[item_idx, frame_idx]),
                os.path.join(outputs_dir, "target", f"gt-frame#{sample_id:04d}-{frame_idx + 1:04d}.jpg"),
            )


def _move_batch(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved
