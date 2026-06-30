"""Tester for paired on/off coded-flash restoration."""

from __future__ import annotations

import json
import logging
import os
import time
import csv

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from srcs.utils.util import instantiate
from srcs.utils.utils_image_kair import tensor2uint


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
    gamma = float(config.get("eval_gamma", 2.2))
    save_img_space = str(config.get("save_img_space", "srgb")).lower()
    gif_duration = float(config.get("gif_duration", 0.2))
    if config.get("save_img", False):
        os.makedirs(os.path.join(config.outputs_dir, "gif"), exist_ok=True)

    total_metrics = _init_metric_totals(metrics, device)
    time_start = time.time()
    n_samples = 0
    per_scene_rows = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Testing")):
            batch = _move_batch(batch, device)
            coded = batch["coded_image"]
            target = batch["target"]
            output = torch.clamp(model(coded), 0, 1)
            target = torch.clamp(target, 0, 1)
            coded = torch.clamp(coded, 0, 1)
            coded_srgb = _linear_to_srgb(coded, gamma)
            output_srgb = _linear_to_srgb(output, gamma)
            target_srgb = _linear_to_srgb(target, gamma)

            if config.get("save_img", False):
                if save_img_space == "linear":
                    _save_batch_gifs(config.outputs_dir, batch_idx, batch, coded, output, target, gif_duration)
                else:
                    _save_batch_gifs(
                        config.outputs_dir,
                        batch_idx,
                        batch,
                        coded_srgb,
                        output_srgb,
                        target_srgb,
                        gif_duration,
                    )

            output_flat = torch.flatten(output, end_dim=1)
            target_flat = torch.flatten(target, end_dim=1)
            output_srgb_flat = torch.flatten(output_srgb, end_dim=1)
            target_srgb_flat = torch.flatten(target_srgb, end_dim=1)
            batch_size = coded.shape[0]
            for metric in metrics:
                _accumulate_metric(total_metrics, metric, output_flat, target_flat, batch_size, "linear")
                _accumulate_metric(total_metrics, metric, output_srgb_flat, target_srgb_flat, batch_size, "srgb")
            per_scene_rows.extend(
                _compute_per_scene_metrics(
                    batch,
                    output,
                    target,
                    coded_srgb,
                    output_srgb,
                    target_srgb,
                    metrics,
                    gamma,
                )
            )
            n_samples += batch_size

    time_cost = time.time() - time_start
    code = getattr(model.module if hasattr(model, "module") else model, "ce_code", None)
    log = {"time/sample": time_cost / max(n_samples, 1)}
    if code is not None:
        log["ce_code"] = code.detach().cpu().view(-1).int().tolist()
    log.update({key: value.item() / max(n_samples, 1) for key, value in total_metrics.items()})
    if per_scene_rows:
        csv_path = os.path.join(config.outputs_dir, "per_scene_metrics.csv")
        _write_metric_csv(csv_path, per_scene_rows)
        log["per_scene_metrics"] = csv_path
    metrics_json_path = os.path.join(config.outputs_dir, "metrics.json")
    metrics_csv_path = os.path.join(config.outputs_dir, "metrics.csv")
    log["metrics_json"] = metrics_json_path
    log["metrics_csv"] = metrics_csv_path
    _write_summary_json(metrics_json_path, log)
    _write_summary_csv(metrics_csv_path, log)
    return log


def _compute_per_scene_metrics(batch, output, target, coded_srgb, output_srgb, target_srgb, metrics, gamma):
    scene_ids = batch.get("scene_id", None)
    if scene_ids is None:
        scene_ids = [str(index) for index in range(output.shape[0])]
    elif isinstance(scene_ids, str):
        scene_ids = [scene_ids]

    rows = []
    for item_idx, scene_id in enumerate(scene_ids):
        output_flat = torch.flatten(output[item_idx : item_idx + 1], end_dim=1)
        target_flat = torch.flatten(target[item_idx : item_idx + 1], end_dim=1)
        output_srgb_flat = torch.flatten(output_srgb[item_idx : item_idx + 1], end_dim=1)
        target_srgb_flat = torch.flatten(target_srgb[item_idx : item_idx + 1], end_dim=1)
        row = {"scene_id": scene_id}
        for metric in metrics:
            if _should_run_linear_metric(metric):
                row[_metric_key(metric, "linear")] = float(metric(output_flat, target_flat).item())
            row[_metric_key(metric, "srgb")] = float(metric(output_srgb_flat, target_srgb_flat).item())
        row["eval_gamma"] = gamma
        row.update(_tensor_range_stats("coded_linear", batch["coded_image"][item_idx]))
        row.update(_tensor_range_stats("target_linear", target[item_idx]))
        row.update(_tensor_range_stats("output_linear", output[item_idx]))
        row.update(_tensor_range_stats("coded_srgb", coded_srgb[item_idx]))
        row.update(_tensor_range_stats("target_srgb", target_srgb[item_idx]))
        row.update(_tensor_range_stats("output_srgb", output_srgb[item_idx]))
        if "exposure_scale" in batch:
            row["exposure_scale"] = float(batch["exposure_scale"][item_idx].item())
        if "flash_energy" in batch:
            row["flash_energy"] = float(batch["flash_energy"][item_idx].item())
        rows.append(row)
    return rows


def _write_metric_csv(path, rows):
    fieldnames = ["scene_id", *[key for key in rows[0].keys() if key != "scene_id"]]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_json(path, log):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(log, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _write_summary_csv(path, log):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(log.keys()))
        writer.writeheader()
        writer.writerow({key: _csv_value(value) for key, value in log.items()})


def _csv_value(value):
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _tensor_range_stats(prefix, tensor):
    values = tensor.detach().float()
    return {
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_max": float(values.max().item()),
    }


def _init_metric_totals(metrics, device):
    totals = {}
    for metric in metrics:
        if _should_run_linear_metric(metric):
            totals[_metric_key(metric, "linear")] = torch.zeros((), device=device)
        totals[_metric_key(metric, "srgb")] = torch.zeros((), device=device)
    return totals


def _accumulate_metric(total_metrics, metric, output, target, batch_size, domain):
    if domain == "linear" and not _should_run_linear_metric(metric):
        return
    total_metrics[_metric_key(metric, domain)] += metric(output, target) * batch_size


def _metric_key(metric, domain):
    return f"{metric.__name__}_{domain}"


def _should_run_linear_metric(metric):
    return metric.__name__.lower() in {"psnr", "ssim"}


def _linear_to_srgb(tensor, gamma):
    return torch.clamp(tensor, 0, 1).pow(1.0 / gamma)


def _save_batch_gifs(outputs_dir, batch_idx, batch, coded, output, target, duration):
    batch_size, frame_n = output.shape[:2]
    scene_ids = batch.get("scene_id", None)
    gif_dir = os.path.join(outputs_dir, "gif")
    os.makedirs(gif_dir, exist_ok=True)
    for item_idx in range(batch_size):
        sample_id = _sample_tag(scene_ids, batch_idx * batch_size + item_idx + 1, item_idx)
        input_img = _to_rgb_uint(coded[item_idx])
        frames = []
        for frame_idx in range(frame_n):
            output_img = _to_rgb_uint(output[item_idx, frame_idx])
            target_img = _to_rgb_uint(target[item_idx, frame_idx])
            frames.append(np.concatenate((input_img, output_img, target_img), axis=1))
        gif_path = os.path.join(gif_dir, f"scene#{_safe_filename(sample_id)}.gif")
        imageio.mimsave(gif_path, frames, duration=duration, loop=0)


def _to_rgb_uint(tensor):
    image = tensor2uint(tensor)
    if image.ndim == 2:
        image = image[:, :, None]
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.shape[2] > 3:
        image = image[:, :, :3]
    return image


def _safe_filename(value):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))


def _sample_tag(scene_ids, fallback_id, item_idx):
    if scene_ids is None:
        return f"{fallback_id:04d}"
    if isinstance(scene_ids, str):
        return scene_ids
    return str(scene_ids[item_idx])


def _move_batch(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved
