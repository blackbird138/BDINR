"""Paired on/off coded-flash data loaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler


DEFAULT_CODE = [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1]
DEFAULT_FRAME_INDICES = list(range(16))


class PairedOnOffCodedFlashDataset(Dataset):
    """Dataset that synthesizes one coded-flash image from paired on/off frames."""

    def __init__(
        self,
        root: str | Path,
        scene_paths: list[Path],
        frame_indices: list[int] | None = None,
        code: list[int] | None = None,
        patch_size: int | list[int] | tuple[int, int] | None = None,
        tform_op: str | list[str] | None = None,
        split: str = "train",
        read_source: str = "npz",
        use_mask: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.scene_paths = list(scene_paths)
        self.frame_indices = list(DEFAULT_FRAME_INDICES if frame_indices is None else frame_indices)
        self.code = np.asarray(DEFAULT_CODE if code is None else code, dtype=np.float32)
        self.patch_size = _normalize_patch_size(patch_size)
        self.tform_op = _normalize_tform_op(tform_op)
        self.split = split
        self.read_source = _normalize_read_source(read_source)
        self.use_mask = bool(use_mask)

        if self.code.ndim != 1 or not np.all(np.isin(self.code, [0.0, 1.0])):
            raise ValueError("code must be a one-dimensional binary list")
        if len(self.frame_indices) != len(self.code):
            raise ValueError("frame_indices length must match code length")
        if not self.scene_paths:
            raise ValueError("scene_paths must not be empty")

    def __len__(self) -> int:
        return len(self.scene_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        scene_path = self.scene_paths[index]
        off, on, source_meta = _read_scene_pair(scene_path, self.frame_indices, self.read_source)
        mask = _read_mask(scene_path / "mask.png", off.shape[1:3]) if self.use_mask else None
        off, on, mask = self._crop_sample(off, on, mask)
        off, on, mask = _spatial_transform_sample(off, on, mask, self.tform_op)

        code = self.code.astype(np.float32)
        coded = np.mean(off + code[:, None, None, None] * (on - off), axis=0)

        sample = {
            "coded_image": _to_chw_tensor(coded),
            "target": _to_tchw_tensor(on),
            "off_frames": _to_tchw_tensor(off),
            "on_frames": _to_tchw_tensor(on),
            "code": torch.from_numpy(code.copy()),
            "frame_indices": torch.tensor(self.frame_indices, dtype=torch.long),
            "scene_id": scene_path.name,
        }
        if mask is not None:
            sample["mask"] = _to_mask_tensor(mask)
        meta = _load_meta(scene_path / "frames_float_meta.json")
        meta.update(source_meta)
        if meta:
            sample["flash_energy"] = torch.tensor(float(meta.get("flash_energy", 0.0)), dtype=torch.float32)
            sample["exposure_scale"] = torch.tensor(float(meta.get("exposure_scale", 0.0)), dtype=torch.float32)
        return sample

    def _crop_sample(
        self,
        off: np.ndarray,
        on: np.ndarray,
        mask: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if self.patch_size is None:
            return off, on, mask
        patch_h, patch_w = self.patch_size
        height, width = off.shape[1:3]
        if height < patch_h or width < patch_w:
            raise ValueError(f"patch_size {self.patch_size} is larger than image size {(height, width)}")
        if self.split == "train":
            y0 = np.random.randint(0, height - patch_h + 1)
            x0 = np.random.randint(0, width - patch_w + 1)
        else:
            y0 = (height - patch_h) // 2
            x0 = (width - patch_w) // 2
        off = off[:, y0 : y0 + patch_h, x0 : x0 + patch_w, :]
        on = on[:, y0 : y0 + patch_h, x0 : x0 + patch_w, :]
        if mask is not None:
            mask = mask[y0 : y0 + patch_h, x0 : x0 + patch_w, :]
        return off, on, mask


def get_coded_flash_pair_loaders(
    root: str | Path,
    batch_size: int,
    frame_indices: list[int] | None = None,
    code: list[int] | None = None,
    patch_size: int | list[int] | tuple[int, int] | None = None,
    tform_op: str | list[str] | None = None,
    status: str = "train",
    shuffle: bool = True,
    num_workers: int = 8,
    pin_memory: bool = False,
    prefetch_factor: int = 2,
    train_fraction: float = 0.90,
    val_fraction: float = 0.05,
    read_source: str = "npz",
    scene_ids: str | list[str] | None = None,
    use_mask: bool = False,
) -> DataLoader | tuple[DataLoader, DataLoader]:
    """Build train/valid/test loaders from sorted scene ids."""
    root = Path(root)
    read_source = _normalize_read_source(read_source)
    status = str(status).lower()
    scene_paths = discover_complete_scenes(root, frame_indices or DEFAULT_FRAME_INDICES, read_source=read_source)
    scene_ids_requested = _has_scene_ids(scene_ids)
    scene_paths = _filter_scene_paths(scene_paths, scene_ids)

    loader_kwargs = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    if status in {"all", "infer", "inference"} or (scene_ids_requested and status in {"valid", "val", "test"}):
        dataset = PairedOnOffCodedFlashDataset(
            root,
            scene_paths,
            frame_indices,
            code,
            patch_size=None,
            tform_op=None,
            split=status,
            read_source=read_source,
            use_mask=use_mask,
        )
        sampler = DistributedSampler(dataset, shuffle=False) if dist.is_initialized() else None
        return DataLoader(dataset, shuffle=False, sampler=sampler, **loader_kwargs)

    train_scenes, val_scenes, test_scenes = split_scenes(scene_paths, train_fraction, val_fraction)

    if status == "train":
        train_dataset = PairedOnOffCodedFlashDataset(
            root,
            train_scenes,
            frame_indices,
            code,
            patch_size,
            tform_op,
            split="train",
            read_source=read_source,
            use_mask=use_mask,
        )
        val_dataset = PairedOnOffCodedFlashDataset(
            root,
            val_scenes,
            frame_indices,
            code,
            patch_size=patch_size,
            tform_op=None,
            split="valid",
            read_source=read_source,
            use_mask=use_mask,
        )
        train_sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if dist.is_initialized() else None
        train_loader = DataLoader(
            train_dataset,
            shuffle=bool(shuffle) and train_sampler is None,
            sampler=train_sampler,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            sampler=val_sampler,
            **loader_kwargs,
        )
        return train_loader, val_loader

    if status in {"valid", "val"}:
        dataset = PairedOnOffCodedFlashDataset(
            root,
            val_scenes,
            frame_indices,
            code,
            patch_size=None,
            tform_op=None,
            split="valid",
            read_source=read_source,
            use_mask=use_mask,
        )
    elif status == "test":
        dataset = PairedOnOffCodedFlashDataset(
            root,
            test_scenes,
            frame_indices,
            code,
            patch_size=None,
            tform_op=None,
            split="test",
            read_source=read_source,
            use_mask=use_mask,
        )
    else:
        raise NotImplementedError(f"status ({status}) should be 'train' | 'valid' | 'test' | 'all'")

    sampler = DistributedSampler(dataset, shuffle=False) if dist.is_initialized() else None
    return DataLoader(dataset, shuffle=False, sampler=sampler, **loader_kwargs)


def discover_complete_scenes(root: str | Path, frame_indices: list[int], read_source: str = "npz") -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"coded-flash root does not exist: {root}")
    if root.is_dir() and root.name.startswith("scene_"):
        scenes = [root]
    else:
        scenes = sorted(path for path in root.glob("scene_*") if path.is_dir())
    complete = []
    read_source = _normalize_read_source(read_source)
    for scene in scenes:
        if _scene_has_source(scene, frame_indices, read_source):
            complete.append(scene)
    if not complete:
        raise FileNotFoundError(f"No complete scene_* folders found under {root}")
    return complete


def _filter_scene_paths(scene_paths: list[Path], scene_ids: str | list[str] | None) -> list[Path]:
    if not _has_scene_ids(scene_ids):
        return scene_paths
    if isinstance(scene_ids, str):
        scene_ids = [scene_ids]
    requested = {Path(scene_id).name for scene_id in scene_ids}
    filtered = [scene for scene in scene_paths if scene.name in requested]
    found = {scene.name for scene in filtered}
    missing = sorted(requested - found)
    if missing:
        raise FileNotFoundError(f"Requested scene_ids were not found or incomplete: {missing}")
    return filtered


def _has_scene_ids(scene_ids: str | list[str] | None) -> bool:
    if scene_ids is None:
        return False
    if isinstance(scene_ids, str):
        return bool(scene_ids)
    return len(scene_ids) > 0


def split_scenes(
    scene_paths: list[Path],
    train_fraction: float = 0.90,
    val_fraction: float = 0.05,
) -> tuple[list[Path], list[Path], list[Path]]:
    total = len(scene_paths)
    if total < 3:
        raise ValueError("At least 3 complete scenes are required for train/val/test split")
    train_count = max(1, int(total * train_fraction))
    val_count = max(1, int(total * val_fraction))
    if train_count + val_count >= total:
        train_count = total - 2
        val_count = 1
    return (
        scene_paths[:train_count],
        scene_paths[train_count : train_count + val_count],
        scene_paths[train_count + val_count :],
    )


def _read_rgb(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Image read failed: {path}")
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def _read_mask(path: Path, expected_hw: tuple[int, int]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing mask.png: {path}")
    try:
        mask = np.asarray(Image.open(path))
    except Exception as exc:
        raise ValueError(f"Mask read failed: {path}") from exc
    if mask.ndim != 2:
        raise ValueError(f"mask.png must be a single-channel image: {path}")
    if tuple(mask.shape) != tuple(expected_hw):
        raise ValueError(f"mask.png shape {mask.shape} does not match frame size {expected_hw}: {path}")
    mask = mask.astype(np.float32)
    if mask.max() > 1.0:
        mask = mask / 255.0
    return np.clip(mask, 0.0, 1.0)[..., None]


def _read_scene_pair(scene_path: Path, frame_indices: list[int], read_source: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if read_source == "npz":
        try:
            off, on = _read_scene_npz(scene_path, frame_indices)
            return off, on, {"read_source": "npz"}
        except (FileNotFoundError, KeyError, ValueError):
            raise
    if read_source == "png":
        off, on = _read_scene_png(scene_path, frame_indices)
        return off, on, {"read_source": "png"}
    if read_source == "npz_or_png":
        try:
            off, on = _read_scene_npz(scene_path, frame_indices)
            return off, on, {"read_source": "npz"}
        except (FileNotFoundError, KeyError, ValueError):
            off, on = _read_scene_png(scene_path, frame_indices)
            return off, on, {"read_source": "png"}
    raise ValueError(f"Unsupported read_source: {read_source}")


def _read_scene_npz(scene_path: Path, frame_indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    path = scene_path / "frames_float.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing frames_float.npz: {path}")
    with np.load(path) as data:
        if "off" not in data or "on" not in data:
            raise KeyError(f"frames_float.npz must contain 'off' and 'on': {path}")
        off_all = np.asarray(data["off"], dtype=np.float32)
        on_all = np.asarray(data["on"], dtype=np.float32)
    if off_all.ndim != 4 or on_all.ndim != 4 or off_all.shape[-1] != 3 or on_all.shape[-1] != 3:
        raise ValueError("npz off/on arrays must have shape [T,H,W,3]")
    if off_all.shape != on_all.shape:
        raise ValueError("npz off/on arrays must have the same shape")
    if max(frame_indices) >= off_all.shape[0]:
        raise ValueError(f"frame index exceeds npz frame count {off_all.shape[0]} in {path}")
    return off_all[frame_indices].astype(np.float32), on_all[frame_indices].astype(np.float32)


def _read_scene_png(scene_path: Path, frame_indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    off_frames = []
    on_frames = []
    for frame_idx in frame_indices:
        off_frames.append(_read_rgb(scene_path / f"frame_{frame_idx:04d}_off.png"))
        on_frames.append(_read_rgb(scene_path / f"frame_{frame_idx:04d}_on.png"))
    return np.stack(off_frames, axis=0), np.stack(on_frames, axis=0)


def _scene_has_source(scene_path: Path, frame_indices: list[int], read_source: str) -> bool:
    if read_source in {"npz", "npz_or_png"} and _scene_has_npz(scene_path, frame_indices):
        return True
    if read_source in {"png", "npz_or_png"}:
        return all(
            (scene_path / f"frame_{idx:04d}_off.png").is_file()
            and (scene_path / f"frame_{idx:04d}_on.png").is_file()
            for idx in frame_indices
        )
    return False


def _scene_has_npz(scene_path: Path, frame_indices: list[int]) -> bool:
    path = scene_path / "frames_float.npz"
    # Keep scene discovery cheap. Opening a compressed NPZ and touching arrays
    # decompresses data; doing that for thousands of scenes can stall startup
    # before the model ever reaches the GPU. Full key/shape validation happens
    # when a sample is actually read in _read_scene_npz.
    return path.is_file()


def _to_chw_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()


def _to_tchw_tensor(frames: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(frames.transpose(0, 3, 1, 2))).float()


def _to_mask_tensor(mask: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(mask.transpose(2, 0, 1))).float()


def _normalize_patch_size(patch_size: int | list[int] | tuple[int, int] | None) -> tuple[int, int] | None:
    if patch_size is None:
        return None
    if isinstance(patch_size, int):
        return (patch_size, patch_size)
    if len(patch_size) != 2:
        raise ValueError("patch_size must be an int or a length-2 sequence")
    return (int(patch_size[0]), int(patch_size[1]))


def _normalize_tform_op(tform_op: str | list[str] | None) -> set[str]:
    if tform_op is None:
        return set()
    if isinstance(tform_op, str):
        if tform_op == "all":
            return {"flip", "rotate"}
        return {tform_op}
    ops = set(tform_op)
    if "all" in ops:
        ops.update({"flip", "rotate"})
    ops.discard("reverse")
    return ops


def _normalize_read_source(read_source: str) -> str:
    if read_source not in {"npz", "png", "npz_or_png"}:
        raise ValueError("read_source must be 'npz', 'png', or 'npz_or_png'")
    return read_source


def _spatial_transform_sample(
    off: np.ndarray,
    on: np.ndarray,
    mask: np.ndarray | None,
    tform_op: set[str],
    prob: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if "flip" in tform_op:
        if np.random.rand() < prob:
            off = off[:, :, ::-1, :]
            on = on[:, :, ::-1, :]
            if mask is not None:
                mask = mask[:, ::-1, :]
        if np.random.rand() < prob:
            off = off[:, ::-1, :, :]
            on = on[:, ::-1, :, :]
            if mask is not None:
                mask = mask[::-1, :, :]
    if "rotate" in tform_op:
        draw = np.random.rand()
        if prob / 4 < draw <= prob / 2:
            off = np.transpose(off, axes=(0, 2, 1, 3))[:, ::-1, ...]
            on = np.transpose(on, axes=(0, 2, 1, 3))[:, ::-1, ...]
            if mask is not None:
                mask = np.transpose(mask, axes=(1, 0, 2))[::-1, ...]
        elif prob / 2 < draw <= prob:
            off = np.transpose(off[:, ::-1, :, :][:, :, ::-1, :], axes=(0, 2, 1, 3))[:, ::-1, ...]
            on = np.transpose(on[:, ::-1, :, :][:, :, ::-1, :], axes=(0, 2, 1, 3))[:, ::-1, ...]
            if mask is not None:
                mask = np.transpose(mask[::-1, :, :][:, ::-1, :], axes=(1, 0, 2))[::-1, ...]
    return off.copy(), on.copy(), None if mask is None else mask.copy()


def _load_meta(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
