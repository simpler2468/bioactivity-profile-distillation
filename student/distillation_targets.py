from pathlib import Path

import numpy as np
import torch


def _build_position_map(row_index):
    return {int(row): idx for idx, row in enumerate(np.asarray(row_index, dtype=np.int64))}


def _indices_for_rows(row_index, position_map, name):
    rows = np.asarray(row_index, dtype=np.int64)
    missing = [int(row) for row in rows if int(row) not in position_map]
    if missing:
        raise KeyError(f"{name} is missing row_index values, first: {missing[:5]}")
    return np.asarray([position_map[int(row)] for row in rows], dtype=np.int64)


def _clip_if_needed(values, clip_value):
    if clip_value is None or float(clip_value) <= 0:
        return values
    clip_value = float(clip_value)
    return np.clip(values, -clip_value, clip_value)


def _resolve_soft_profile_mask(task_indices, target_task_indices, mode):
    mode = str(mode or "full")
    if mode == "full":
        return None
    if task_indices is None:
        raise ValueError("--soft_profile_mode other than full requires selected task indices.")
    if target_task_indices is None:
        raise ValueError("--soft_profile_mode other than full requires target task indices.")
    selected = np.asarray(task_indices, dtype=np.int64)
    targets = np.asarray(target_task_indices, dtype=np.int64)
    is_target = np.isin(selected, targets)
    if mode == "target_only":
        mask = is_target
    elif mode == "non_target":
        mask = ~is_target
    else:
        raise ValueError(f"Unsupported soft_profile_mode: {mode}")
    if not np.any(mask):
        raise ValueError(f"soft_profile_mode={mode} leaves no teacher-profile dimensions.")
    return mask


class DistillationTargetProvider:
    """
    Row-index aligned teacher targets for student soft-label loss.

    The provider only loads train split targets because soft-label loss is used
    during training. Evaluation still uses hard labels only.
    """

    def __init__(
        self,
        teacher_profile_dir=None,
        task_indices=None,
        target_task_indices=None,
        soft_profile_mode="full",
        target_clip=None,
    ):
        self.teacher_profile_dir = Path(teacher_profile_dir) if teacher_profile_dir else None
        self.task_indices = None if task_indices is None else np.asarray(task_indices, dtype=np.int64)
        self.target_task_indices = (
            None if target_task_indices is None else np.asarray(target_task_indices, dtype=np.int64)
        )
        self.soft_profile_mode = str(soft_profile_mode or "full")
        self.soft_profile_mask = _resolve_soft_profile_mask(
            self.task_indices,
            self.target_task_indices,
            self.soft_profile_mode,
        )
        self.target_clip = target_clip

        self.teacher_pred = None
        self.teacher_mask = None
        self.teacher_position = None
        if self.teacher_profile_dir is not None:
            self._load_teacher()

    @property
    def has_teacher(self):
        return self.teacher_pred is not None

    def _select_tasks(self, values):
        if self.task_indices is None:
            return values
        return values[:, self.task_indices]

    def _load_teacher(self):
        arrays_path = self.teacher_profile_dir / "profile_arrays.npz"
        if not arrays_path.exists():
            raise FileNotFoundError(arrays_path)
        arrays = np.load(arrays_path, allow_pickle=True)
        raw = self._select_tasks(arrays["train_oof_raw_score"].astype(np.float32))
        raw = _clip_if_needed(raw, self.target_clip).astype(np.float32)
        self.teacher_pred = raw
        self.teacher_mask = np.isfinite(raw).astype(np.float32)
        if self.soft_profile_mask is not None:
            self.teacher_mask[:, ~self.soft_profile_mask] = 0.0
        self.teacher_position = _build_position_map(arrays["train_oof_row_index"])

    def batch(self, row_index, device):
        if torch.is_tensor(row_index):
            rows = row_index.detach().cpu().numpy().astype(np.int64)
        else:
            rows = np.asarray(row_index, dtype=np.int64)

        out = {}
        if self.has_teacher:
            pos = _indices_for_rows(rows, self.teacher_position, "teacher profile")
            teacher_pred = self.teacher_pred[pos]
            teacher_mask = self.teacher_mask[pos]
            out["teacher_pred"] = torch.as_tensor(teacher_pred, dtype=torch.float32, device=device)
            out["teacher_mask"] = torch.as_tensor(teacher_mask, dtype=torch.bool, device=device)

        return out


def validate_distillation_args(args):
    if getattr(args, "soft_cls_weight", 0.0) and not getattr(args, "teacher_profile_dir", None):
        raise ValueError("--teacher_profile_dir is required when --soft_cls_weight is non-zero.")
