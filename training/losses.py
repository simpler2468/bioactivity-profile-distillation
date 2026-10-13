import torch
import torch.nn.functional as F


CLASSIFICATION_ALIASES = {"classification", "class", "cls", "binary", "bce"}
REGRESSION_ALIASES = {"regression", "reg", "continuous", "mse"}


def as_task_type_tensor(task_types, num_tasks, device=None):
    """
    Return a bool tensor where True means classification and False means regression.

    If task_types is None, all tasks are treated as classification. This keeps
    Tox21/ToxCast-only experiments concise.
    """
    if task_types is None:
        return torch.ones(num_tasks, dtype=torch.bool, device=device)
    if torch.is_tensor(task_types):
        task_types = task_types.to(device=device)
        if task_types.dtype == torch.bool:
            cls_mask = task_types
        else:
            cls_mask = task_types.long() > 0
        if cls_mask.numel() != num_tasks:
            raise ValueError(f"task_types has {cls_mask.numel()} entries, expected {num_tasks}.")
        return cls_mask.bool()

    values = []
    for task_type in task_types:
        value = str(task_type).strip().lower()
        if value in CLASSIFICATION_ALIASES:
            values.append(True)
        elif value in REGRESSION_ALIASES:
            values.append(False)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    if len(values) != num_tasks:
        raise ValueError(f"task_types has {len(values)} entries, expected {num_tasks}.")
    return torch.tensor(values, dtype=torch.bool, device=device)


def valid_label_mask(labels):
    return torch.isfinite(labels.float())


def _masked_reduce(loss, mask, reduction="observation"):
    loss = loss.float()
    mask = mask.float()
    if reduction in {"observation", "observed", "mean"}:
        denom = mask.sum().clamp_min(1.0)
        return (loss * mask).sum() / denom
    if reduction in {"task", "task_balanced", "task_mean"}:
        per_task_denom = mask.sum(dim=0).clamp_min(1.0)
        per_task = (loss * mask).sum(dim=0) / per_task_denom
        valid_tasks = mask.sum(dim=0) > 0
        if valid_tasks.sum() == 0:
            return loss.sum() * 0.0
        return per_task[valid_tasks].mean()
    raise ValueError(f"Unsupported reduction: {reduction}")


def masked_bce_with_logits(logits, labels, mask=None, pos_weight=None, reduction="observation"):
    logits = logits.float()
    labels = labels.float()
    if mask is None:
        mask = valid_label_mask(labels)
    labels_clean = torch.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
    loss = F.binary_cross_entropy_with_logits(
        logits,
        labels_clean,
        reduction="none",
        pos_weight=pos_weight,
    )
    return _masked_reduce(loss, mask, reduction=reduction)


def masked_mse(pred, target, mask=None, reduction="observation"):
    pred = pred.float()
    target = target.float()
    if mask is None:
        mask = valid_label_mask(target)
    target_clean = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    loss = (pred - target_clean).pow(2)
    return _masked_reduce(loss, mask, reduction=reduction)


def compute_pos_weight_from_labels(labels, cls_mask=None):
    labels = torch.as_tensor(labels, dtype=torch.float32)
    if labels.ndim != 2:
        raise ValueError(f"labels must be 2D, got shape {tuple(labels.shape)}.")
    if cls_mask is None:
        cls_mask = torch.ones(labels.shape[1], dtype=torch.bool)
    else:
        cls_mask = torch.as_tensor(cls_mask, dtype=torch.bool).cpu()

    weights = torch.ones(labels.shape[1], dtype=torch.float32)
    for task_idx in range(labels.shape[1]):
        if not cls_mask[task_idx]:
            continue
        y = labels[:, task_idx]
        valid = torch.isfinite(y)
        y_valid = y[valid]
        pos = (y_valid == 1).sum().item()
        neg = (y_valid == 0).sum().item()
        weights[task_idx] = 1.0 if pos <= 0 else float(neg / max(pos, 1e-8))
    return weights


def _select_columns(tensor, col_mask):
    if tensor is None:
        return None
    return tensor[:, col_mask]


class MultiTaskLossComputer:
    """
    Composes hard-label and soft-label losses for mixed tasks.

    Expected tensors are shaped [batch, num_tasks]. Classification columns are
    interpreted as logits; regression columns are interpreted as raw values.
    """

    def __init__(
        self,
        task_types=None,
        task_names=None,
        hard_cls_weight=1.0,
        hard_reg_weight=1.0,
        soft_cls_weight=0.0,
        soft_reg_weight=0.0,
        hard_reduction="observation",
        aux_reduction="observation",
        pos_weight=None,
    ):
        self.task_types = task_types
        self.task_names = task_names
        self.hard_cls_weight = float(hard_cls_weight)
        self.hard_reg_weight = float(hard_reg_weight)
        self.soft_cls_weight = float(soft_cls_weight)
        self.soft_reg_weight = float(soft_reg_weight)
        self.hard_reduction = hard_reduction
        self.aux_reduction = aux_reduction
        self.pos_weight = None if pos_weight is None else torch.as_tensor(pos_weight, dtype=torch.float32)

    @classmethod
    def from_config(cls, config, task_types=None, task_names=None, pos_weight=None):
        config = dict(config or {})
        return cls(
            task_types=config.get("task_types", task_types),
            task_names=config.get("task_names", task_names),
            hard_cls_weight=config.get("hard_cls_weight", 1.0),
            hard_reg_weight=config.get("hard_reg_weight", 1.0),
            soft_cls_weight=config.get("soft_cls_weight", 0.0),
            soft_reg_weight=config.get("soft_reg_weight", 0.0),
            hard_reduction=config.get("hard_reduction", "observation"),
            aux_reduction=config.get("aux_reduction", "observation"),
            pos_weight=pos_weight,
        )

    def _task_masks(self, pred):
        cls_mask = as_task_type_tensor(self.task_types, pred.shape[1], device=pred.device)
        reg_mask = ~cls_mask
        return cls_mask, reg_mask

    def compute(
        self,
        pred,
        labels,
        teacher_pred=None,
        teacher_mask=None,
    ):
        pred = pred.float()
        labels = labels.to(device=pred.device, dtype=torch.float32)
        cls_mask, reg_mask = self._task_masks(pred)
        label_mask = valid_label_mask(labels)
        pos_weight = self.pos_weight.to(pred.device) if self.pos_weight is not None else None

        zero = pred.new_tensor(0.0)
        items = {}
        total = zero

        if cls_mask.any() and self.hard_cls_weight != 0:
            cls_pos_weight = pos_weight[cls_mask] if pos_weight is not None else None
            hard_cls = masked_bce_with_logits(
                _select_columns(pred, cls_mask),
                _select_columns(labels, cls_mask),
                mask=_select_columns(label_mask, cls_mask),
                pos_weight=cls_pos_weight,
                reduction=self.hard_reduction,
            )
        else:
            hard_cls = zero
        total = total + self.hard_cls_weight * hard_cls
        items["hard_cls_loss"] = hard_cls.detach()

        if reg_mask.any() and self.hard_reg_weight != 0:
            hard_reg = masked_mse(
                _select_columns(pred, reg_mask),
                _select_columns(labels, reg_mask),
                mask=_select_columns(label_mask, reg_mask),
                reduction=self.hard_reduction,
            )
        else:
            hard_reg = zero
        total = total + self.hard_reg_weight * hard_reg
        items["hard_reg_loss"] = hard_reg.detach()

        if teacher_pred is not None:
            teacher_pred = teacher_pred.to(device=pred.device, dtype=torch.float32)
            if teacher_mask is None:
                teacher_mask = valid_label_mask(teacher_pred)
            else:
                teacher_mask = teacher_mask.to(device=pred.device).bool()

            if cls_mask.any() and self.soft_cls_weight != 0:
                soft_cls = masked_mse(
                    _select_columns(pred, cls_mask),
                    _select_columns(teacher_pred, cls_mask),
                    mask=_select_columns(teacher_mask, cls_mask),
                    reduction=self.aux_reduction,
                )
            else:
                soft_cls = zero
            total = total + self.soft_cls_weight * soft_cls
            items["soft_cls_loss"] = soft_cls.detach()

            if reg_mask.any() and self.soft_reg_weight != 0:
                soft_reg = masked_mse(
                    _select_columns(pred, reg_mask),
                    _select_columns(teacher_pred, reg_mask),
                    mask=_select_columns(teacher_mask, reg_mask),
                    reduction=self.aux_reduction,
                )
            else:
                soft_reg = zero
            total = total + self.soft_reg_weight * soft_reg
            items["soft_reg_loss"] = soft_reg.detach()

        items["loss"] = total.detach()
        return total, items
