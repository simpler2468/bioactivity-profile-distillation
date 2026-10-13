import re


FEATURE_ALIASES = {
    "Avalon": "av",
    "ECFP4": "ec4",
    "descriptor": "desc",
    "Avalon_descriptor": "avd",
    "ECFP4_descriptor": "e4d",
    "graph": "g",
    "in_vitro_feature": "ivl",
    "in_vitro_feature_prob": "ivp",
    "Avalon_in_vitro_zscore": "av_ivz",
    "Avalon_in_vitro_raw": "av_ivr",
    "Avalon_in_vitro_prob": "av_ivp",
    "taskwise_selected_hpo_mlp_platt": "twmlp",
    "taskwise_selected_hpo_mlp_only_platt": "twmlpo",
}

MODEL_ALIASES = {
    "MLP": "mlp",
    "RF": "rf",
    "SVM": "svm",
    "GIN": "gin",
    "Teacher": "tea",
    "ToxACoL": "tac",
}

TASK_ALIASES = {
    "tox21_12": "t12",
    "tox21_12_plus_20": "t12p20",
    "tox21_12_plus_40": "t12p40",
    "tox21_12_plus_60": "t12p60",
    "all_93": "a93",
}

LOSS_ALIASES = {
    "mask_bce": "bce",
    "mask_bce_soft": "soft",
    "soft": "soft",
}

SPLIT_ALIASES = {
    "contain_valid_ecfp4_similarity_hpo": "cve",
    "contain_valid_graph_hpo": "cvg",
    "contain_valid_task_count_ablation": "tcab",
    "contain_valid_aux_soft": "axs",
    "contain_valid_aux_soft_target_only": "axst",
    "contain_valid_aux_soft_non_target": "axsn",
    "contain_valid_aux_soft_pc1": "axp1",
    "contain_valid_aux_soft_pc1_5": "axp15",
    "contain_valid_aux_soft_pc1_10": "axp110",
    "contain_valid_aux_soft_pc2_10": "axp210",
    "train_oof_auc": "oofauc",
    "mixed": "mix",
    "canonical_in_vivo": "civ",
}

METHOD_ALIASES = {
    "zscore_kmeans": "zkm",
}

KEY_ALIASES = {
    "feature": "f",
    "model": "m",
    "tasks": "t",
    "loss": "l",
    "split": "sp",
    "fold": "fd",
    "seed": "s",
    "profile": "p",
    "method": "meth",
}

REVERSE_KEY_ALIASES = {value: key for key, value in KEY_ALIASES.items()}


def sanitize_id(value):
    return str(value).replace(" ", "-").replace("/", "-").replace("\\", "-")


def _short_seed(value):
    text = sanitize_id(value)
    if text.isdigit() and len(text) == 4 and text.startswith("20"):
        return text[-2:]
    return text


def _alias(value, aliases):
    text = sanitize_id(value)
    return aliases.get(text, text)


def short_value(key, value):
    text = sanitize_id(value)
    if key == "feature":
        return _alias(value, FEATURE_ALIASES)
    if key == "model":
        return _alias(value, MODEL_ALIASES)
    if key == "tasks":
        return _alias(value, TASK_ALIASES)
    if key == "loss":
        return _alias(value, LOSS_ALIASES)
    if key == "method":
        return _alias(value, METHOD_ALIASES)
    if key == "seed":
        return _short_seed(value)
    if key == "split":
        m = re.fullmatch(r"hpo_aux_mlp_(.+)_(tox21_12|all_93)_(soft)_trial_(\d+)", text)
        if m:
            feature, tasks, loss_mode, trial = m.groups()
            return f"ax_mlp_{short_value('feature', feature)}_{short_value('tasks', tasks)}_{short_value('loss', loss_mode)}_tr{trial}"
        m = re.fullmatch(r"hpo_aux_graph_(GIN)_(tox21_12|all_93)_(soft)_trial_(\d+)", text)
        if m:
            model, tasks, loss_mode, trial = m.groups()
            return f"ax_{short_value('model', model)}_g_{short_value('tasks', tasks)}_{short_value('loss', loss_mode)}_tr{trial}"
        m = re.fullmatch(r"hpo_student_mlp_(.+)_(tox21_12|all_93)_trial_(\d+)", text)
        if m:
            feature, tasks, trial = m.groups()
            return f"hpo_mlp_{short_value('feature', feature)}_{short_value('tasks', tasks)}_tr{trial}"
        m = re.fullmatch(r"hpo_student_(GIN)_graph_(tox21_12|all_93)_trial_(\d+)", text)
        if m:
            model, tasks, trial = m.groups()
            return f"hpo_{short_value('model', model)}_g_{short_value('tasks', tasks)}_tr{trial}"
        return _alias(value, SPLIT_ALIASES)
    return text


def short_kv_id(**parts):
    tokens = []
    for key, value in parts.items():
        if value is None:
            continue
        short_key = KEY_ALIASES.get(key, key)
        tokens.append(f"{short_key}={short_value(key, value)}")
    return "__".join(tokens)


def student_run_id(feature, model, tasks, loss, split, seed):
    return short_kv_id(feature=feature, model=model, tasks=tasks, loss=loss, split=split, seed=seed)


def teacher_run_id(feature, model, split, fold, seed):
    return short_kv_id(feature=feature, model=model, split=split, fold=int(fold), seed=seed)


def teacher_profile_id(feature, model, split, seed):
    return short_kv_id(feature=feature, model=model, split=split, seed=seed)


def toxacol_run_id(feature, split, fold, seed):
    return short_kv_id(feature=feature, model="ToxACoL", split=split, fold=int(fold), seed=seed)


def parse_kv_name(name):
    parts = {}
    for item in str(name).split("__"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        parts[REVERSE_KEY_ALIASES.get(key, key)] = value
    return parts


def compact_hpo_dir_name(name):
    name = str(name)
    m = re.fullmatch(r"student_aux_MLP_(.+)_(tox21_12|all_93)_(soft)", name)
    if m:
        feature, tasks, loss_mode = m.groups()
        return f"ax_mlp_{short_value('feature', feature)}_{short_value('tasks', tasks)}_{short_value('loss', loss_mode)}"
    m = re.fullmatch(r"student_aux_(GIN)_graph_(tox21_12|all_93)_(soft)", name)
    if m:
        model, tasks, loss_mode = m.groups()
        return f"ax_{short_value('model', model)}_g_{short_value('tasks', tasks)}_{short_value('loss', loss_mode)}"
    m = re.fullmatch(r"student_MLP_(.+)_(tox21_12|all_93)", name)
    if m:
        feature, tasks = m.groups()
        return f"hpo_mlp_{short_value('feature', feature)}_{short_value('tasks', tasks)}"
    m = re.fullmatch(r"student_(GIN)_graph_(tox21_12|all_93)", name)
    if m:
        model, tasks = m.groups()
        return f"hpo_{short_value('model', model)}_g_{short_value('tasks', tasks)}"
    m = re.fullmatch(r"teacher_MLP_(.+)", name)
    if m:
        return f"hpo_t_mlp_{short_value('feature', m.group(1))}"
    return name
