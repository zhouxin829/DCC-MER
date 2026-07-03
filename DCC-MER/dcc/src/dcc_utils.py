# src/utils.py
import torch
import os
import random
import numpy as np






# =========================================================
# innovation switch
# =========================================================
def innovation_enabled(args) -> bool:
    flags = [
        getattr(args, "use_tplr", False),
        getattr(args, "use_pcrp", False),
        getattr(args, "use_rccr", False),
    ]
    return any(flags)

def save_load_name(args, names):
    model_name = names['model_name']
    tag = "OURS" if innovation_enabled(args) else "BASE"
    return f"{args.dataset}_{tag}_{model_name}"

def pseudo_tag(args) -> str:
    return "OURS" if innovation_enabled(args) else "BASE"


def get_pseudo_label_filename(args, split: str) -> str:
    # split: train / valid / test
    tag = pseudo_tag(args)
    return f"{args.dataset}_{tag}_{split}_pseudo_labels.pkl"


def get_pseudo_label_path(args, split: str) -> str:
    pseudo_dir = resolve_pseudo_dir(args)
    return os.path.join(pseudo_dir, get_pseudo_label_filename(args, split))


def get_all_pseudo_label_paths(args):
    return {
        "train": get_pseudo_label_path(args, "train"),
        "valid": get_pseudo_label_path(args, "valid"),
        "test": get_pseudo_label_path(args, "test"),
    }

def _to_history_dir(path: str, history_basename: str) -> str:
    """
    Convert a base dir (e.g., .../savemodel) into .../savemodel_history
    in a robust way. If basename doesn't match, append '_history'.
    """
    path = os.path.normpath(path)
    parent = os.path.dirname(path)
    base = os.path.basename(path)

    # Common case: user uses ".../savemodel"
    if base == "savemodel" and history_basename == "savemodel_history":
        return os.path.join(parent, "savemodel_history")

    # Fallback: append suffix
    if not base.endswith("_history"):
        base = base + "_history"
    return os.path.join(parent, base)


def resolve_model_dir(args) -> str:
    return args.model_path



def resolve_pseudo_dir(args) -> str:
    root = os.getcwd()
    return os.path.join(root, "pseudo_labels")



# =========================================================
# Save / Load
# =========================================================
def save_model(args, model, names):
    name = save_load_name(args, names)
    model_dir = resolve_model_dir(args)
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model, f'{model_dir}/{name}.pt')
    print(f"Saved model at {model_dir}/{name}.pt!")


def load_model(args, names):
    name = save_load_name(args, names)
    model_dir = resolve_model_dir(args)
    print(f"Loading model at {model_dir}/{name}.pt!")
    model = torch.load(f'{model_dir}/{name}.pt', weights_only=False)
    return model


def seed_everything(args):
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)

    torch.manual_seed(args.seed)
    if not args.no_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.use_deterministic_algorithms(True)

def transfer_models(new_model, pretrained_models):
    pretrained_t_model, pretrained_a_model, pretrained_v_model = pretrained_models
    new_dict = new_model.state_dict()

    t_model = torch.load(pretrained_t_model, map_location=torch.device('cuda'), weights_only=False)
    pretrain_t_dict = t_model.state_dict()
    t_proj_state_dict = {}
    t_enc_state_dict = {}
    for k, v in pretrain_t_dict.items():
        if k in [
            "proj1.weight",
            "proj1.bias",
            "proj2.weight",
            "proj2.bias",
            "out_layer.weight",
            "out_layer.bias"
        ]:
            k_list = k.split('.')
            k_list[0] = k_list[0] + 's.0'
            new_k = '.'.join(k_list)
            t_proj_state_dict[new_k] = v
        else:
            t_enc_state_dict[k] = v
    new_dict.update(t_proj_state_dict)
    new_dict.update(t_enc_state_dict)

    a_model = torch.load(pretrained_a_model, map_location=torch.device('cuda'), weights_only=False)
    pretrain_a_dict = a_model.state_dict()
    a_proj_state_dict = {}
    a_enc_state_dict = {}
    for k, v in pretrain_a_dict.items():
        if k in [
            "proj1.weight",
            "proj1.bias",
            "proj2.weight",
            "proj2.bias",
            "out_layer.weight",
            "out_layer.bias"
        ]:
            k_list = k.split('.')
            k_list[0] = k_list[0] + 's.1'
            new_k = '.'.join(k_list)
            a_proj_state_dict[new_k] = v
        else:
            a_enc_state_dict[k] = v
    new_dict.update(a_proj_state_dict)
    new_dict.update(a_enc_state_dict)

    v_model = torch.load(pretrained_v_model, map_location=torch.device('cuda'), weights_only=False)
    pretrain_v_dict = v_model.state_dict()
    v_proj_state_dict = {}
    v_enc_state_dict = {}
    for k, v in pretrain_v_dict.items():
        if k in [
            "proj1.weight",
            "proj1.bias",
            "proj2.weight",
            "proj2.bias",
            "out_layer.weight",
            "out_layer.bias"
        ]:
            k_list = k.split('.')
            k_list[0] = k_list[0] + 's.2'
            new_k = '.'.join(k_list)
            v_proj_state_dict[new_k] = v
        else:
            v_enc_state_dict[k] = v
    new_dict.update(v_proj_state_dict)
    new_dict.update(v_enc_state_dict)

    new_model.load_state_dict(new_dict)
    return new_model
