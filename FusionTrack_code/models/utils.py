
import torch
import copy
import math
import torch.distributed
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP
from utils.utils import is_distributed, distributed_rank, is_main_process


def save_checkpoint(model: nn.Module, path: str, states: dict = None,
                    optimizer: optim = None, scheduler: optim.lr_scheduler = None):
    # print(f"[Debug] Saving model of type: {type(model)}")
    # print(f"[Debug] Has .module? {hasattr(model, 'module')}")
    model = get_model(model)
    if is_main_process():
        save_state = {
            "model": model.state_dict(),
            "optimizer": None if optimizer is None else optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            'states': states
        }
        torch.save(save_state, path)
    else:
        pass
    return


def load_checkpoint(model: nn.Module, path: str, states: dict = None,
                    optimizer: optim = None, scheduler: optim.lr_scheduler = None,
                    strict: bool = False):

    load_state = torch.load(path, map_location="cpu")

    if is_main_process():
        try:
            # 尝试严格加载
            incompatible_keys = model.load_state_dict(load_state["model"], strict=strict)
            
            if not strict:
                # 非严格模式下，报告不匹配的键
                if incompatible_keys.missing_keys:
                    print(f"[WARNING] Missing keys when loading checkpoint from {path}:")
                    for key in incompatible_keys.missing_keys[:10]:  # 只显示前10个
                        print(f"  - {key}")
                    if len(incompatible_keys.missing_keys) > 10:
                        print(f"  ... and {len(incompatible_keys.missing_keys) - 10} more keys")
                
                if incompatible_keys.unexpected_keys:
                    print(f"[WARNING] Unexpected keys when loading checkpoint from {path}:")
                    for key in incompatible_keys.unexpected_keys[:10]:  # 只显示前10个
                        print(f"  - {key}")
                    if len(incompatible_keys.unexpected_keys) > 10:
                        print(f"  ... and {len(incompatible_keys.unexpected_keys) - 10} more keys")
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"\n[ERROR] Size mismatch when loading checkpoint from {path}")
                print(f"[INFO] This is likely because the model architecture has changed.")
                print(f"[INFO] Attempting to load with strict=False to skip mismatched parameters...")
                
                # 手动过滤掉维度不匹配的参数
                model_state = model.state_dict()
                checkpoint_state = load_state["model"]
                
                # 过滤掉维度不匹配的参数
                filtered_state = {}
                mismatched_keys = []
                for key, value in checkpoint_state.items():
                    if key in model_state:
                        if model_state[key].shape == value.shape:
                            filtered_state[key] = value
                        else:
                            mismatched_keys.append(
                                f"{key}: checkpoint {value.shape} vs model {model_state[key].shape}"
                            )
                    else:
                        # 键不存在于当前模型
                        pass
                
                # 加载过滤后的状态
                model.load_state_dict(filtered_state, strict=False)
                
                print(f"[INFO] Successfully loaded {len(filtered_state)}/{len(checkpoint_state)} parameters")
                if mismatched_keys:
                    print(f"[INFO] Skipped {len(mismatched_keys)} parameters due to size mismatch:")
                    for key_info in mismatched_keys[:5]:  # 只显示前5个
                        print(f"  - {key_info}")
                    if len(mismatched_keys) > 5:
                        print(f"  ... and {len(mismatched_keys) - 5} more")
            else:
                raise e
    else:
        pass
    
    if optimizer is not None:
        try:
            optimizer.load_state_dict(load_state["optimizer"])
        except Exception as e:
            print(f"[WARNING] Failed to load optimizer state: {e}")
    
    if scheduler is not None:
        try:
            scheduler.load_state_dict(load_state["scheduler"])
        except Exception as e:
            print(f"[WARNING] Failed to load scheduler state: {e}")
    
    if states is not None:
        states.update(load_state["states"])
    
    return


def get_activation_layer(activation: str):
    if activation == "ReLU":
        return nn.ReLU(True)
    elif activation == "GELU":
        return nn.GELU()
    else:
        raise ValueError(f"Do not support activation layer: {activation}")


def get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])


def get_model(model):
    return model if is_distributed() is False else model.module


# I think I do not use this function at all...
def query_masks_to_attn_mask(query_mask: torch.Tensor, n_heads: int, src_len: int):
    attn_mask = torch.ones((query_mask.shape[0], 1, query_mask.shape[1], query_mask.shape[1]),
                           dtype=torch.bool,
                           device=query_mask.device)
    for b in range(query_mask.shape[0]):
        usefull_length = sum(~query_mask[b]).item()
        attn_mask[b, :, :usefull_length, :usefull_length] = False
    attn_mask = attn_mask.repeat(1, n_heads, 1, 1)
    attn_mask = attn_mask.reshape(query_mask.shape[0]*n_heads, query_mask.shape[1], query_mask.shape[1])
    return attn_mask


def pos_to_pos_embed(pos, num_pos_feats: int = 64, temperature: int = 10000, scale: float = 2 * math.pi):#将坐标值转换成高维的正弦位置编码
    pos = pos * scale
    dim_i = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_i = temperature ** (2 * (torch.div(dim_i, 2, rounding_mode="trunc")) / num_pos_feats)
    pos_embed = pos[..., None] / dim_i      # (N, M, n_feats) or (B, N, M, n_feats)
    pos_embed = torch.stack((pos_embed[..., 0::2].sin(), pos_embed[..., 1::2].cos()), dim=-1)
    pos_embed = torch.flatten(pos_embed, start_dim=-3)
    return pos_embed


def load_pretrained_model(model: nn.Module, pretrained_path: str, show_details: bool = False):
    if not is_main_process():
        return model
    pretrained_checkpoint = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    pretrained_state_dict = pretrained_checkpoint["model"]
    model_state_dict = model.state_dict()

    pretrained_keys = list(pretrained_state_dict.keys())
    for k in pretrained_keys:
        if k in model_state_dict:
            if model_state_dict[k].shape != pretrained_state_dict[k].shape:
                if "class_embed" in k:
                    if model_state_dict[k].shape[0] == 1:
                        pretrained_state_dict[k] = pretrained_state_dict[k][1:2]
                    elif model_state_dict[k].shape[0] == 2:
                        pretrained_state_dict[k] = pretrained_state_dict[k][1:3]
                    elif model_state_dict[k].shape[0] == 3:
                        pretrained_state_dict[k] = pretrained_state_dict[k][1:4]
                    elif model_state_dict[k].shape[0] == 8:     # BDD100K
                        pretrained_state_dict[k] = model_state_dict[k]
                        # We directly do not use the pretrained class embed for BDD100K
                    else:
                        raise NotImplementedError('invalid shape: {}'.format(model_state_dict[k].shape))
                else:
                    print(f"Parameter {k} has shape{pretrained_state_dict[k].shape} in pretrained model, "
                          f"but get shape{model_state_dict[k].shape} in current model.")
        elif "query_embed" in k:
            if pretrained_state_dict[k].shape == model_state_dict["det_query_embed"].shape:
                pretrained_state_dict["det_query_embed"] = pretrained_state_dict[k].clone()
            else:
                print(f"Det Query shape is not equal. Check if you turn on 'USE_DAB'.")
                pretrained_state_dict["det_query_embed"] = model_state_dict["det_query_embed"]
            del pretrained_state_dict[k]
        elif "tgt_embed" in k:  # for DAB
            if pretrained_state_dict[k].shape == model_state_dict["det_query_embed"].shape:
                pretrained_state_dict["det_query_embed"] = pretrained_state_dict[k].clone()
            else:
                pretrained_state_dict["det_query_embed"] = model_state_dict["det_query_embed"]
            del pretrained_state_dict[k]
        elif "refpoint_embed" in k:
            if pretrained_state_dict[k].shape == model_state_dict["det_anchor"].shape:
                pretrained_state_dict["det_anchor"] = pretrained_state_dict[k].clone()
            else:
                pretrained_state_dict["det_anchor"] = model_state_dict["det_anchor"]
                print(f"Pretrain model's query num is {pretrained_state_dict[k].shape[0]}, "
                      f"current model's query num is {model_state_dict['det_anchor'].shape[0]}, "
                      f"do not load these parameters.")
            del pretrained_state_dict[k]
        elif "backbone" in k:
            new_k = k[15:]
            new_k = "backbone.backbone.backbone" + new_k
            pretrained_state_dict[new_k] = pretrained_state_dict[k].clone()
            del pretrained_state_dict[k]
        elif "input_proj" in k:
            new_k = k[10:]
            new_k = "feature_projs" + new_k
            pretrained_state_dict[new_k] = pretrained_state_dict[k].clone()
            del pretrained_state_dict[k]
        else:
            pass

    not_in_model = 0
    for k in pretrained_state_dict:
        if k not in model_state_dict:
            not_in_model += 1
            if show_details:
                print(f"Parameter {k} in the pretrained model but not in the current model.")

    not_in_pretrained = 0
    for k in model_state_dict:
        if k not in pretrained_state_dict:
            pretrained_state_dict[k] = model_state_dict[k]
            not_in_pretrained += 1
            if show_details:
                print(f"There is a new parameter {k} in the current model, but not in the pretrained model.")

    model.load_state_dict(state_dict=pretrained_state_dict, strict=False)
    print(f"Pretrained model is loaded, there are {not_in_model} parameters droped "
          f"and {not_in_pretrained} parameters unloaded, set 'show details' True to see more details.")

    return model


def logits_to_scores(logits: torch.Tensor):
    return logits.sigmoid()
