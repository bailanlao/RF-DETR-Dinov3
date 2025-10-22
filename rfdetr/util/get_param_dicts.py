# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Functions to get params dict"""
import torch.nn as nn

from rfdetr.models.backbone import Joiner


def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    """
    Calculate lr decay rate for different ViT blocks.
    
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.
    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if ".pos_embed" in name or ".patch_embed" in name:
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1
    print("name: {}, lr_decay: {}".format(name, lr_decay_rate ** (num_layers + 1 - layer_id)))
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_vit_weight_decay_rate(name, weight_decay_rate=1.0):
    if ('gamma' in name) or ('pos_embed' in name) or ('rel_pos' in name) or ('bias' in name) or ('norm' in name):
        weight_decay_rate = 0.
    print("name: {}, weight_decay rate: {}".format(name, weight_decay_rate))
    return weight_decay_rate


def get_param_dict(args, model_without_ddp: nn.Module):
    assert isinstance(model_without_ddp.backbone, Joiner)
    backbone = model_without_ddp.backbone[0]
    backbone_named_param_lr_pairs = backbone.get_named_param_lr_pairs(args, prefix="backbone.0")
    backbone_param_lr_pairs = [param_dict for _, param_dict in backbone_named_param_lr_pairs.items()]

    decoder_key = 'transformer.decoder'
    decoder_params = [
        p
        for n, p in model_without_ddp.named_parameters() if decoder_key in n and p.requires_grad
    ]

    decoder_param_lr_pairs = [
        {"params": param, "lr": args.lr * args.lr_component_decay} 
        for param in decoder_params
    ]
    
    sta_param_lr_pairs = []
    if hasattr(backbone, 'select_mode') and backbone.select_mode != 1:
        if not hasattr(args, 'lr_sta'):
            args.lr_sta = args.lr * 2.0 
        
        sta_named_param_lr_pairs = backbone.get_sta_param_lr_pairs(args, prefix="backbone.0")
        sta_param_lr_pairs = [param_dict for _, param_dict in sta_named_param_lr_pairs.items()]
        
        backbone_param_lr_pairs = [
            p for p in backbone_param_lr_pairs 
            if not any(sta_p['params'] is p['params'] for sta_p in sta_param_lr_pairs)
        ]

    other_params = []
    all_param_ids = set()
    all_param_ids.update(id(p['params']) for p in backbone_param_lr_pairs)
    all_param_ids.update(id(p['params']) for p in decoder_param_lr_pairs)
    all_param_ids.update(id(p['params']) for p in sta_param_lr_pairs)
    
    for n, p in model_without_ddp.named_parameters():
        if id(p) not in all_param_ids and p.requires_grad:
            other_params.append(p)
    
    other_param_dicts = [
        {"params": param, "lr": args.lr} 
        for param in other_params
    ]
    
    final_param_dicts = (
        other_param_dicts + backbone_param_lr_pairs + decoder_param_lr_pairs + sta_param_lr_pairs
    )

    return final_param_dicts
