# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""
Transformer class
"""
import math
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from rfdetr.models.ops.modules import MSDeformAttn
from typing import Tuple
from rfdetr.models.ropeAttn import RopeAttention

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def gen_sineembed_for_position(pos_tensor, dim=128):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(dim, dtype=pos_tensor.dtype, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / dim)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos

def gen_ropeembed_for_position(pos_tensor, dim=128, base=100.0):
    bs, num_queries, pos_dim = pos_tensor.shape
    device = pos_tensor.device
    dtype = pos_tensor.dtype
    scale = 2 * math.pi
    assert dim % 8 == 0, f"decoder rope dim must be divisible by 8, got {dim}"
    half_dim = dim // 2
    per_coord_dim = half_dim // 4
    dim_t = torch.arange(dim, dtype=dtype, device=device)
    periods = base ** (2 * (dim_t // 2) / dim)

    angle_list = []

    coord_order = [1, 0]
    if pos_dim == 4:
        coord_order.extend([2, 3])
    for i, coord_idx in enumerate(coord_order):
        coord = pos_tensor[..., coord_idx]  # (bs, num_queries)
        start = i * per_coord_dim
        end = (i + 1) * per_coord_dim
        periods_i = periods[start:end]
        # coord.unsqueeze(-1) -> (bs, num_queries, 1)
        # periods_i.unsqueeze(0).unsqueeze(0) -> (1, 1, per_coord_dim)
        angles = coord.unsqueeze(-1) * scale / periods_i.unsqueeze(0).unsqueeze(0)
        angle_list.append(angles)

    angles_half = torch.cat(angle_list, dim=-1) # (bs, num_queries, dim//2)
    angles = angles_half.tile(1, 1, 2)  # (bs, num_queries, dim)
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    return (sin, cos)


def gen_encoder_output_proposals(memory, memory_padding_mask, spatial_shapes, unsigmoid=True):
    """
    Input:
        - memory: bs, \sum{hw}, d_model
        - memory_padding_mask: bs, \sum{hw} (False is valid, True is padding)
        - spatial_shapes: nlevel, 2
    Output:
        - output_memory: bs, \sum{hw}, d_model (filtered memory,invalid features are set to 0)
        - output_proposals: bs, \sum{hw}, 4
    """
    N_, S_, C_ = memory.shape
    base_scale = 4.0
    proposals = []
    _cur = 0
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        if memory_padding_mask is not None:
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1) # B (each img valid_H)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1) # B (each img valid_W)
        else:
            valid_H = torch.tensor([H_ for _ in range(N_)], device=memory.device) # B (each img H)
            valid_W = torch.tensor([W_ for _ in range(N_)], device=memory.device) # B (each img W)

        grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                        torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1) # H_, W_, 2

        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale # center normalize each grid index

        wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl) # init wh each lv -> 0.05, 0.05*2, 0.05*4

        proposal = torch.cat((grid, wh), -1).view(N_, -1, 4) # init proposal x,y,w,h as center of each grid and flatten
        proposals.append(proposal)
        _cur += (H_ * W_)
    # append and filter invalid proposals
    output_proposals = torch.cat(proposals, 1) # -> bs, \sum{hw}, 4
    # filter x/y/w/h out of range [0, 1] -> bs, \sum{HW}, 1 (bool)
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)

    if unsigmoid:
        output_proposals = torch.log(output_proposals / (1 - output_proposals)) # unsigmoid to logits
        # invalid -> inf (to be filtered out in decoder)
        if memory_padding_mask is not None:
            output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))
    else:
        if memory_padding_mask is not None:
            output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float(0))

    output_memory = memory
    # fill invalid memory with 0
    if memory_padding_mask is not None:
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))

    return output_memory.to(memory.dtype), output_proposals.to(memory.dtype)


class Transformer(nn.Module):

    def __init__(self, d_model=512, sa_nhead=8, ca_nhead=8, num_queries=300,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, group_detr=1, 
                 two_stage=False,
                 num_feature_levels=4, dec_n_points=4,
                 lite_refpoint_refine=False,
                 decoder_norm_type='LN',
                 bbox_reparam=False,
                 sa_type="normal"):
        super().__init__()
        self.encoder = None

        decoder_layer = TransformerDecoderLayer(d_model, sa_nhead, ca_nhead, dim_feedforward,
                                                dropout, activation, normalize_before, 
                                                group_detr=group_detr,
                                                num_feature_levels=num_feature_levels,
                                                dec_n_points=dec_n_points,
                                                skip_self_attn=False, sa_type=sa_type)
        assert decoder_norm_type in ['LN', 'Identity']
        norm = { 
            "LN": lambda channels: nn.LayerNorm(channels),
            "Identity": lambda channels: nn.Identity(),
        }
        decoder_norm = norm[decoder_norm_type](d_model)

        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,decoder_sa_nhead=sa_nhead,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model,
                                          lite_refpoint_refine=lite_refpoint_refine,
                                          bbox_reparam=bbox_reparam)
        
        
        self.two_stage = two_stage
        if two_stage:
            self.enc_output = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(group_detr)])
            self.enc_output_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(group_detr)])

        self._reset_parameters()

        self.num_queries = num_queries
        self.d_model = d_model
        self.dec_layers = num_decoder_layers
        self.group_detr = group_detr
        self.num_feature_levels = num_feature_levels
        self.bbox_reparam = bbox_reparam

        self._export = False
    
    def export(self):
        self._export = True

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
    
    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, refpoint_embed, query_feat):
        src_flatten = []
        mask_flatten = [] if masks is not None else None
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        valid_ratios = [] if masks is not None else None
        for lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)                # bs, hw, c
            pos_embed = pos_embed.flatten(2).transpose(1, 2)    # bs, hw, c
            lvl_pos_embed_flatten.append(pos_embed)
            src_flatten.append(src)
            if masks is not None:
                mask = masks[lvl].flatten(1)                    # bs, hw
                mask_flatten.append(mask)
        memory = torch.cat(src_flatten, 1)    # bs, \sum{hxw}, c 
        if masks is not None:
            mask_flatten = torch.cat(mask_flatten, 1)   # bs, \sum{hxw}
            valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1) # each img valid ratio (bs, 2) w and h
        
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1) # bs, \sum{hxw}, c 
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=memory.device) # each level shape (bs, 2) h and w
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        # get the index of flatten tensor 0, lv1, lv2...
        
        if self.two_stage: # select high level proposal
            # get init proposals, each x,y is center of grid, wh is scaled by lv, invalid feature token is set to 0
            output_memory, output_proposals = gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes, unsigmoid=not self.bbox_reparam
            )
            # group detr for first stage
            refpoint_embed_ts, memory_ts, boxes_ts = [], [], []
            group_detr = self.group_detr if self.training else 1
            for g_idx in range(group_detr):
                # enc_output_norm: layernorm, enc_output: linear(d_model,d_model)
                output_memory_gidx = self.enc_output_norm[g_idx](self.enc_output[g_idx](output_memory))
                # self.enc_out_class_embed deepcopy from lwdetr, classify head (d_model, num_classes)
                # out: (bs, \sum{hw}, num_classes) background + foreground
                enc_outputs_class_unselected_gidx = self.enc_out_class_embed[g_idx](output_memory_gidx)
                
                # to modify proposal:
                if self.bbox_reparam: # use valid tokens to generate delta to reparam proposal
                    # self.enc_out_bbox_embed: linear(d_model, 4) deepcopy from lwdetr
                    enc_outputs_coord_delta_gidx = self.enc_out_bbox_embed[g_idx](output_memory_gidx)
                    enc_outputs_coord_cxcy_gidx = enc_outputs_coord_delta_gidx[...,
                        :2] * output_proposals[..., 2:] + output_proposals[..., :2]
                    # exp: make sure delta (0, +inf) to multi wh
                    enc_outputs_coord_wh_gidx = enc_outputs_coord_delta_gidx[..., 2:].exp() * output_proposals[..., 2:]
                    enc_outputs_coord_unselected_gidx = torch.concat(
                        [enc_outputs_coord_cxcy_gidx, enc_outputs_coord_wh_gidx], dim=-1)
                else:
                    enc_outputs_coord_unselected_gidx = self.enc_out_bbox_embed[g_idx](
                        output_memory_gidx) + output_proposals # (bs, \sum{hw}, 4) unsigmoid
                # enc_outputs_class_unselected_gidx: (bs, \sum{hw}, num_classes)
                topk = min(self.num_queries, enc_outputs_class_unselected_gidx.shape[-2]) # min queries and anchor num
                # select topk proposals index by max class score 
                topk_proposals_gidx = torch.topk(enc_outputs_class_unselected_gidx.max(-1)[0], topk, dim=1)[1] # bs, nq
                # refpoint_embed_gidx_undetach：(bs, nq, 4) topk proposals coord
                refpoint_embed_gidx_undetach = torch.gather(
                    enc_outputs_coord_unselected_gidx, 1, topk_proposals_gidx.unsqueeze(-1).repeat(1, 1, 4)) # unsigmoid
                # for decoder layer, detached as initial ones, (bs, nq, 4)
                refpoint_embed_gidx = refpoint_embed_gidx_undetach.detach()
                
                # get memory tgt
                tgt_undetach_gidx = torch.gather(
                    output_memory_gidx, 1, topk_proposals_gidx.unsqueeze(-1).repeat(1, 1, self.d_model))
                
                refpoint_embed_ts.append(refpoint_embed_gidx)
                memory_ts.append(tgt_undetach_gidx)
                boxes_ts.append(refpoint_embed_gidx_undetach)
            # concat on dim=1, the nq dimension, (bs, nq, d) --> (bs, nq, d)
            refpoint_embed_ts = torch.cat(refpoint_embed_ts, dim=1)
            # (bs, nq, d)
            memory_ts = torch.cat(memory_ts, dim=1)#.transpose(0, 1)
            boxes_ts = torch.cat(boxes_ts, dim=1)#.transpose(0, 1)
        # memory_ts is img patch tokens for topk anchor
        # boxes_ts and refpoint_embed_ts are anchor boxes and refpoints for anchor
        if self.dec_layers > 0:
            tgt = query_feat.unsqueeze(0).repeat(bs, 1, 1) # bs, num_queries, d
            refpoint_embed = refpoint_embed.unsqueeze(0).repeat(bs, 1, 1) # bs, num_queries, 4
            if self.two_stage:
                ts_len = refpoint_embed_ts.shape[-2] # num proposals for first stage
                # ts modified part
                refpoint_embed_ts_subset = refpoint_embed[..., :ts_len, :] # two stage part
                refpoint_embed_subset = refpoint_embed[..., ts_len:, :] # regular part
                # refpoint_embed nums = query nums
                if self.bbox_reparam:
                    refpoint_embed_cxcy = refpoint_embed_ts_subset[..., :2] * refpoint_embed_ts[..., 2:]
                    refpoint_embed_cxcy = refpoint_embed_cxcy + refpoint_embed_ts[..., :2]
                    refpoint_embed_wh = refpoint_embed_ts_subset[..., 2:].exp() * refpoint_embed_ts[..., 2:]
                    refpoint_embed_ts_subset = torch.concat(
                        [refpoint_embed_cxcy, refpoint_embed_wh], dim=-1
                    ) # use linear to reparam xy, use exp to reparam wh
                else:
                    refpoint_embed_ts_subset = refpoint_embed_ts_subset + refpoint_embed_ts # regular + proposal
                # two stage part use generate ref point to refine
                refpoint_embed = torch.concat(
                    [refpoint_embed_ts_subset, refpoint_embed_subset], dim=-2)

            hs, references = self.decoder(tgt, memory, memory_key_padding_mask=mask_flatten,
                            pos=lvl_pos_embed_flatten, refpoints_unsigmoid=refpoint_embed,
                            level_start_index=level_start_index, 
                            spatial_shapes=spatial_shapes,
                            valid_ratios=valid_ratios.to(memory.dtype) if valid_ratios is not None else valid_ratios)
        else:
            assert self.two_stage, "if not using decoder, two_stage must be True"
            hs = None
            references = None
        
        if self.two_stage:
            if self.bbox_reparam:
                return hs, references, memory_ts, boxes_ts
            else:
                return hs, references, memory_ts, boxes_ts.sigmoid()
        return hs, references, None, None


class TransformerDecoder(nn.Module):

    def __init__(self,
                 decoder_layer,
                 num_layers,
                 norm=None, 
                 decoder_sa_nhead=8,
                 return_intermediate=False,
                 d_model=256,
                 lite_refpoint_refine=False,
                 bbox_reparam=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.d_model = d_model
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.lite_refpoint_refine = lite_refpoint_refine
        self.bbox_reparam = bbox_reparam
        self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2) # in hidden out layer
        self.rope_ref_point_head = MLP(2 * d_model // decoder_sa_nhead, d_model, d_model, 2) # in hidden out layer
        self.decoder_sa_nhead = decoder_sa_nhead
        self._export = False
    
    def export(self):
        self._export = True

    def refpoints_refine(self, refpoints_unsigmoid, new_refpoints_delta):
        if self.bbox_reparam:
            new_refpoints_cxcy = new_refpoints_delta[..., :2] * refpoints_unsigmoid[..., 2:] + refpoints_unsigmoid[..., :2]
            new_refpoints_wh = new_refpoints_delta[..., 2:].exp() * refpoints_unsigmoid[..., 2:]
            new_refpoints_unsigmoid = torch.concat(
                [new_refpoints_cxcy, new_refpoints_wh], dim=-1
            )
        else:
            new_refpoints_unsigmoid = refpoints_unsigmoid + new_refpoints_delta
        return new_refpoints_unsigmoid

    def forward(self, 
                tgt, # queries
                memory, # img features
                tgt_mask: Optional[Tensor] = None, # None
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, # padding mask
                pos: Optional[Tensor] = None, # position
                refpoints_unsigmoid: Optional[Tensor] = None, # unsigmoid refpoints cxcywh
                # for memory
                level_start_index: Optional[Tensor] = None, # num_levels
                spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                valid_ratios: Optional[Tensor] = None):
        output = tgt 

        intermediate = []
        hs_refpoints_unsigmoid = [refpoints_unsigmoid] # store refpoints for each layer
        use_rope = True
        
        def get_reference(refpoints):
            # [batch_size, num_queries, 4]
            obj_center = refpoints[..., :4]
            # TODO: 这里使用的是sine的位置编码
            if self._export:
                if use_rope:
                    query_embed = gen_ropeembed_for_position(obj_center, dim=self.d_model // self.decoder_sa_nhead, base=100.0)
                    # sin, cos  [bs,num_queries, d_model/decoder_sa_nhead]
                else:
                    query_embed = gen_sineembed_for_position(obj_center, self.d_model / 2) # bs, nq, 256*2 
                
                refpoints_input = obj_center[:, :, None] # bs, nq, 1, 4
            else:
                refpoints_input = obj_center[:, :, None] \
                                        * torch.cat([valid_ratios, valid_ratios], -1)[:, None] # bs, nq, nlevel, 4
                if use_rope:
                    query_embed = gen_ropeembed_for_position(
                        refpoints_input[:, :, 0, :], dim=self.d_model // self.decoder_sa_nhead, base=100.0
                    )
                else:
                    query_embed = gen_sineembed_for_position(
                        refpoints_input[:, :, 0, :], self.d_model / 2) # bs, nq, 256*2 
            if use_rope:
                # query_embed: sin, cos  [bs,num_queries, d_model/decoder_sa_nhead]
                # query_pos: [bs,num_queries, d_model]
                query_pos = self.rope_ref_point_head(torch.cat([query_embed[0], query_embed[1]], dim=-1))
            else:
                query_pos = self.ref_point_head(query_embed)
            return obj_center, refpoints_input, query_pos, query_embed
        
        # always use init refpoints
        if self.lite_refpoint_refine: # only use init refpoints reduce cul
            if self.bbox_reparam:
                obj_center, refpoints_input, query_pos, query_embed = get_reference(refpoints_unsigmoid)
            else:
                obj_center, refpoints_input, query_pos, query_embed = get_reference(refpoints_unsigmoid.sigmoid())

        for layer_id, layer in enumerate(self.layers):
            # iter refine each layer
            if not self.lite_refpoint_refine:
                if self.bbox_reparam:
                    obj_center, refpoints_input, query_pos, query_embed = get_reference(refpoints_unsigmoid)
                else:
                    obj_center, refpoints_input, query_pos, query_embed = get_reference(refpoints_unsigmoid.sigmoid())

            # For the first decoder layer, we do not apply transformation over p_s
            pos_transformation = 1

            query_pos = query_pos * pos_transformation
            
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, query_embed=query_embed, 
                           is_first=(layer_id == 0),
                           reference_points=refpoints_input,
                           spatial_shapes=spatial_shapes,
                           level_start_index=level_start_index,use_rope=use_rope)

            if not self.lite_refpoint_refine:
                # box iterative update
                new_refpoints_delta = self.bbox_embed(output)
                new_refpoints_unsigmoid = self.refpoints_refine(refpoints_unsigmoid, new_refpoints_delta)
                if layer_id != self.num_layers - 1:
                    hs_refpoints_unsigmoid.append(new_refpoints_unsigmoid)
                refpoints_unsigmoid = new_refpoints_unsigmoid.detach()
                # update ref points

            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if self._export:
                # to shape: B, N, C
                hs = intermediate[-1]
                if self.bbox_embed is not None:
                    ref = hs_refpoints_unsigmoid[-1]
                else:
                    ref = refpoints_unsigmoid
                return hs, ref
            # box iterative update
            if self.bbox_embed is not None:
                return [
                    torch.stack(intermediate),
                    torch.stack(hs_refpoints_unsigmoid),
                ]
            else:
                return [
                    torch.stack(intermediate), 
                    refpoints_unsigmoid.unsqueeze(0)
                ]

        return output.unsqueeze(0)




class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, sa_nhead, ca_nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, group_detr=1, 
                 num_feature_levels=4, dec_n_points=4, 
                 skip_self_attn=False, sa_type="normal"):
        super().__init__()
        # Decoder Self-Attention
        if sa_type=="normal":
            self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=sa_nhead, dropout=dropout, batch_first=True)
        elif sa_type=="diff":
            self.self_attn = DiffAttention(embed_dim=d_model, num_heads=sa_nhead,  qkv_bias=True, attn_drop=dropout, proj_drop=dropout, lambda_init=0.8)
        
        self.rope_self_attn = RopeAttention(embed_dim=d_model, num_heads=sa_nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Decoder Cross-Attention
        self.cross_attn = MSDeformAttn(
            d_model, n_levels=num_feature_levels, n_heads=ca_nhead, n_points=dec_n_points)

        self.nhead = ca_nhead

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.group_detr = group_detr

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     query_embed = None,
                     is_first = False,
                     reference_points = None,
                     spatial_shapes=None,
                     level_start_index=None,
                     use_rope = False,
                     ):
        bs, num_queries, _ = tgt.shape
        
        # ========== Begin of Self-Attention =============
        # Apply projections here
        # shape: batch_size x num_queries x 256
        if use_rope:
            # TODO: add rope attn
            sin,cos = query_embed
            if self.training:
                queries=torch.cat(tgt.split(num_queries // self.group_detr, dim=1), dim=0)
                sin = torch.cat(sin.split(num_queries // self.group_detr, dim=1), dim=0)
                cos = torch.cat(cos.split(num_queries // self.group_detr, dim=1), dim=0)
            else:
                queries=tgt
            tgt2 = self.rope_self_attn(queries, sin, cos, key_padding_mask=tgt_key_padding_mask)[0]
        else:
            q = k = tgt + query_pos
            v = tgt
            if self.training:
                # (bs, 300*13, 256)->[(bs, 300, 256), ...]->(bs*13, 300, 256)
                q = torch.cat(q.split(num_queries // self.group_detr, dim=1), dim=0)
                k = torch.cat(k.split(num_queries // self.group_detr, dim=1), dim=0)
                v = torch.cat(v.split(num_queries // self.group_detr, dim=1), dim=0)

            tgt2 = self.self_attn(q, k, v, attn_mask=tgt_mask, # None
                                key_padding_mask=tgt_key_padding_mask,
                                need_weights=False)[0] # between group queries
        
        if self.training:
            tgt2 = torch.cat(tgt2.split(bs, dim=0), dim=1) # (bs, 300*13, 256)
        # ========== End of Self-Attention =============

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos), # queries
            reference_points, # ref points
            memory,
            spatial_shapes,
            level_start_index,
            memory_key_padding_mask
        )
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_embed = None,
                is_first = False,
                reference_points = None,
                spatial_shapes=None,
                level_start_index=None,
                use_rope = False,
                 ):
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, 
                                 query_embed, is_first,
                                 reference_points, spatial_shapes, level_start_index, use_rope)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    
    try:
        two_stage = args.two_stage
    except:
        two_stage = False
    # print("Transformer args:")
    # print(args)
    
    # 创建transformer实例
    transformer = Transformer(
        d_model=args.hidden_dim,
        sa_nhead=args.sa_nheads,
        ca_nhead=args.ca_nheads,
        num_queries=args.num_queries,
        dropout=args.dropout,
        dim_feedforward=args.dim_feedforward,
        num_decoder_layers=args.dec_layers,
        return_intermediate_dec=True,
        group_detr=args.group_detr,
        two_stage=two_stage,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        lite_refpoint_refine=args.lite_refpoint_refine,
        decoder_norm_type=args.decoder_norm,
        bbox_reparam=args.bbox_reparam,
        # sa_type=args.decoder_sa_type,
    )
    
    if hasattr(args, 'feataug_enable'):
        transformer._feataug_enable = args.feataug_enable
    if hasattr(args, 'feataug_types'):
        transformer._feataug_types = args.feataug_types
        
    return transformer


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.weight is not None}'

class DiffAttention(nn.Module):
    r"""
    Differential Attention Module.

    Given an input tensor X ∈ ℝ^(B×N×d_model), we first compute the linear projections:

        Q = X Wᵠ, K = X Wᵏ, V = X Wᵛ

    The queries and keys are then reshaped and split into two parts:
        Q → [Q₁; Q₂] ∈ ℝ^(B, N, 2·h_effective, d_head)
        K → [K₁; K₂] ∈ ℝ^(B, N, 2·h_effective, d_head)
    with h_effective = num_heads // 2 and d_head = d_model / num_heads.

    The value projection is reshaped to:
        V ∈ ℝ^(B, N, h_effective, 2·d_head)

    We then compute two attention maps:
        A₁ = softmax((Q₁ K₁ᵀ) / √d_head)
        A₂ = softmax((Q₂ K₂ᵀ) / √d_head)

    A learnable scalar λ is computed via:
        λ = exp(λ_{q1} ⋅ λ_{k1}) − exp(λ_{q2} ⋅ λ_{k2}) + λ_init

    Finally, the differential attention output is:
        DiffAttn(X) = (A₁ − λ · A₂) · V

    The per-head outputs are then normalized headwise with RMSNorm and projected back to d_model.

    Args:
        dim (int): Embedding dimension (d_model).
        num_heads (int): Number of heads in the original transformer (must be even).
        qkv_bias (bool): If True, add a bias term to the Q, K, V projections.
        attn_drop (float): Dropout probability after softmax.
        proj_drop (float): Dropout probability after the output projection.
        lambda_init (float): Initial constant for lambda re-parameterization.
    """
    def __init__(self, embed_dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0., lambda_init=0.8):
        super().__init__()
        if num_heads % 2 != 0:
            raise ValueError("num_heads must be even for Differential Attention.")
        self.dim = embed_dim
        self.num_heads = num_heads # original number of heads
        self.effective_heads = num_heads // 2  # differential attention operates on half as many heads
        self.head_dim = embed_dim // num_heads # per-head dimension
        self.scaling = self.head_dim ** -0.5

        # Linear projections for Q, K, V: mapping from embed_dim → embed_dim.
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True) # final output projection

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # RMSNorm for headwise normalization on outputs (each head's output has dimension 2·head_dim)
        self.diff_norm = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

        # Learnable lambda parameters (shared across all heads)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_init = lambda_init

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N_q, _ = q.shape
        _, N_k, _ = k.shape
        assert q.shape[-1] == self.dim and k.shape[-1] == self.dim and v.shape[-1] == self.dim, \
            f"q/k/v dim must be {self.dim} (got q_dim={q.shape[-1]}, k_dim={k.shape[-1]}, v_dim={v.shape[-1]})"

        q = q.view(B, N_q, 2 * self.effective_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N_k, 2 * self.effective_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N_k, self.effective_heads, 2 * self.head_dim).transpose(1, 2)

        q = q * self.scaling

        attn_scores = torch.matmul(q, k.transpose(-1, -2))

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.view(B, 2 * self.effective_heads, N_q, N_k)
            attn_scores = attn_scores.masked_fill(attn_mask.bool(), -1e9)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)
            attn_scores = attn_scores.masked_fill(key_padding_mask.bool(), -1e9)


        attn_probs = F.softmax(attn_scores, dim=-1)  # (B, 2×h_eff, N_q, N_k)
        attn_probs = self.attn_drop(attn_probs)

        attn_probs = attn_probs.view(B, self.effective_heads, 2, N_q, N_k)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2))
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        diff_attn = attn_probs[:, :, 0] - lambda_full * attn_probs[:, :, 1]

        attn_output = torch.matmul(diff_attn, v)  # (B, h_eff, N_q, 2×head_dim)
        attn_output = self.diff_norm(attn_output) * (1 - self.lambda_init)

        attn_output = attn_output.transpose(1, 2)  # (B, N_q, h_eff, 2×head_dim)
        attn_output = attn_output.reshape(B, N_q, self.dim)

        x_out = self.out_proj(attn_output)
        x_out = self.proj_drop(x_out)

        attn_weights = diff_attn if need_weights else None
        return x_out, attn_weights