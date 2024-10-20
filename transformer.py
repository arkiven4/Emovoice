# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import SeparableConv
from commons import mask_from_lens


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()
        self.demb = demb
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.matmul(torch.unsqueeze(pos_seq, -1),
                                    torch.unsqueeze(self.inv_freq, 0))
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=1)
        if bsz is not None:
            return pos_emb[None, :, :].expand(bsz, -1, -1)
        else:
            return pos_emb[None, :, :]


class PositionwiseConvFF(nn.Module):
    def __init__(self, d_model, d_inner, kernel_size, dropout, sepconv=False, pre_lnorm=False):
        super(PositionwiseConvFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        if sepconv:
            self.conv_fn = SeparableConv
        else:
            self.conv_fn = nn.Conv1d

        self.CoreNet = nn.Sequential(
            self.conv_fn(d_model, d_inner, kernel_size, 1, (kernel_size // 2)),
            nn.ReLU(),
            # nn.Dropout(dropout),  # worse convergence
            self.conv_fn(d_inner, d_model, kernel_size, 1, (kernel_size // 2)),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        return self._forward(inp)

    def _forward(self, inp):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = inp.transpose(1, 2)
            core_out = self.CoreNet(self.layer_norm(core_out).to(inp.dtype))
            core_out = core_out.transpose(1, 2)

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = inp.transpose(1, 2)
            core_out = self.CoreNet(core_out)
            core_out = core_out.transpose(1, 2)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out).to(inp.dtype)

        return output


class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0.1, pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.scale = 1 / (d_head ** 0.5)
        self.pre_lnorm = pre_lnorm

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head)
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inp, attn_mask=None):
        return self._forward(inp, attn_mask)

    def _forward(self, inp, attn_mask=None):
        residual = inp

        if self.pre_lnorm:
            # layer normalization
            inp = self.layer_norm(inp)

        n_head, d_head = self.n_head, self.d_head

        head_q, head_k, head_v = torch.chunk(self.qkv_net(inp), 3, dim=2)
        head_q = head_q.view(inp.size(0), inp.size(1), n_head, d_head)
        head_k = head_k.view(inp.size(0), inp.size(1), n_head, d_head)
        head_v = head_v.view(inp.size(0), inp.size(1), n_head, d_head)

        q = head_q.permute(2, 0, 1, 3).reshape(-1, inp.size(1), d_head)
        k = head_k.permute(2, 0, 1, 3).reshape(-1, inp.size(1), d_head)
        v = head_v.permute(2, 0, 1, 3).reshape(-1, inp.size(1), d_head)

        attn_score = torch.bmm(q, k.transpose(1, 2))
        attn_score.mul_(self.scale)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).to(attn_score.dtype)
            attn_mask = attn_mask.repeat(n_head, attn_mask.size(2), 1)
            attn_score.masked_fill_(attn_mask.to(torch.bool), -float('inf'))

        attn_prob = F.softmax(attn_score, dim=2)
        attn_prob = self.dropatt(attn_prob)
        attn_vec = torch.bmm(attn_prob, v)

        attn_vec = attn_vec.view(n_head, inp.size(0), inp.size(1), d_head)
        attn_vec = attn_vec.permute(1, 2, 0, 3).contiguous().view(
            inp.size(0), inp.size(1), n_head * d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = residual + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(residual + attn_out)

        output = output.to(attn_out.dtype)

        return output


class TransformerLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, kernel_size, dropout, dropatt=0.1, sepconv=False, pre_lnorm=False):
        super(TransformerLayer, self).__init__()

        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, dropatt, pre_lnorm)
        self.pos_ff = PositionwiseConvFF(d_model, d_inner, kernel_size, dropout, sepconv, pre_lnorm)

    def forward(self, dec_inp, mask=None):
        output = self.dec_attn(dec_inp, attn_mask=~mask.squeeze(2))
        output *= mask
        output = self.pos_ff(output)
        output *= mask
        return output


class FFTransformer(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_head, d_inner, kernel_size,
                 dropout, dropatt, dropemb=0.0, embed_input=True,
                 n_embed=None, d_embed=None, padding_idx=0, input_type=None,
                 sepconv=False, pre_lnorm=False, lang_emb_dim=0, g_emb_dim=0, emo_emb_dim=0):
        super(FFTransformer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.padding_idx = padding_idx
        self.lang_emb_dim = lang_emb_dim
        self.hidden_channels = d_embed
        self.g_emb_dim = g_emb_dim
        self.emo_emb_dim = emo_emb_dim

        self.input_type = input_type
        if embed_input:
            if input_type == 'pf':
                self.word_emb = nn.Linear(n_embed, d_embed or d_model,
                                          bias=False)
            else:
                self.word_emb = nn.Embedding(n_embed, d_embed or d_model,
                                             padding_idx=self.padding_idx)
        else:
            self.word_emb = None

        self.pos_emb = PositionalEmbedding(self.d_model)
        self.drop = nn.Dropout(dropemb)
        self.layers = nn.ModuleList()

        for _ in range(n_layer):
            self.layers.append(
                TransformerLayer(
                    n_head, d_model, d_head, d_inner, kernel_size, dropout,
                    dropatt=dropatt, sepconv=sepconv, pre_lnorm=pre_lnorm)
            )

        if emo_emb_dim != 0:
            cond_layer = torch.nn.Conv1d(emo_emb_dim, 2 * self.hidden_channels * n_layer, 1)
            self.cond_pre = torch.nn.Conv1d(self.hidden_channels, 2 * self.hidden_channels, 1)
            self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name="weight")

        if g_emb_dim != 0:
            cond_layer_g = torch.nn.Conv1d(g_emb_dim, 2 * self.hidden_channels * n_layer, 1)
            self.cond_pre_g = torch.nn.Conv1d(self.hidden_channels, 2 * self.hidden_channels, 1)
            self.cond_layer_g = torch.nn.utils.weight_norm(cond_layer_g, name="weight")

    def forward(self, dec_inp, seq_lens=None, g=None, emo=None, pre_cond=None, post_cond=None):
        if emo is not None:
            emo = self.cond_layer(emo.transpose(1, 2))

        if g is not None and self.word_emb is None:
            g = self.cond_layer_g(g.transpose(1, 2))

        if self.word_emb is None:
            inp = dec_inp
            mask = mask_from_lens(seq_lens).unsqueeze(2)
        else:
            inp = self.word_emb(dec_inp)
            # [bsz x L x 1]
            if self.input_type == 'pf':
                mask = (torch.count_nonzero(dec_inp, dim=2) > 0).unsqueeze(2)
            else:
                mask = (dec_inp != self.padding_idx).unsqueeze(2)

        pos_seq = torch.arange(inp.size(1), device=inp.device).to(inp.dtype)
        pos_emb = self.pos_emb(pos_seq) * mask

        if pre_cond is not None:
            inp = inp + pre_cond
        
        out = self.drop(inp + pos_emb)
        # if pre_cond is not None:
        #     out = torch.cat((out, pre_cond.expand(out.size(0), out.size(1), -1)), dim=-1)

        layer_count = 0
        for layer in self.layers:
            if self.emo_emb_dim != 0:
                out = self.cond_pre(out.transpose(1, 2))
                cond_offset = layer_count * 2 * self.hidden_channels
                emo_l = emo[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
                out = fused_add_tanh_sigmoid_multiply(out, emo_l, torch.IntTensor([self.hidden_channels]))
                out = out.transpose(1, 2)

            if self.g_emb_dim != 0:
                out = self.cond_pre_g(out.transpose(1, 2))
                cond_offset = layer_count * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
                out = fused_add_tanh_sigmoid_multiply(out, g_l, torch.IntTensor([self.hidden_channels]))
                out = out.transpose(1, 2)

            out = layer(out, mask=mask)
            layer_count = layer_count + 1

        if g is not None and self.word_emb:
            out = torch.cat((out, g.expand(out.size(0), out.size(1), -1)), dim=-1)

        # out = self.drop(out)
        return out, mask


def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts