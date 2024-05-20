# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import torch.nn.functional as F
import torch.nn as nn
from typing import Optional

import torch
from librosa.filters import mel as librosa_mel_fn
from audio_processing import dynamic_range_compression, dynamic_range_decompression
from stft import STFT


def mask_from_lens(lens, max_len: Optional[int] = None):
    if max_len is None:
        max_len = lens.max()
    ids = torch.arange(0, max_len, device=lens.device, dtype=lens.dtype)
    mask = torch.lt(ids, lens.unsqueeze(1))
    return mask


class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sr=sampling_rate, n_fft=filter_length, n_mels=n_mel_channels,
            fmin=mel_fmin, fmax=mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert (torch.min(y.data) >= -1)
        assert (torch.max(y.data) <= 1)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        energy = torch.norm(magnitudes, dim=1)

        return mel_output, energy


# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class AttentionCTCLoss(torch.nn.Module):
    def __init__(self, blank_logprob=-1, batched=False):
        super(AttentionCTCLoss, self).__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.blank_logprob = blank_logprob
        self.CTCLoss = nn.CTCLoss(zero_infinity=True)

        if batched:
            self.forward = self.forward_batched

    def forward(self, attn_logprob, text_lens, mel_lens):
        """Calculate CTC alignment loss between embedded texts and mel features

        Args:
          attn_logprob: batch x 1 x max(mel_lens) x max(text_lens)
            Batched tensor of attention log probabilities, padded to length of
            longest sequence in each dimension.
          text_lens: batch-D vector of lengths of each text sequence
          mel_lens: batch-D vector of lengths of each mel sequence

        Returns:
          cost: Average CTC loss over batch
        """
        # Add blank token to attention matrix, with small emission probability
        # at all timesteps
        attn_logprob = F.pad(
            attn_logprob, pad=(1, 0, 0, 0, 0, 0), value=self.blank_logprob)

        cost = 0.0
        for bid in range(attn_logprob.shape[0]):
            # Construct target sequence: each text token is mapped to its
            # sequence index, enforcing monotonicity constraint
            target_seq = torch.arange(1, text_lens[bid] + 1).unsqueeze(0)

            curr_logprob = attn_logprob[bid].permute(1, 0, 2)
            curr_logprob = curr_logprob[:mel_lens[bid], :, :text_lens[bid] + 1]
            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            cost += self.CTCLoss(curr_logprob, target_seq,
                                 input_lengths=mel_lens[bid], target_lengths=text_lens[bid])

        cost = cost / attn_logprob.shape[0]
        return cost

    def forward_batched(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        max_key_len = attn_logprob.size(-1)

        # Reorder input to [query_len, batch_size, key_len]
        attn_logprob = attn_logprob.squeeze(1)
        attn_logprob = attn_logprob.permute(1, 0, 2)

        # Add blank label
        attn_logprob = F.pad(
            input=attn_logprob,
            pad=(1, 0, 0, 0, 0, 0),
            value=self.blank_logprob)

        # Convert to log probabilities
        # Note: Mask out probs beyond key_len
        key_inds = torch.arange(
            max_key_len+1,
            device=attn_logprob.device,
            dtype=torch.long)
        attn_logprob.masked_fill_(
            key_inds.view(1, 1, -1) > key_lens.view(1, -
                                                    1, 1),  # key_inds >= key_lens+1
            -float("inf"))
        attn_logprob = self.log_softmax(attn_logprob)

        # Target sequences
        target_seqs = key_inds[1:].unsqueeze(0)
        target_seqs = target_seqs.repeat(key_lens.numel(), 1)

        # Evaluate CTC loss
        cost = self.CTCLoss(
            attn_logprob, target_seqs,
            input_lengths=query_lens, target_lengths=key_lens)
        return cost


class AttentionBinarizationLoss(torch.nn.Module):
    def __init__(self):
        super(AttentionBinarizationLoss, self).__init__()

    def forward(self, hard_attention, soft_attention, eps=1e-12):
        log_sum = torch.log(torch.clamp(soft_attention[hard_attention == 1],
                            min=eps)).sum()
        return -log_sum / hard_attention.sum()
