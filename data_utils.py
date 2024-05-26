# *****************************************************************************
#  Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import functools
import json
import os
import random
from itertools import groupby

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.stats import betabinom

from commons import TacotronSTFT
from utils import load_filepaths_and_text, load_speaker_lang_ids, load_wav_to_torch, to_gpu
from text.text_processing import TextProcessor, PhoneProcessor, UnitProcessor

class BetaBinomialInterpolator:
    """Interpolates alignment prior matrices to save computation.

    Calculating beta-binomial priors is costly. Instead cache popular sizes
    and use img interpolation to get priors faster.
    """
    def __init__(self, round_mel_len_to=100, round_text_len_to=20):
        self.round_mel_len_to = round_mel_len_to
        self.round_text_len_to = round_text_len_to
        self.bank = functools.lru_cache(beta_binomial_prior_distribution)

    def round(self, val, to):
        return max(1, int(np.round((val + 1) / to))) * to

    def __call__(self, w, h):
        bw = self.round(w, to=self.round_mel_len_to)
        bh = self.round(h, to=self.round_text_len_to)
        ret = ndimage.zoom(self.bank(bw, bh).T, zoom=(w / bw, h / bh), order=1)
        assert ret.shape[0] == w, ret.shape
        assert ret.shape[1] == h, ret.shape
        return ret


def beta_binomial_prior_distribution(phoneme_count, mel_count, scaling=1.0):
    P = phoneme_count
    M = mel_count
    x = np.arange(0, P)
    mel_text_probs = []
    for i in range(1, M+1):
        a, b = scaling * i, scaling * (M + 1 - i)
        rv = betabinom(P, a, b)
        mel_i_prob = rv.pmf(x)
        mel_text_probs.append(mel_i_prob)
    return np.array(mel_text_probs)


def extract_duration_prior(text_len, mel_len):
    binomial_interpolator = BetaBinomialInterpolator()
    attn_prior = binomial_interpolator(mel_len, text_len)
    #attn_prior = beta_binomial_prior_distribution(text_len, mel_len)
    assert mel_len == attn_prior.shape[0]
    return attn_prior


def estimate_pitch(wav, mel_len, fmin=40, fmax=600, sr=None, hop_length=256, method='yin', start=None, end=None):
    try:
        trimmed_dur = end - start
    except TypeError:
        # either start or end is None => don't need to calculate final duration
        trimmed_dur = end
    snd, sr = librosa.load(wav, sr=sr, offset=start, duration=trimmed_dur)

    if method == 'yin':
        pitch = librosa.yin(snd, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)
    elif method == 'pyin':
        pitch, voiced_flags, voiced_probs  = librosa.pyin(
            snd, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length, fill_na=0.0)

    # TODO: handle truncated hubert feats loaded from disk
    #if pitch.shape[0] - mel_len <= 2:
    #    pitch = pitch[:mel_len]
    assert np.abs(mel_len - pitch.shape[0]) <= 1.0, f'{mel_len}, {pitch.shape[0]} ({pitch.shape})'
    return pitch


def normalize_pitch(pitch, mean, std):
    zero_idxs = np.where(pitch == 0.0)[0]
    pitch -= mean
    pitch /= std
    pitch[zero_idxs] = 0.0
    return pitch


def average_pitch_per_symbol(pitch, durs):
    durs_cum = np.cumsum(np.pad(durs, (1, 0)))
    pitch_char = np.zeros((durs.shape[0],), dtype=float)
    for idx, (a, b) in enumerate(zip(durs_cum[:-1], durs_cum[1:])):
        values = pitch[a:b][np.where(pitch[a:b] != 0.0)[0]]
        pitch_char[idx] = np.mean(values) if len(values) > 0 else 0.0
    return pitch_char


class TextMelAliLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams,
                 input_type='char',
                 load_durs_from_disk=True, load_pitch_from_disk=False,
                 peak_norm=False, trim_silence_dur=None,
                 pitch_fmin=40.0, pitch_fmax=600.0, pitch_method='yin',
                 pitch_mean=None, pitch_std=None, pitch_mean_std_file=None,
                 **kwargs):
        
        # if type(audiopaths_and_text) is str:
        #     audiopaths_and_text = [audiopaths_and_text]

        self.n_mel_channels = hparams.n_mel_channels
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.hop_length = hparams.hop_length
        self.filter_length = hparams.filter_length
        self.win_length = hparams.win_length
        self.load_mel_from_disk = hparams.load_mel_from_disk

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.peak_norm = peak_norm
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        
        self.spk_embeds_path = hparams.spk_embeds_path
        self.emo_embeds_path = hparams.emo_embeds_path
        self.f0_embeds_path = hparams.f0_embeds_path
        self.database_name_index = hparams.database_name_index

        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)

        self.input_type = input_type
        self.symbol_set = None
        self.text_cleaners = hparams.text_cleaners
        self.tp = TextProcessor(self.symbol_set, self.text_cleaners)
        self.n_symbols = len(self.tp.symbols)
        self.padding_idx = self.tp.padding_idx

        if not hparams.load_mel_from_disk:
            self.stft = TacotronSTFT(
                hparams.filter_length, hparams.hop_length, hparams.win_length,
                hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin, hparams.mel_fmax)

        self.load_durs_from_disk = load_durs_from_disk
        self.durations_from = 'attn_prior'
        self.trim_silence_dur = trim_silence_dur
        if trim_silence_dur is not None:
            assert self.durations_from == 'textgrid', \
                "Can only trim silences based on TextGrid alignments"

        self.load_pitch_from_disk = load_pitch_from_disk
        self.pitch_fmin = pitch_fmin
        self.pitch_fmax = pitch_fmax
        self.pitch_method = pitch_method
        if pitch_mean_std_file is not None:
            with open(pitch_mean_std_file) as f:
                stats = json.load(f)
            self.pitch_mean = stats['mean']
            self.pitch_std = stats['std']
        else:
            self.pitch_mean = pitch_mean
            self.pitch_std = pitch_std
        self.pitch_char = (not self.load_pitch_from_disk) and (self.durations_from != 'attn_prior')

    def __len__(self):
        return len(self.audiopaths_and_text)

    def __getitem__(self, index):
        audiopath, lid, text = self.audiopaths_and_text[index][0], self.audiopaths_and_text[index][1], self.audiopaths_and_text[index][2]
        filename = audiopath.split("/")[-1].split(".")[0]
        database_name = audiopath.split("/")[self.database_name_index]

        mel, energy = self.get_mel(audiopath)
        energy = F.pad(input=torch.diff(energy), pad=(0, 1), mode='constant', value=0)

        text = self.get_text(text, lid)
        dur, text, start_time, end_time = self.get_duration(text, mel.size(-1))
        pitch = self.get_pitch(audiopath, dur, start_time, end_time, per_sym=self.pitch_char)
        
        if self.trim_silence_dur is not None:
            text, mel, dur, pitch = self.trim_silence(
                text, mel, dur, pitch, self.trim_silence_dur,
                self.sampling_rate, self.hop_length)

        speaker = torch.Tensor(np.load(f"{self.spk_embeds_path.replace('dataset_name', database_name)}/{filename}.npy"))
        lang = self.get_lid(lid)

        return text, mel, len(text), torch.from_numpy(dur), pitch, energy, speaker, lang, audiopath

    def get_mel(self, filename):
        if self.load_mel_from_disk:
            melspec = torch.load(filename)
        else:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    filename, sampling_rate, self.stft.sampling_rate))
            if self.peak_norm:
                audio = (audio / torch.max(torch.abs(audio))) * (self.max_wav_value - 1)
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec, energy = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
            energy = torch.squeeze(energy, 0)
        return melspec, energy

    def get_text(self, text, lang):
        text_encoded = torch.IntTensor(self.tp.encode_text(text, lang))
        return text_encoded

    def get_duration(self, text=None, mel_len=None, utt_id=None):
        start_time, end_time = None, None
        durations = extract_duration_prior(len(text), mel_len)
        #text = self.tp.ids_to_text(text.numpy())

        return durations, text, start_time, end_time

    def get_pitch(self, audiopath, dur=None, start_time=None, end_time=None, per_sym=False):
        if self.durations_from == 'attn_prior':
            mel_len, text_len = dur.shape
        else:
            mel_len, text_len = dur.sum(), dur.size

        filename = audiopath.split("/")[-1].split(".")[0]
        database_name = audiopath.split("/")[self.database_name_index]
        pitch = torch.from_numpy(np.load(f"{self.f0_embeds_path.replace('dataset_name', database_name)}/{filename}.npy"))
        pitch = pitch[:mel_len]
        pitch[pitch == 0] = torch.nan
        pitch = F.pad(input=torch.diff(pitch), pad=(0, 1), mode='constant', value=0)
        pitch = torch.nan_to_num(pitch, nan=0.0)
        if pitch.shape[0] < mel_len:
            pitch = F.pad(input=pitch, pad=(0, mel_len - pitch.shape[0]), mode='constant', value=0)

        if self.pitch_mean is not None:
            assert self.pitch_std is not None
            pitch = normalize_pitch(pitch, self.pitch_mean, self.pitch_std)

        if per_sym:
            pitch = average_pitch_per_symbol(pitch, dur)

        if not self.load_pitch_from_disk:
            if self.durations_from == 'attn_prior':
                assert pitch.shape[0] == mel_len
            else:
                assert pitch.shape[0] == text_len
        return pitch

    def get_speaker(self, speaker):
        if self.speaker_ids is not None:
            # closed set of speaker ids
            speaker = self.speaker_ids[speaker]
        elif speaker is not None:
            # load speaker embeddings from disk
            speaker = torch.load(speaker)
        # speaker is None if not specified in meta file
        return speaker

    def get_lang(self, lang):
        if self.lang_ids is not None:
            # closed set of lang ids
            lang = self.lang_ids[lang]
        elif lang is not None:
            # load lang embeddings from disk
            lang = torch.load(lang)
        # lang is None if not specified in meta file
        return lang

    def get_lid(self, lid):
        lid = torch.IntTensor([int(lid)])
        return lid

    def trim_silence(self, text, mel, durations, pitch, keep_sil_frames,
                    sampling_rate, hop_length):
        if keep_sil_frames > 0:
            keep_sil_frames = np.round(keep_sil_frames * sampling_rate / hop_length)
        keep_sil_frames = int(keep_sil_frames)

        if type(text) is str:
            text = text.split()
        text = np.array(text)
        sil_idx = np.where(text == 'sil')[0]

        sil_durs = durations[sil_idx]
        trim_durs = np.array(
            [d - keep_sil_frames if d > keep_sil_frames else 0 for d in sil_durs],
            dtype=np.int64)
        if trim_durs.size == 2:
            # trim both sides
            trim_start, trim_end = trim_durs
            trim_end = -trim_end
        elif trim_durs.size == 0:
            # nothing to trim
            trim_start, trim_end = None, None
        elif sil_idx[0] == 0:
            # trim only leading silence
            trim_start, trim_end = trim_durs[0], None
        elif sil_idx[0] == len(text) - 1:
            # trim only trailing silence
            trim_start, trim_end = None, -trim_durs[0]
        if trim_end == 0:
            # don't trim trailing silence if already short enough
            trim_end = None

        if keep_sil_frames == 0:
            sil_mask = text != "sil"
        else:
            sil_mask = np.ones_like(text, dtype=bool)
        mel = mel[:, trim_start:trim_end]
        durations.put(sil_idx, sil_durs - trim_durs)
        durations = durations[sil_mask]
        assert mel.shape[1] == durations.sum()#, \
            #"{}: Trimming led to mismatched durations ({}) and mels ({})".format(
            #    fname, sum(durations), mel.shape[1])
        pitch = pitch[sil_mask]
        assert len(pitch) == len(durations)#, \
            #"{}: Trimming led to mismatched durations ({}) and pitches ({})".format(
            #    fname, len(durations), len(pitch))
        text = text[sil_mask]
        assert len(text) == len(durations)
        text = ' '.join(text)
        return text, mel, durations, pitch


class TextMelAliCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, symbol_type='char', n_symbols=148, mas=False):
        self.symbol_type = symbol_type
        self.n_symbols = n_symbols
        self.mas = mas

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True)
        max_input_len = input_lengths[0]
        audiopath = [batch[i][-1] for i in ids_sorted_decreasing]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])

        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            output_lengths[i] = mel.size(1)

        dur_padded = torch.zeros(len(batch), max_target_len, max_input_len)
        dur_padded.zero_()
        dur_lens = None
        for i in range(len(ids_sorted_decreasing)):
            dur = batch[ids_sorted_decreasing[i]][3]
            dur_padded[i, :dur.size(0), :dur.size(1)] = dur

        pitch_padded = torch.zeros(mel_padded.size(0), mel_padded.size(2), dtype=batch[0][4].dtype)
        for i in range(len(ids_sorted_decreasing)):
            pitch = batch[ids_sorted_decreasing[i]][4]
            pitch_padded[i, :pitch.shape[0]] = pitch

        energy_padded = torch.zeros(mel_padded.size(0), mel_padded.size(2), dtype=batch[0][5].dtype)
        for i in range(len(ids_sorted_decreasing)):
            energy = batch[ids_sorted_decreasing[i]][5]
            energy_padded[i, :energy.shape[0]] = energy

        speaker = torch.zeros((len(batch), batch[0][6].shape[0]))
        for i in range(len(ids_sorted_decreasing)):
            speaker[i] = batch[ids_sorted_decreasing[i]][6]

        lang = torch.zeros_like(input_lengths)
        for i in range(len(ids_sorted_decreasing)):
            lang[i] = batch[ids_sorted_decreasing[i]][7]

        # count number of items - characters in text
        len_x = [x[2] for x in batch]
        len_x = torch.Tensor(len_x)

        return (text_padded, input_lengths, mel_padded, output_lengths,
                len_x, dur_padded, dur_lens, pitch_padded, energy_padded, speaker, lang, audiopath)


def batch_to_gpu(batch, symbol_type='char', mas=False):
    text_padded, input_lengths, mel_padded, output_lengths, \
        len_x, dur_padded, dur_lens, pitch_padded, energy_padded, speaker, lang, audiopath = batch

    input_lengths = to_gpu(input_lengths).long()
    mel_padded = to_gpu(mel_padded).float()
    output_lengths = to_gpu(output_lengths).long()
    pitch_padded = to_gpu(pitch_padded).float()
    energy_padded = to_gpu(energy_padded).float()

    text_padded = to_gpu(text_padded)
    text_padded = text_padded.float() if symbol_type == 'pf' else text_padded.long()

    dur_padded = to_gpu(dur_padded)
    dur_padded = dur_padded.float() if mas else dur_padded.long()

    if speaker is not None:
        speaker = to_gpu(speaker)
        speaker = speaker.float() if speaker.dim() > 1 else speaker.long()

    if lang is not None:
        lang = to_gpu(lang)
        lang = lang.float() if lang.dim() > 1 else lang.long()

    # Alignments act as both inputs and targets - pass shallow copies
    x = [text_padded, input_lengths, mel_padded, output_lengths,
        dur_padded, dur_lens, pitch_padded, energy_padded, speaker, lang]
    if mas:
        y = [mel_padded, input_lengths, output_lengths]
    else:
        dur_lens = to_gpu(dur_lens).long()
        y = [mel_padded, dur_padded, dur_lens, pitch_padded, energy_padded]
    len_x = torch.sum(output_lengths)
    return (x, y, len_x)
