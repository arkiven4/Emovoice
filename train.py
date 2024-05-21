import os
import json
import argparse
import math
from collections import defaultdict, OrderedDict
import time
from tqdm import tqdm
import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.distributions as D
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data.sampler import WeightedRandomSampler

import librosa

from torch.optim import Adam
from torch_optimizer import Lamb

import tb_dllogger as logger

from loss_function import FastPitchLoss, FastPitchMASLoss, FastPitchTVCGMMLoss

from data_utils import batch_to_gpu, TextMelAliCollate, TextMelAliLoader
import models
import commons
import utils

import sys
sys.path.append('./BigVGAN_/')

from BigVGAN_.env import AttrDict
from BigVGAN_.meldataset import MAX_WAV_VALUE
from BigVGAN_.models import BigVGAN as Generator

global_step = 0
global_tqdm = None

def log_stdout(logger, subset, epoch_iters, total_steps, loss, mel_loss,
               dur_loss, pitch_loss, energy_loss, align_loss, took, now_lr=None):
    logger_data = [
        ('Loss/Total', loss),
        ('Loss/Mel', mel_loss),
        ('Loss/Duration', dur_loss),
        ('Loss/Pitch', pitch_loss),
        ('Loss/Energy', energy_loss),
        #('Error/Duration', iter_dur_error),
        #('Error/Pitch', iter_pitch_error),
        #('Time/FPS', iter_num_frames / iter_time),
        # only relevant per step, not averaged over epoch
        #('Hyperparameters/Learning rate', optimizer.param_groups[0]['lr']),
    ]
    if align_loss is not None:
        logger_data.extend([
            ('Loss/Alignment', align_loss),
            #('Align/Attention loss', iter_attn_loss),
            #('Align/KL loss', iter_kl_loss),
            #('Align/KL weight', iter_kl_weight),  # step, not avg
        ])
    logger_data.append(('Time/Iter time', took))
    if now_lr != None:
        logger_data.append(('Parameter/Learning Rate', now_lr))
    logger.log(epoch_iters,
               tb_total_steps=total_steps,
               subset=subset,
               data=OrderedDict(logger_data)
    )


def plot_spectrograms(y, audiopaths, sorted_idx, step, n=4, label='Predicted spectrogram', mas=False):
    """Plot spectrograms for n utterances in batch"""
    fnames = []
    for idx_loop, audiopath in enumerate(audiopaths):
        fnames.append(os.path.splitext(os.path.basename(audiopath))[0])
    
    bs = len(fnames)
    n = min(n, bs)
    s = bs // n
    idx = sorted_idx[::s]
    fnames = [fnames[i] for i in idx]
    if label == 'Predicted spectrogram':
        # y: mel_out, dec_mask, dur_pred, log_dur_pred, pitch_pred
        mel_specs = y[0][idx].transpose(1, 2).cpu().numpy()
        mel_lens = y[1][idx].squeeze().cpu().numpy().sum(axis=1) - 1
    elif label == 'Reference spectrogram':
        # y: mel_padded, dur_padded, dur_lens, pitch_padded
        mel_specs = y[0][idx].cpu().numpy()
        if mas:
            mel_lens = y[2][idx].cpu().numpy()  # output_lengths
        else:
            mel_lens = y[1][idx].cpu().numpy().sum(axis=1) - 1
    for mel_spec, mel_len, fname in zip(mel_specs, mel_lens, fnames):
        mel_spec = mel_spec[:, :mel_len]
        utt_id = os.path.splitext(os.path.basename(fname))[0]
        logger.log_spectrogram_tb(
            step, '{}/{}'.format(label, utt_id), mel_spec, tb_subset='val')


def generate_audio(y, audiopaths, sorted_idx, step, vocoder=None, sampling_rate=22050, hop_length=256,
                   n=4, label='Predicted audio', mas=False, dataset_path=''):
    """Generate audio from spectrograms for n utterances in batch"""
    fnames = []
    for idx_loop, audiopath in enumerate(audiopaths):
        fnames.append(os.path.splitext(os.path.basename(audiopath))[0])

    bs = len(fnames)
    n = min(n, bs)
    s = bs // n
    idx = sorted_idx[::s]
    fnames = [fnames[i] for i in idx]
    audiopaths = [audiopaths[i] for i in idx]
    with torch.no_grad():
        if label == 'Predicted audio':
            # y: mel_out, dec_mask, dur_pred, log_dur_pred, pitch_pred
            audios = vocoder(torch.FloatTensor(y[0][idx].transpose(1, 2).cpu().detach().numpy()).to('cpu')).cpu().squeeze().numpy()
            #audios = vocoder(y[0][idx].transpose(1, 2)).cpu().squeeze().numpy()
            mel_lens = y[1][idx].squeeze().cpu().numpy().sum(axis=1) - 1
        else:
            # y: mel_padded, dur_padded, dur_lens, pitch_padded
            if label == 'Copy synthesis':
                audios = vocoder(torch.FloatTensor(y[0][idx].cpu().detach().numpy()).to('cpu')).cpu().squeeze().numpy()
                #audios = vocoder(y[0][idx]).cpu().squeeze().numpy()
            elif label == 'Reference audio':
                audios = []
                for audiopath in audiopaths:
                    audio, _ = librosa.load(audiopath, sr=sampling_rate)
                    audios.append(audio)
            if mas:
                mel_lens = y[2][idx].cpu().numpy()  # output_lengths
            else:
                mel_lens = y[1][idx].cpu().numpy().sum(axis=1) - 1


    index = 0
    for audio, mel_len, fname in zip(audios, mel_lens, fnames):
        index = index + 1
        audio = audio[:mel_len * hop_length]
        audio = audio / np.max(np.abs(audio))
        logger.log_audio_tb(
            step, '{}/{}'.format(label, "Audio_" + str(index)), audio, sampling_rate, tb_subset='val')


def plot_attn_maps(y, audiopaths, sorted_idx, step, n=4, label='Predicted alignment'):
    fnames = []
    for idx_loop, audiopath in enumerate(audiopaths):
        fnames.append(os.path.splitext(os.path.basename(audiopath))[0])

    bs = len(fnames)
    n = min(n, bs)
    s = bs // n
    idx = sorted_idx[::s]
    fnames = [fnames[i] for i in idx]
    _, dec_mask, *_, attn_softs, attn_hards, attn_hard_durs, _ = y
    attn_softs = attn_softs[idx].cpu().numpy()
    attn_hards = attn_hards[idx].cpu().numpy()
    attn_hard_durs = attn_hard_durs[idx].cpu().numpy()
    text_lens = np.count_nonzero(attn_hard_durs, 1)
    mel_lens = dec_mask[idx].cpu().numpy().squeeze(2).sum(1)
    for attn_soft, attn_hard, mel_len, text_len, fname in zip(
            attn_softs, attn_hards, mel_lens, text_lens, fnames):
        attn_soft = attn_soft[:,:mel_len,:text_len].squeeze(0).transpose()
        attn_hard = attn_hard[:,:mel_len,:text_len].squeeze(0).transpose()
        utt_id = os.path.splitext(os.path.basename(fname))[0]
        logger.log_attn_maps_tb(
            step, '{}/{}'.format(label, utt_id), attn_soft, attn_hard, tb_subset='val')

def reduce_tensor(tensor, num_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt.true_divide(num_gpus)

def adjust_learning_rate(total_iter, opt, learning_rate, warmup_iters=None):
    if warmup_iters == 0:
        scale = 1.0
    elif total_iter > warmup_iters:
        scale = 1. / (total_iter ** 0.5)
    else:
        scale = total_iter / (warmup_iters ** 1.5)

    for param_group in opt.param_groups:
        param_group['lr'] = learning_rate * scale

def maybe_save_checkpoint(rank, model, ema_model, optimizer, scaler, epoch,
                        total_iter, config, hparams):
    
    if rank != 0:
        return

    intermediate = (hparams.train.epochs_per_checkpoint > 0 and epoch % hparams.train.epochs_per_checkpoint == 0)

    if not intermediate and epoch < hparams.train.epochs:
        return

    fpath = os.path.join(hparams.model_dir, f"fastpitch_{epoch}.pt")
    print(f"Saving model and optimizer state at epoch {epoch} to {fpath}")
    ema_dict = None if ema_model is None else ema_model.state_dict()
    checkpoint = {'epoch': epoch,
                  'iteration': total_iter,
                  'config': config,
                  'state_dict': model.state_dict(),
                  'ema_state_dict': ema_dict,
                  'optimizer': optimizer.state_dict()}
    if hparams.train.fp16_run:
        checkpoint['scaler'] = scaler.state_dict()
    torch.save(checkpoint, fpath)

def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "65530"
    
    hps = utils.get_hparams()
    torch.manual_seed(hps.train.seed)
    np.random.seed(hps.train.seed)

    hps.train.num_gpus = torch.cuda.device_count()
    hps.train.batch_size = int(hps.train.batch_size / hps.train.num_gpus)

    if hps.train.distributed_run:
        mp.spawn(train_and_eval, nprocs=hps.train.num_gpus, args=(
            hps.train.num_gpus,
            hps,
        ))
    else:
        train_and_eval(0, hps.train.num_gpus, hps)

def validate(model, epoch, total_iter, criterion, valset, batch_size, collate_fn,
             distributed_run, batch_to_gpu, use_gt_durations=False, ema=False,
             mas=False, attention_kl_loss=None, kl_weight=None,
             vocoder=None, sampling_rate=22050, hop_length=256, n_mel=80,
             tvcgmm_k=0, audio_interval=5):
    """Handles all the validation scoring and printing"""
    was_training = model.training
    model.eval()

    tik = time.perf_counter()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, num_workers=4, shuffle=False,
                                sampler=val_sampler,
                                batch_size=batch_size, pin_memory=False,
                                collate_fn=collate_fn)
        val_meta = defaultdict(float)
        val_num_frames = 0
        for i, batch in enumerate(val_loader):
            x, y, num_frames = batch_to_gpu(batch, collate_fn.symbol_type, mas=mas)
            y_pred = model(x, use_gt_durations=use_gt_durations, use_gt_pitch=False, use_gt_energy=False)
            loss, meta = criterion(y_pred, y, is_training=False, meta_agg='sum')

            if mas:
                _, _, _, _, _, _, _, _, attn_soft, attn_hard, _, _ = y_pred
                binarization_loss = attention_kl_loss(attn_hard, attn_soft)
                kl_loss = binarization_loss * kl_weight
                loss += kl_loss
                meta['kl_loss'] = kl_loss.clone().detach()
                meta['kl_weight'] = kl_weight.clone().detach()
                meta['align_loss'] = meta['attn_loss'] + meta['kl_loss']

            if distributed_run:
                for k, v in meta.items():
                    val_meta[k] += reduce_tensor(v, 1)
                val_num_frames += reduce_tensor(num_frames.data, 1).item()
            else:
                for k, v in meta.items():
                    val_meta[k] += v
                val_num_frames = num_frames.item()

            # log spectrograms and generated audio for first few utterances
            if (i == 0) and (epoch % audio_interval == 0 if epoch is not None else True):
                audiopaths = batch[-1]
                # reorder utterances by mel length
                if mas:
                    tgt_mel_lens = y[2]
                else:
                    tgt_mel_lens = y[1].sum(axis=1)
                tgt_mel_lens_sorted_idx = [
                    i for i, _ in sorted(enumerate(tgt_mel_lens), key=lambda x: x[1], reverse=True)]

                if epoch == audio_interval:
                    # plot ref and copy synthesis only on first epoch
                    plot_spectrograms(
                        y, audiopaths, tgt_mel_lens_sorted_idx, total_iter,
                        n=4, label='Reference spectrogram', mas=mas)
                    if vocoder is not None:
                        generate_audio(y, audiopaths, tgt_mel_lens_sorted_idx, total_iter,
                                       vocoder, sampling_rate, hop_length,
                                       n=2, label='Reference audio', mas=mas)
                        generate_audio(y, audiopaths, tgt_mel_lens_sorted_idx, total_iter,
                                       vocoder, sampling_rate, hop_length,
                                       n=2, label='Copy synthesis', mas=mas)
                plot_spectrograms(
                    y_pred, audiopaths, tgt_mel_lens_sorted_idx, total_iter,
                    n=4, label='Predicted spectrogram', mas=mas)
                if vocoder is not None:
                    generate_audio(y_pred, audiopaths, tgt_mel_lens_sorted_idx, total_iter,
                                   vocoder, sampling_rate, hop_length, n=2,
                                   label='Predicted audio', mas=mas)
                if mas:
                    plot_attn_maps(
                        y_pred, audiopaths, tgt_mel_lens_sorted_idx, total_iter,
                        n=4, label='Predicted alignment')

        val_meta = {k: v / len(valset) for k, v in val_meta.items()}

    val_meta['took'] = time.perf_counter() - tik

    log_stdout(logger,
               'val_ema' if ema else 'val',
               (epoch,) if epoch is not None else (),
               total_iter,
               val_meta['loss'].item(),
               val_meta['mel_loss'].item(),
               val_meta['duration_predictor_loss'].item(),
               val_meta['pitch_loss'].item(),
               val_meta['energy_loss'].item(),
               None if not mas else val_meta['align_loss'].item(),
               val_meta['took']
    )

    if was_training:
        model.train()
    return val_meta

def train_and_eval(rank, n_gpus, hps):
    global global_step

    log_fpath = os.path.join(hps.model_dir, 'nvlog.json')
    tb_subsets = ['train', 'val']
    logger.init(log_fpath, hps.model_dir, enabled=(rank == 0),
                tb_subsets=tb_subsets)
    logger.parameters(vars(hps.train), tb_subset='train')

    # if rank == 0:
        # logger = utils.get_logger(hps.model_dir)
        # logger.info(hps)
        # utils.check_git_hash(hps.model_dir)
        # writer = SummaryWriter(log_dir=hps.model_dir)
        # writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    if hps.train.distributed_run:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=n_gpus, rank=rank
        )
        print("Rank {}: Done initializing distributed training".format(rank))
    torch.manual_seed(hps.train.seed)

    trainset = TextMelAliLoader(audiopaths_and_text=hps.data.training_files, hparams=hps.data)
    valset = TextMelAliLoader(audiopaths_and_text=hps.data.training_files, hparams=hps.data)

    collate_fn = TextMelAliCollate(symbol_type='char', n_symbols=trainset.n_symbols, mas=True)

    if hps.train.distributed_run:
        train_sampler, shuffle = DistributedSampler(trainset), False
    else:
        train_sampler, shuffle = None, True
    
    train_loader = DataLoader(
        trainset, num_workers=4, shuffle=shuffle, sampler=train_sampler,
        batch_size=int(hps.train.batch_size / 1),
        pin_memory=False, drop_last=True, collate_fn=collate_fn)
    
    model = models.FastPitch(n_mel_channels=hps.data.n_mel_channels, n_lang=hps.data.n_lang, n_symbols=trainset.n_symbols, padding_idx=trainset.padding_idx, **hps.model).cuda(rank)

    if hasattr(model, 'forward_mas'):
        print("Using MAS")
        model.forward = model.forward_mas

    vocoder_config_file = os.path.join("BigVGAN_/cp_model", 'config.json')
    with open(vocoder_config_file) as f:
        vocoder_data_c = f.read()

    vocoder_json_config = json.loads(vocoder_data_c)
    vocoder_h = AttrDict(vocoder_json_config)

    vocoder = Generator(vocoder_h).to('cpu')
    state_dict_vocoder = torch.load("BigVGAN_/cp_model/g_05000000.zip", map_location="cpu")
    vocoder.load_state_dict(state_dict_vocoder['generator'])
    vocoder.eval()
    vocoder.remove_weight_norm()
    print("Succcess Load Vocoder")

    kw = dict(lr=hps.train.learning_rate, betas=(0.9, 0.98), eps=1e-9, weight_decay=hps.train.learning_rate)
    if hps.train.optimizer == 'adam':
        optimizer = Adam(model.parameters(), **kw)
    elif hps.train.optimizer == 'lamb':
        optimizer = Lamb(model.parameters(), **kw)

    scaler = GradScaler(enabled=hps.train.fp16_run)
    ema_model = None
    if hps.train.distributed_run:
        model = DistributedDataParallel(
            model, device_ids=[rank], output_device=rank)

    start_epoch = [1]
    start_iter = [0]

    if hps.train.warm_start:
        model = utils.warm_start_model(hps.train.warm_start_checkpoint, model, hps.train.ignored_layer)
    else:
        try:
            utils.load_checkpoint(model, ema_model, optimizer, scaler,
                        start_epoch, start_iter, hps.train.fp16_run, utils.latest_checkpoint_path(hps.model_dir, "fastpitch_*.pt"))
        except Exception as e:
            print("Not Found Checkpoint")
            print(e)

    start_epoch = start_epoch[0]
    total_iter = start_iter[0]

    kl_weight = None
    attention_kl_loss = None  # for validation

    criterion = FastPitchMASLoss(
        dur_predictor_loss_scale=0.1,
        pitch_predictor_loss_scale=0.1,
        energy_predictor_loss_scale=0.1,
        attn_loss_scale=1.0)
    attention_kl_loss = commons.AttentionBinarizationLoss()  # L_bin
    
    model.train()
    torch.cuda.synchronize()

    for epoch in range(start_epoch, hps.train.epochs + 1):
        epoch_start_time = time.perf_counter()

        epoch_loss = 0.0
        epoch_mel_loss = 0.0
        epoch_dur_loss = 0.0
        epoch_pitch_loss = 0.0
        epoch_energy_loss = 0.0
        epoch_align_loss = 0.0
        epoch_attn_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_dur_error = 0.0
        epoch_pitch_error = 0.0
        epoch_energy_error = 0.0
        epoch_num_frames = 0
        epoch_frames_per_sec = 0.0

        if hps.train.distributed_run:
            train_loader.sampler.set_epoch(epoch)

        accumulated_steps = 0
        iter_loss = 0
        iter_num_frames = 0
        iter_meta = {}

        epoch_iter = 0
        num_iters = len(train_loader) // 1
        for batch in train_loader:

            if accumulated_steps == 0:
                if epoch_iter == num_iters:
                    break
                total_iter += 1
                epoch_iter += 1
                iter_start_time = time.perf_counter()

                adjust_learning_rate(total_iter, optimizer, hps.train.learning_rate, hps.train.warmup_steps)

                model.zero_grad()

            x, y, num_frames = batch_to_gpu(batch, 'char', True)

            with autocast(enabled=hps.train.fp16_run):
                y_pred = model(x, use_gt_durations=True)
                loss, meta = criterion(y_pred, y)

                if True:
                    if epoch >= hps.train.kl_loss_start_epoch:
                        _, _, _, _, _, _, _, _, attn_soft, attn_hard, _, _ = y_pred
                        binarization_loss = attention_kl_loss(attn_hard, attn_soft)
                        kl_weight = torch.tensor(
                            min((epoch - hps.train.kl_loss_start_epoch) / hps.train.kl_loss_warmup_epochs,
                                1.0) * hps.train.kl_loss_weight, device=loss.device)
                        kl_loss = binarization_loss * kl_weight
                        loss += kl_loss
                        meta['kl_loss'] = kl_loss.clone().detach()
                        meta['kl_weight'] = kl_weight.clone().detach()
                    else:
                        binarization_loss = 0
                        kl_weight = torch.tensor(0, device=loss.device)
                        meta['kl_weight'] = kl_weight.clone().detach()
                        meta['kl_loss'] = torch.zeros_like(loss).detach()
                    meta['align_loss'] = meta['attn_loss'] + meta['kl_loss']

                loss /= 1

            meta = {k: v / 1
                    for k, v in meta.items()}

            if hps.train.fp16_run:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if hps.train.distributed_run:
                reduced_loss = reduce_tensor(loss.data, hps.train.num_gpus).item()
                reduced_num_frames = reduce_tensor(num_frames.data, 1).item()
                meta = {k: reduce_tensor(v, hps.train.num_gpus) for k, v in meta.items()}
            else:
                reduced_loss = loss.item()
                reduced_num_frames = num_frames.item()
            if np.isnan(reduced_loss):
                raise Exception("loss is NaN")

            accumulated_steps += 1
            iter_loss += reduced_loss
            iter_num_frames += reduced_num_frames
            iter_meta = {k: iter_meta.get(k, 0) + meta.get(k, 0) for k in meta}

            if accumulated_steps % 1 == 0:
                if hps.train.fp16_run:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), hps.train.grad_clip_thresh)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), hps.train.grad_clip_thresh)
                    optimizer.step()
                logger.log_grads_tb(total_iter, model)

                iter_time = time.perf_counter() - iter_start_time
                iter_mel_loss = iter_meta['mel_loss'].item()
                iter_dur_loss = iter_meta['duration_predictor_loss'].item()
                iter_pitch_loss = iter_meta['pitch_loss'].item()
                iter_energy_loss = iter_meta['energy_loss'].item()
                iter_dur_error = iter_meta['duration_error'].item()
                iter_pitch_error = iter_meta['pitch_error'].item()
                iter_energy_error = iter_meta['energy_error'].item()
                epoch_loss += iter_loss
                epoch_mel_loss += iter_mel_loss
                epoch_dur_loss += iter_dur_loss
                epoch_pitch_loss += iter_pitch_loss
                epoch_energy_loss += iter_energy_loss
                epoch_dur_error += iter_dur_error
                epoch_pitch_error += iter_pitch_error
                epoch_energy_error += iter_energy_error
                epoch_num_frames += iter_num_frames
                epoch_frames_per_sec += iter_num_frames / iter_time
                
                if True:
                    iter_align_loss = iter_meta['align_loss'].item()
                    iter_attn_loss = iter_meta['attn_loss'].item()
                    iter_kl_loss = iter_meta['kl_loss'].item()
                    iter_kl_weight = iter_meta['kl_weight']
                    epoch_align_loss += iter_align_loss
                    epoch_attn_loss += iter_attn_loss
                    epoch_kl_loss += iter_kl_loss

                log_stdout(logger,
                           'train',
                           (epoch, epoch_iter, num_iters),
                           total_iter,
                           iter_loss,
                           iter_mel_loss,
                           iter_dur_loss,
                           iter_pitch_loss,
                           iter_energy_loss,
                           None if not True else iter_align_loss,
                           iter_time,
                           optimizer.param_groups[0]["lr"]
                )

                accumulated_steps = 0
                iter_loss = 0
                iter_num_frames = 0
                iter_meta = {}

        # Finished epoch
        epoch_time = time.perf_counter() - epoch_start_time

        log_stdout(logger,
                   'train_avg',
                   (epoch,),
                   None,
                   epoch_loss / epoch_iter,
                   epoch_mel_loss / epoch_iter,
                   epoch_dur_loss / epoch_iter,
                   epoch_pitch_loss / epoch_iter,
                   epoch_energy_loss / epoch_iter,
                   None if not True else epoch_align_loss / epoch_iter,
                   epoch_time,
                   optimizer.param_groups[0]["lr"]
        )

        validate(model, epoch, total_iter, criterion, valset, hps.train.batch_size,
            collate_fn, hps.train.distributed_run, batch_to_gpu, use_gt_durations=True,
            mas=True, attention_kl_loss=attention_kl_loss, kl_weight=kl_weight,
            vocoder=vocoder, sampling_rate=hps.data.sampling_rate, hop_length=hps.data.hop_length,
            n_mel=hps.data.n_mel_channels, tvcgmm_k=0, audio_interval=1)

        print("====> Epoch: {}".format(epoch))
        maybe_save_checkpoint(rank, model, ema_model, optimizer, scaler, epoch, total_iter, hps, hps)
        logger.flush()

    # Finished training
    log_stdout(logger,
               'train_avg',
               (),
               None,
               epoch_loss / epoch_iter,
               epoch_mel_loss / epoch_iter,
               epoch_dur_loss / epoch_iter,
               epoch_pitch_loss / epoch_iter,
               epoch_energy_loss / epoch_iter,
               None if not True else epoch_align_loss / epoch_iter,
               epoch_time
    )

    validate(model, None, total_iter, criterion, valset, hps.train.batch_size,
        collate_fn, hps.train.distributed_run, batch_to_gpu, use_gt_durations=True,
        mas=True, attention_kl_loss=attention_kl_loss, kl_weight=kl_weight,
        vocoder=vocoder, sampling_rate=hps.data.sampling_rate, hop_length=hps.data.hop_length,
        n_mel=hps.data.n_mel_channels, tvcgmm_k=0)

if __name__ == "__main__":
    main()
