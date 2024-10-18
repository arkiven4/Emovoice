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

import csv
import os
import argparse
import glob
import json
from typing import Optional

import librosa
import numpy as np
import torch
from scipy.io.wavfile import read

def load_wav_to_torch(full_path, force_sampling_rate=None):
    if force_sampling_rate is not None:
        data, sampling_rate = librosa.load(full_path, sr=force_sampling_rate)
    else:
        sampling_rate, data = read(full_path)

    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
  with open(filename, encoding='utf-8') as f:
    filepaths_and_text = [line.strip().split(split) for line in f] #if int(line.strip().split(split)[1]) == 0 or int(line.strip().split(split)[1]) == 2
  return filepaths_and_text

# def load_filepaths_and_text(dataset_path, fnames, delim="|"):
#     data_fields = ['audio', 'pitch', 'duration', 'speaker', 'language']
#     fields = data_fields + ['text']  # TODO: add fname here
#     fpaths_and_text = []
#     for fname in fnames:
#         with open(fname, encoding='utf-8') as f:
#             reader = csv.DictReader(f, delimiter=delim, restval=None)
#             # return default None for any unspecified fields
#             reader.fieldnames.extend(
#                 i for i in fields if i not in reader.fieldnames)
#             for line in reader:
#                 for k, v in line.items():
#                     if k in data_fields and v is not None:
#                         data_path = os.path.join(dataset_path, v)
#                         if os.path.exists(data_path):
#                             line[k] = data_path
#                         else:
#                             line[k] = v  # pass through speaker/lang ids
#                 fpaths_and_text.append(line)
#     return fpaths_and_text


def load_speaker_lang_ids(fname):
    if fname is None:
        return
    lab2id = {}
    with open(fname) as inf:
        for line in inf:
            lab, id_ = line.strip().split()
            lab2id[lab] = int(id_)
    return lab2id


def to_gpu(x):
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


def to_device_async(tensor, device):
    return tensor.to(device, non_blocking=True)


def to_numpy(x):
    return x.cpu().numpy() if isinstance(x, torch.Tensor) else x


def get_hparams(init=True):
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', type=str, default="./configs/base.json",
                      help='JSON file for configuration')
  parser.add_argument('-m', '--model', type=str, required=True,
                      help='Model name')
  
  args = parser.parse_args()
  model_dir = os.path.join("./logs", args.model)

  if not os.path.exists(model_dir):
    os.makedirs(model_dir)

  config_path = args.config
  config_save_path = os.path.join(model_dir, "config.json")
  if init:
    with open(config_path, "r") as f:
      data = f.read()
    with open(config_save_path, "w") as f:
      f.write(data)
  else:
    with open(config_save_path, "r") as f:
      data = f.read()
  config = json.loads(data)
  
  hparams = HParams(**config)
  hparams.model_dir = model_dir
  return hparams

def get_hparams_from_dir(model_dir):
  config_save_path = os.path.join(model_dir, "config.json")
  with open(config_save_path, "r") as f:
    data = f.read()
  config = json.loads(data)

  hparams =HParams(**config)
  hparams.model_dir = model_dir
  return hparams

def latest_checkpoint_path(dir_path, regex="G_*.pth"):
  f_list = glob.glob(os.path.join(dir_path, regex))
  f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
  x = f_list[-1]
  print(x)
  return x

def load_checkpoint(model, ema_model, optimizer, scaler, epoch,
                    total_iter, fp16_run, filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    epoch[0] = checkpoint['epoch'] + 1
    total_iter[0] = checkpoint['iteration']

    sd = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    getattr(model, 'module', model).load_state_dict(sd)
    optimizer.load_state_dict(checkpoint['optimizer'])

    if fp16_run:
        scaler.load_state_dict(checkpoint['scaler'])

    if ema_model is not None:
        ema_model.load_state_dict(checkpoint['ema_state_dict'])

def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    #print(checkpoint_dict.keys())
    model_dict = checkpoint_dict['state_dict']

    random_weight_layer = []
    mismatched_layers = []
    unfound_layers = []
    
    for key, value in model_dict.items(): # model_dict warmstart weight
        if hasattr(model, 'module'): # model is current model
            if key in model.module.state_dict() and value.size() != model.module.state_dict()[key].size():
                try:
                    model_dict[key] = transfer_weight(model_dict[key], model.module.state_dict()[key].size())
                    if model_dict[key].size() != model.module.state_dict()[key].size():
                      mismatched_layers.append(key)
                    else:
                      random_weight_layer.append(key)
                except:
                    mismatched_layers.append(key)
        else:
            if key in model.state_dict() and value.size() != model.state_dict()[key].size():
                try:
                    model_dict[key] = transfer_weight(model_dict[key], model.state_dict()[key].size())
                    if model_dict[key].size() != model.state_dict()[key].size():
                      mismatched_layers.append(key)
                    else:
                      random_weight_layer.append(key)
                except:
                    mismatched_layers.append(key)
    
    # for key, value in model_dict.items():
    #   if hasattr(model, 'module'):
    #     if key not in model.module.state_dict():
    #       unfound_layers.append(key)
    #   else:
    #     if key not in model.state_dict():
    #       unfound_layers.append(key)
        
    print("Mismatched")
    print(mismatched_layers)

    print("random_weight_layer")
    print(random_weight_layer)
    
    ignore_layers = ignore_layers + mismatched_layers + random_weight_layer
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        if hasattr(model, 'module'):
          dummy_dict = model.module.state_dict()
          dummy_dict.update(model_dict)
        else:
          dummy_dict = model.state_dict()
          dummy_dict.update(model_dict)
        model_dict = dummy_dict

    if hasattr(model, 'module'):
      model.module.load_state_dict(model_dict, strict=False)
    else:
      model.load_state_dict(model_dict, strict=False)
    
    #del checkpoint_dict, model_dict, dummy_dict
    #gc.collect()
    #torch.cuda.empty_cache()
    return model

class HParams():
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = HParams(**v)
      self[k] = v
    
  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()

  def __len__(self):
    return len(self.__dict__)

  def __getitem__(self, key):
    return getattr(self, key)

  def __setitem__(self, key, value):
    return setattr(self, key, value)

  def __contains__(self, key):
    return key in self.__dict__

  def __repr__(self):
    return self.__dict__.__repr__()
  
def transfer_weight(original_tensor, target_size):
    differences = [target_size[i] - original_tensor.size(i) for i in range(len(target_size))]
    for i, diff in enumerate(differences):
        if diff > 0:
            new_dims = list(original_tensor.size())
            new_dims[i] = diff
            rand_weight = torch.randn(*new_dims)
            original_tensor = torch.cat([original_tensor, rand_weight], dim=i)
        # elif diff < 0:
        #     slices = []
        #     for j in range(len(target_size)):
        #         if j == i:
        #             slices.append(slice(0, original_tensor.size(j) + diff))
        #         else:
        #             slices.append(slice(0, original_tensor.size(j)))
        #     slices[i] = slice(0, target_size[i])
        #     original_tensor = original_tensor[slices]

    return original_tensor