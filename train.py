import os
import yaml
import argparse
import logging
import shutil

from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

import ffvid
from ffvid import FMLP, FramesDataset, GaussianFourierFeatureTransform, \
    generate_video, set_random_seed, setup_logger, LFF, ModulatedMLP


parser = argparse.ArgumentParser(description='Train videofitting')
parser.add_argument('-c', '--config', type=str,
                    help='path to config .yaml')
args = parser.parse_args()
config_path = args.config

with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

experiment_name = config['train']['experiment_name']
seed = config['train']['seed']
device = config['train']['device']
save_model = config['train']['save_model']
checkpoint = config['train'].get('checkpoint')
output_folder = config['train']['output_folder']

dset_folder = config['data']['dset_folder']
batch_size = config['data']['batch_size']
resolution = config['data']['resolution']

phase_mod = config['generator']['phase_mod']
phase_mod_type = config['generator']['phase_mod_type']

renderrer = getattr(ffvid, config['generator']['renderrer']['arch'])
internal_dim_rend = config['generator']['renderrer']['internal_dim']
num_layers_rend = config['generator']['renderrer']['num_layers']
gen_conv = getattr(ffvid, config['generator']['renderrer']['conv'])
num_freqs_rend = config['generator']['renderrer']['num_freqs']
ff_func_rend = getattr(ffvid, config['generator']['renderrer']['ff_func'])
act_rend = getattr(nn, config['generator']['renderrer']['act'])()

modulator = getattr(ffvid, config['generator']['modulator']['arch'])
internal_dim_mod = config['generator']['modulator']['internal_dim']
num_layers_mod = config['generator']['modulator']['num_layers']
mod_conv = getattr(ffvid, config['generator']['modulator']['conv'])
num_freqs_mod = config['generator']['modulator']['num_freqs']
ff_func_mod = getattr(ffvid, config['generator']['modulator']['ff_func'])
act_mod = getattr(nn, config['generator']['modulator']['act'])()

curdate = datetime.now()
experiment_name = f'{experiment_name}_{curdate.hour}_{curdate.minute}_{curdate.second}'

if seed:
    set_random_seed(seed)

artefacts_folder = os.path.join(output_folder, 'output', experiment_name)

os.makedirs(artefacts_folder, exist_ok=True)

shutil.copy(config_path, os.path.join(artefacts_folder, 'config.yaml'))

logging_level = logging.DEBUG if config['train']['logging_level'] == 'DEBUG' else logging.INFO
setup_logger('base', artefacts_folder,
             level=logging_level, screen=True, tofile=True)
logger = logging.getLogger('base')

renderrer = renderrer(internal_dim=internal_dim_rend,
                      num_layers=num_layers_rend,
                      act=act_rend, 
                      ff_func=ff_func_rend,
                      num_input_channels=2,
                      conv=gen_conv,
                      num_freqs=num_freqs_rend,
                      )

modulator = modulator(num_input_channels=1,
                      internal_dim=internal_dim_mod,
                      num_layers=num_layers_mod,
                      num_freqs=num_freqs_mod,
                      ff_func=ff_func_mod,
                      conv=mod_conv,
                      act=act_mod,)

fmlp = ModulatedMLP(renderrer=renderrer,
                    modulator=modulator).to(device)

if checkpoint:
    logger.info(f'Loading checkpoint from {checkpoint}')
    fmlp.load_state_dict(torch.load(checkpoint))

model_parameters = filter(lambda p: p.requires_grad, fmlp.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
logger.info(f'Number of trainable parameters is {params}')

logger.info(f'Saving outputs to {artefacts_folder}')

config['num_trainable_params'] = params
wandb.init(project='ffvid',
           entity="belkakari",
           name=experiment_name,
           config=config,
           dir=os.path.join(output_folder, 'wandb'))

dset = FramesDataset(dset_folder, phase_mod=phase_mod, resolution=resolution)
dloader = DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=8)

opt = torch.optim.Adam(fmlp.parameters())

for i in range(1000):
    losses = []
    for img, coords in tqdm(dloader):
        if type(coords) is list:
            img, coords = img.to(device), [coordss.to(device) for coordss in coords]
        else:
            img, coords = img.to(device), coords.to(device)

        opt.zero_grad()
        out = torch.sigmoid(fmlp(coords))
        loss = ((img - out) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        wandb.log({"loss": loss.item()})
    
    if i % 5 == 0:
        path_to_vid = os.path.join(artefacts_folder, f'video_train_{i}.mp4')
        frames = generate_video(model=fmlp,
                                path_to_vid=path_to_vid,
                                time_range=np.linspace(-0.1, 1.1, 400),
                                device=device,
                                phase_mod=phase_mod,
                                resolution=tuple(resolution),
                                )
        wandb.log({"img": wandb.Image(frames[100], caption="100th frame")})
        if save_model:
            torch.save(fmlp.state_dict(), os.path.join(artefacts_folder, 'model.pth'))
