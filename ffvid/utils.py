import os
from math import pi
import random
import logging
from time import gmtime, strftime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm.auto import tqdm


def get_grid(h, w, b=0, norm=True, device='cpu'):
    if norm:
        xgrid = np.linspace(0, w, num=w) / w
        ygrid = np.linspace(0, h, num=h) / h
    else:
        xgrid = np.linspace(0, w, num=w)
        ygrid = np.linspace(0, h, num=h)
    xv, yv = np.meshgrid(xgrid, ygrid, indexing='xy')
    grid = np.stack([xv, yv], axis=-1)[None]

    grid = torch.from_numpy(grid).float().to(device)
    if b > 0:
        grid = grid.expand(b, -1, -1, -1)  # [Batch, H, W, UV]
        return grid.permute(0, 3, 1, 2)  # [Batch, UV, H, W]
    else:
        return grid[0].permute(2, 0, 1)  # [UV, H, W]


@torch.no_grad()
def generate_video(model,
                   path_to_vid,
                   fps=60.0,
                   resolution=(256, 256),
                   time_range=np.linspace(-0.2, 1.2, 200),
                   device='cpu',
                   phase_mod=True):
    frames = []
    out_file = cv2.VideoWriter(path_to_vid, cv2.VideoWriter_fourcc(*'mp4v'), fps, resolution)
    coords = get_grid(resolution[1], resolution[0], 1, True, device)
    for frame in tqdm(time_range):
        if phase_mod:
            coords_stacked = [coords, torch.cat([coords, coords[:, [0]] * 0. + frame], dim=1)]
        else:
            coords_stacked = torch.cat([coords, coords[:, [0]] * 0. + frame], dim=1)
        out = torch.sigmoid(model(coords_stacked))
        prepared_frame = (out[0].permute(1, 2, 0).cpu().data.numpy()[:, :, ::-1] * 255).clip(0, 255).astype(np.uint8)
        out_file.write(prepared_frame)
        frames.append(prepared_frame)
    out_file.release()
    return frames


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_timestamp():
    return strftime("%Y-%m-%d-%H:%M:%S", gmtime())


def setup_logger(logger_name, root, level=logging.INFO,
                 screen=False, tofile=False):
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, f'_{get_timestamp()}.log')
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)
