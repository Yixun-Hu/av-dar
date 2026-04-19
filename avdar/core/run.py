import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from torch.utils.tensorboard.writer import SummaryWriter

import torchaudio.functional as F

import numpy as np
import datetime
import pathlib

from avdar.core.base_config import BaseConfig
from avdar.geometry.pathspace import SpecularPathSampler
from avdar.model.renderer import RirRenderer
from avdar.data import AcousticDataset

from avdar.utils.visualize_utils import loss_table
from avdar.utils.sample_utils import generate_pink_noise
from avdar.utils.loss_utils import (
    training_loss,
    RafC50Error, RafEdtError, RafT60Error, decay_loss, 
    DiffRirLoss, RafLoudnessError,
)
from avdar.utils.io_utils import save_json

import matplotlib.pyplot as plt
from typing import Dict, Optional, Callable

from tqdm import tqdm
import logging
import termcolor

logger = logging.getLogger(__name__)

def train_step(
        cfg: BaseConfig,
        epoch: int,
        renderer: RirRenderer,
        dataset: data.Dataset,
        pathsampler: SpecularPathSampler,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler._LRScheduler,
        criterion: nn.Module,
        loss_name: str = 'loss',
        enable_pink_noise: bool = False,
        pink_noise_lambda: float = 0.2,
        enable_decay_loss: bool = False,
        decay_lambda: float = 5.0,
        writer: SummaryWriter = None,
        dataset_test: data.Dataset = None,
        working_dir: pathlib.Path = None,
        tot_iter: int = 0,
):
    
    indices = torch.arange(len(dataset))

    if cfg.train.shuffle:
        indices = torch.randperm(len(dataset))

    pbar = tqdm(indices, desc='Training') if not cfg.no_terminal else indices
    epoch_loss = 0
    batch_size = cfg.train.batch_size
    optimizer.zero_grad()

    losses = {
        loss_name: 0,
        'pink': 0,
        'decay': 0,
    }
    l1_reg_loss_fn = nn.L1Loss(reduction='sum')

    grad_clip = cfg.train.clip_value if 'clip_value' in cfg.train else None

    for i, data_idx in enumerate(pbar):
        data = dataset[data_idx]
        rir_gt = data['rir'].to(cfg.device)
        # source_idx = data['source_idx']
        # listener_idx = data['listener_idx']
        source_xyz = data['source_xyz']
        listener_xyz = data['listener_xyz']
        rotation = data.get('source_rotation', None)
        quat = data.get('source_rotation_quat', None)


        paths = directions = lengths = None
        mc_samples = pathsampler.fast_sample(source_xyz.cpu().numpy(), listener_xyz.cpu().numpy())

        pred_dict = renderer(paths, directions, lengths, None, rotation, source_xyz, listener_xyz, quat, mc_samples=mc_samples)
        rir_pred = pred_dict['rir_full']
        rir_ambient = pred_dict['rir_ambient']
        loss_output = criterion(rir_pred, rir_gt)

        if isinstance(loss_output, dict):
            loss = 0
            for k, v in loss_output.items():
                loss += v
                if k not in losses:
                    losses[k] = 0
                losses[k] += v.item()
        elif isinstance(loss_output, tuple):
            loss = 0
            for v in loss_output:
                loss += v
        else:
            loss = loss_output

        if 'enable_reg_ambient' in cfg.train:
            lambda_ambient = cfg.train.enable_reg_ambient['lambda']
            multiplier = cfg.train.enable_reg_ambient.get('multiplier', 'const')
            if multiplier == 'const':
                times = 1
            elif multiplier == 'linear':
                times = torch.arange(len(rir_ambient), device=rir_ambient.device, dtype=rir_ambient.dtype) / dataset.sample_rate
            else:
                raise ValueError(f"Multiplier {multiplier} not supported")
            
            # reg_loss = l1_reg_loss_fn(rir_ambient, torch.zeros_like(rir_ambient))
            rir_ambient_smoothed = torch.nn.functional.conv1d(
                (rir_ambient * times)[None, None], 
                torch.ones(1, 1, 100, dtype=rir_ambient.dtype, device=rir_ambient.device) / 100, padding=50
            )[0, 0]
            reg_loss_bunch = criterion(rir_ambient_smoothed, torch.zeros_like(rir_ambient_smoothed))

            reg_loss = 0
            if isinstance(reg_loss_bunch, dict):
                for k, v in reg_loss_bunch.items():
                    reg_loss += v
            elif isinstance(reg_loss_bunch, tuple):
                for v in reg_loss_bunch:
                    reg_loss += v
            else:
                reg_loss = reg_loss_bunch

            loss += lambda_ambient * reg_loss
            losses['reg_ambient'] = losses.get('reg_ambient', 0) + reg_loss.item()


        losses[loss_name] += loss.item()

        if enable_pink_noise:
            pink_noise = generate_pink_noise(5 * dataset.sample_rate, fs = dataset.sample_rate)
            pink_noise = pink_noise.to(cfg.device)
            
            convolved_pred = F.fftconvolve(rir_pred, pink_noise)[..., :5*dataset.sample_rate]
            convolved_gt =  F.fftconvolve(rir_gt, pink_noise)[..., :5*dataset.sample_rate]
            pink_loss = criterion(convolved_pred, convolved_gt)
            loss += pink_noise_lambda * pink_loss
            losses['pink'] += pink_loss.item()

        if enable_decay_loss is not None and enable_decay_loss:
            dloss = decay_loss(rir_pred, rir_gt)
            loss += decay_lambda * dloss
            losses['decay'] += dloss.item()

        if not cfg.no_terminal:
            pbar.set_description(f"Training: Loss: {loss.item():.6f}")
        
        epoch_loss += loss.item()

        loss = loss / batch_size
        loss.backward()
        
        if (i + 1) % batch_size == 0 or i == (len(indices) - 1):
            # torch.nn.utils.clip_grad_value_(renderer.parameters(), clip_value=0.5)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(renderer.parameters(), grad_clip)
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            optimizer.zero_grad()

        tot_iter += 1
        if 'vis_iter_num' in cfg.train \
            and (tot_iter % cfg.train['vis_iter_num'] == 0) \
            and dataset_test is not None \
            and working_dir is not None:

            logger.info(f'Visualizing...')
            vis_dir = working_dir / f'visualizations/epoch_{epoch}_{tot_iter}'
            vis_dir.mkdir(parents=True, exist_ok=True)

            for i in range(5):
                vis_data = dataset_test[i]
                rir_gt = vis_data['rir']
                source_idx = vis_data['source_idx']
                listener_idx = vis_data['listener_idx']
                source_xyz = vis_data['source_xyz']
                listener_xyz = vis_data['listener_xyz']
                source_orient = vis_data.get('source_rotation_quat', None)
                rotation = vis_data.get('source_rotation', None)

                mc_samples = pathsampler.fast_sample(source_xyz.cpu().numpy(), listener_xyz.cpu().numpy())
                path_xyzs = mc_samples['path_xyzs']
                visualize_source_listener_xyzs(dataset_test.points, source_xyz, listener_xyz, rotation, path_xyzs, vis_dir / f'path_{i}.png', axes = (2, 0, 1) if cfg.dataset.name == 'real_acoustic_field' else (0, 1, 2))
                renderer.model_visulization_mc(vis_dir, 
                                                writer, epoch, i,
                                                mc_samples, None, 
                                                rotation, source_xyz, listener_xyz, source_orient, rir_gt)

            if  cfg.train.visualization.wave_dist:
                visualize_wave_xyzs(dataset_test, pathsampler, dataset_test.points, renderer, pathsampler.max_path_length, vis_dir, f'epoch_{epoch}', 
                        axes = (2, 0, 1) if cfg.dataset.name == 'real_acoustic_field' else (0, 1, 2))
    
    # if (i + 1) % batch_size != 0:
    #     # torch.nn.utils.clip_grad_value_(renderer.parameters(), clip_value=0.5)
    #     grad_clip_fn(renderer.parameters())
    #     optimizer.step()
    #     optimizer.zero_grad()

    if writer is not None:
        for k, v in losses.items():
            writer.add_scalar(f'train/{k}', v / len(dataset), epoch)
        writer.add_scalar('train/total', epoch_loss / len(dataset), epoch)

    return epoch_loss / len(dataset), tot_iter


@torch.no_grad()
def eval_step_rir(
        cfg: BaseConfig,
        renderer: nn.Module,
        dataset: data.Dataset,
        pathsampler: SpecularPathSampler,
        criterions: Dict[str, nn.Module],
):
    losses = {k: 0 for k in criterions.keys()}
    losses_curve = {k: [] for k in criterions.keys()}

    pbar = tqdm(dataset, desc='Evaluating') if not cfg.no_terminal else dataset

    for i, (data) in enumerate(pbar):
        rir_gt = data['rir'].to(cfg.device)
        source_xyz = data['source_xyz']
        listener_xyz = data['listener_xyz']
        rotation = data.get('source_rotation', None)
        quat = data.get('source_rotation_quat', None)

        paths = directions = lengths = None
        mc_samples = pathsampler.fast_sample(source_xyz.cpu().numpy(), listener_xyz.cpu().numpy())
    

        pred_dict = renderer(paths, directions, lengths, None, rotation, source_xyz, listener_xyz, quat, mc_samples = mc_samples)
        rir_pred = pred_dict['rir_full'].detach()
        for k, criterion in criterions.items():
            try:
                loss = criterion(rir_pred.cpu(), rir_gt.cpu()).item()
            except Exception as e:
                logger.warning(f'Error computing loss {k}: {e}')
                loss = np.nan
            losses[k] += loss
            losses_curve[k].append(loss)
       
        if not cfg.no_terminal:
            loss_text = ' '.join([f"{k}: {v / (i + 1):.6f}" for k, v in losses.items()])
            pbar.set_description(f"Evaluating: [{loss_text}]")

    return {k: v / len(dataset) for k, v in losses.items()}

  

def train_loop(
        cfg: BaseConfig, 
        rir_renderer: RirRenderer, 
        optimizer: optim.Optimizer,
        dataset_train: AcousticDataset, 
        dataset_test: AcousticDataset, 
        working_dir: str, 
        out_dir: str):
    """
    Run the full RIR training schedule for one experiment.

    Trains ``rir_renderer`` for ``cfg.train.n_epochs``, optionally logs to
    TensorBoard, saves checkpoints and per-epoch evaluation metrics under
    ``working_dir``, runs visualization on a fixed interval, and gradually
    increases specular path order (``max_path_length``) after
    ``start_growing_epoch``. Writes ``weight_final.pt``, ``losses.npy``, and a
    final ``eval_losses_epoch_final.json`` when training completes.

    Parameters
    ----------
    cfg : BaseConfig
        Full Hydra config (dataset, train, paths, device, tensorboard, etc.).
    rir_renderer : RirRenderer
        Differentiable room renderer to optimize.
    optimizer : optim.Optimizer
        Optimizer bound to ``rir_renderer`` parameters.
    dataset_train : AcousticDataset
        Training split (RIR supervision).
    dataset_test : AcousticDataset
        Test split used for evaluation and optional visualization samples.
    working_dir : str
        Directory for checkpoints, logs, tensorboard, eval JSON, and images
        (converted to ``pathlib.Path`` inside this function).
    out_dir : str
        Hydra run output directory (reserved for callers; not heavily used here).

    Returns
    -------
    None
        Side effects only: files under ``working_dir`` and console / TensorBoard
        logging.
    """

    working_dir = pathlib.Path(working_dir)
    out_dir = pathlib.Path(out_dir)
    
    writer = None
    if cfg.tensorboard is not None:
        writer = SummaryWriter(log_dir=working_dir / cfg.tensorboard)

    # rir_renderer = rir_renderer.to(cfg.device)
    rir_renderer.train()
    # print(cfg.device)
    optimizer.zero_grad()
    
    max_path_length = cfg.train.start_bounce
    # in the first epoch, max_path_length = 1
    mc_path_sampler = SpecularPathSampler.from_config(cfg.train.sampler_opts, max_path_length, dataset_test.get_mesh_path())

    losses = []

    loss_fn = training_loss
    loss_name = 'loss'

    if cfg.train.loss_opts.name == 'diffrir_loss':
        options = cfg.train.loss_opts.options
        logger.info('Using diffrir loss...')
        logger.info(f'diff_rir: {options.get("nffts", None)} ')
        loss_fn = DiffRirLoss(**options)
        loss_name = 'diffrir_loss'

    if writer is not None:
        writer.add_scalar('config/path_length', max_path_length, 0)

    lr_scheduler = None
    if hasattr(cfg.train, 'lr_scheduler'):
        lr_scheduler = getattr(optim.lr_scheduler, cfg.train.lr_scheduler.name)(optimizer, **cfg.train.lr_scheduler.options)

    tot_iter = 0
    for epoch in range(cfg.train.n_epochs):

        logger.info(f'Epoch {epoch}/{cfg.train.n_epochs}: max_path_length={max_path_length}')
        
        start_time = datetime.datetime.now()
        loss, tot_iter = train_step(
            cfg, epoch,
            rir_renderer, dataset_train, mc_path_sampler, optimizer, lr_scheduler,
            loss_fn, loss_name,
            cfg.train.pink_noise_supervision and (epoch > cfg.train.pink_start_epoch), 
            cfg.train.lambda_pink, cfg.train.decay_loss, cfg.train.decay_loss_lambda, 
            writer, dataset_test, working_dir, tot_iter
        )
        losses.append(loss)
        loss_text = termcolor.colored(f'{loss:.6f}', 'yellow')
        elapsed = datetime.datetime.now() - start_time
        logger.info(f'Epoch {epoch}/{cfg.train.n_epochs}: {loss_text} elapse (H:MM:SS.MI){elapsed}')

        if writer is not None:
            writer.add_scalar('statistic/epoch_train_time', elapsed.total_seconds(), epoch)

        if cfg.train.save_interval is not None and epoch % cfg.train.save_interval == 0:
            torch.save(
                {'model_state_dict': rir_renderer.state_dict(), 
                 'optimizer_state_dict': optimizer.state_dict(),}, 
                    working_dir / f'weight_epoch_{epoch}.pt')
            np.save(working_dir / 'losses', np.array(losses))

        if cfg.train.eval_interval is not None and (epoch % cfg.train.eval_interval == 0) and epoch > 0:
            logger.info(f'Evaluation...')
            start_time = datetime.datetime.now()
            eval_loss_dict = eval_step_rir(
                cfg, rir_renderer, dataset_test, mc_path_sampler, 
                {
                 'C50': RafC50Error(dataset_test.sample_rate), 
                 'EDT': RafEdtError(dataset_test.sample_rate),
                 'T60': RafT60Error(dataset_test.sample_rate, 20),
                 'Loudness': RafLoudnessError(dataset_test.sample_rate),
                }
            )
            elapsed = datetime.datetime.now() - start_time

            if writer is not None:
                for name, value in eval_loss_dict.items():
                    writer.add_scalar(f'eval/{name}', value, epoch)
                writer.add_scalar('statistic/epoch_eval_time', elapsed.total_seconds(), epoch)
        
            print(loss_table(eval_loss_dict))
            save_json(eval_loss_dict, working_dir / f'eval_losses_epoch{epoch}.json')

        if cfg.train.vis_interval is not None and cfg.train.vis_interval >= 1 and epoch % cfg.train.vis_interval == 0:
            logger.info(f'Visualizing...')
            vis_dir = working_dir / f'visualizations/epoch_{epoch}'
            vis_dir.mkdir(parents=True, exist_ok=True)

            for i in range(5):
                vis_data = dataset_test[i]
                rir_gt = vis_data['rir']
               
                source_xyz = vis_data['source_xyz']
                listener_xyz = vis_data['listener_xyz']
                source_orient = vis_data.get('source_rotation_quat', None)
                rotation = vis_data.get('source_rotation', None)


                mc_samples = mc_path_sampler.fast_sample(source_xyz.cpu().numpy(), listener_xyz.cpu().numpy())
                path_xyzs = mc_samples['path_xyzs']
                visualize_source_listener_xyzs(dataset_train.points, source_xyz, listener_xyz, rotation, path_xyzs, vis_dir / f'path_{i}.png', axes = (2, 0, 1) if cfg.dataset.name == 'real_acoustic_field' else (0, 1, 2))
                rir_renderer.model_visulization_mc(vis_dir, 
                                                writer, epoch, i,
                                                mc_samples, None, 
                                                rotation, source_xyz, listener_xyz, source_orient, rir_gt)
            
            if  cfg.train.visualization.wave_visualization:
                visualize_wave_xyzs(dataset_test, mc_path_sampler, dataset_train.points, rir_renderer, max_path_length, working_dir / f'visualizations/epoch_{epoch}', f'epoch_{epoch}', 
                        axes = (2, 0, 1) if cfg.dataset.name == 'real_acoustic_field' else (0, 1, 2))
        
            

    
        if epoch > cfg.train.start_growing_epoch and epoch % cfg.train.growing_interval == 0 and max_path_length < cfg.train.max_bounce:
            if writer is not None:
                writer.add_scalar('config/path_length', max_path_length, epoch)
            max_path_length = max_path_length + 1
            mc_path_sampler.reset_length(max_path_length)


    torch.save(
        {'model_state_dict': rir_renderer.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict(),}, 
            working_dir / f'weight_final.pt')
    np.save(working_dir / 'losses', np.array(losses))

    logger.info(f'Evaluation...')
    start_time = datetime.datetime.now()
    eval_loss_dict = eval_step_rir(
        cfg, rir_renderer, dataset_test, mc_path_sampler, 
        {
            'C50': RafC50Error(dataset_test.sample_rate), 
            'EDT': RafEdtError(dataset_test.sample_rate), 
            'T60': RafT60Error(dataset_test.sample_rate, 20),
            'Loudness': RafLoudnessError(dataset_test.sample_rate),
        }
    )
    elapsed = datetime.datetime.now() - start_time

    if writer is not None:
        for name, value in eval_loss_dict.items():
            writer.add_scalar(f'eval/{name}', value, epoch)
        writer.add_scalar('statistic/epoch_eval_time', elapsed.total_seconds(), epoch)

    print(loss_table(eval_loss_dict))
    save_json(eval_loss_dict, working_dir / f'eval_losses_epoch_final.json')


def visualize_source_listener_xyzs(xyzs, source_xyz, listener_xyz, rot, path_xyzs, save_path, axes = (2, 0, 1)):
    plt.figure(figsize=(4, 3))
    p_color = np.array([0.5, 0.5, 0.5])
    yellow = plt.get_cmap('Paired')(5.5/12)

    source_xyz = np.array(source_xyz)
    listener_xyz = np.array(listener_xyz)
    plt.scatter(xyzs[:100000, axes[0]], xyzs[:100000, axes[1]], c=p_color, s=2, alpha=0.005)
    plt.scatter(source_xyz[axes[0]], source_xyz[axes[1]], c=yellow, s=50, alpha=1.0, label='speaker')
    plt.scatter(listener_xyz[axes[0]], listener_xyz[axes[1]], c=yellow, s=50, marker='x', alpha=1.0, label='microphone')
    if rot is not None:
        dxy = np.array([0, 0, -1])
        dxy = rot @ dxy
        dxy *= 0.5
        plt.arrow(source_xyz[axes[0]], source_xyz[axes[1]], dxy[axes[0]], dxy[axes[1]], head_width=0.1, head_length=0.1, fc='k', ec='k')

    for path in path_xyzs:
        path = [source_xyz] + list(path) + [listener_xyz]
        for i in range(len(path)-1):
            src = path[i]
            dst = path[i + 1]
            # src = xyzs[path[i]]
            # next_idx = path[i + 1] if path[i + 1] != -1 else listener_idx
            # dst = xyzs[next_idx]
            plt.plot([src[axes[0]], dst[axes[0]]], [src[axes[1]], dst[axes[1]]], c='r', alpha=0.2)
    # plt.title(f'Path ({data_idx+1}/{len(dataset)}; path_length={path_sampler_vis.max_path_length})')
    plt.tight_layout()
    plt.legend()
    # no scale
    plt.axis('equal')
    # no axis
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()



def visualize_source_listener_ax_xyzs(xyzs, source_xyz, listener_xyz, rot, path_xyzs, fig, ax, axis0=2, axis1=0, axis2=1):
    p_color = np.array([0.5, 0.5, 0.5])
    yellow = plt.get_cmap('Paired')(5.5/12)

    source_xyz = np.array(source_xyz)
    listener_xyz = np.array(listener_xyz)
    ax.scatter(xyzs[:100000, axis0], xyzs[:100000, axis1], c=p_color, s=2, alpha=0.005)
    ax.scatter(source_xyz[axis0], source_xyz[axis1], c=yellow, s=50, alpha=1.0, label='speaker')
    ax.scatter(listener_xyz[axis0], listener_xyz[axis1], c=yellow, s=50, marker='x', alpha=1.0, label='microphone')
    
    if rot is not None:
        dxy = np.array([0, 0, -1])
        dxy = rot @ dxy
        dxy *= 0.5
        ax.arrow(source_xyz[axis0], source_xyz[axis1], dxy[axis0], dxy[axis1], head_width=0.1, head_length=0.1, fc='k', ec='k')
    
    for path in path_xyzs:
        path = [source_xyz] + list(path) + [listener_xyz]
        for i in range(len(path)-1):
            src = path[i]
            dst = path[i+1]
            ax.plot([src[axis0], dst[axis0]], [src[axis1], dst[axis1]], c='r', alpha=0.2)
    # plt.title(f'Path ({data_idx+1}/{len(dataset)}; path_length={path_sampler_vis.max_path_length})')
    # plt.tight_layout()
    # plt.legend()
    ax.axis('equal')
    ax.axis('off')
    ax.legend()

    
def visualize_wave_xyzs(dataset_test, mc_sampler, xyzs, rir_renderer: RirRenderer, max_path_length, working_dir, name, axes=(2, 0, 1)):
    logger.info(f'Inferencing...')
    data_idx = np.random.randint(len(dataset_test))
    # data_idx = 4711
    logger.info(f'Inferencing data index {data_idx}')
    
    data_val = dataset_test[data_idx]
    # source_idx = data_val['source_idx']
    # listener_idx = data_val['listener_idx']
    rot = data_val.get('source_rotation', None)
    source_xyz = data_val['source_xyz']
    olistener_xyz = listener_xyz = data_val['listener_xyz']
    quat = data_val.get('source_rotation_quat', None)

    mc_samples = mc_sampler.fast_sample(source_xyz.cpu().numpy(), listener_xyz.cpu().numpy())
    path_xyzs = mc_samples['path_xyzs']
    visualize_source_listener_xyzs(xyzs, source_xyz, listener_xyz, rot, path_xyzs, working_dir / f'path_{data_idx}_samples.png', axes)

    # rir_renderer.model_visulization(working_dir, None, 'none', 'wave', paths_, directions, lengths, None, rot, source_xyz, data_val['listener_xyz'], quat, data_val['rir'])
    rir_renderer.model_visulization_mc(working_dir, None, 'none', 'wave', mc_samples, None, rot, source_xyz, listener_xyz, quat, data_val['rir'])

    x_min = xyzs[:, axes[1]].min()
    x_max = xyzs[:, axes[1]].max()
    z_min = xyzs[:, axes[0]].min()
    z_max = xyzs[:, axes[0]].max()

    x_min = min(x_min, z_min)
    x_max = max(x_max, z_max)
    z_min = x_min
    z_max = x_max


    y = source_xyz[axes[2]]
    num_xs = 128
    num_zs = 128
    xs = np.linspace(x_min, x_max, num_xs)
    zs = np.linspace(z_min, z_max, num_zs)
    xz = np.meshgrid(xs, zs)
    # zx = np.meshgrid(zs, xs)

    ij = np.meshgrid(np.arange(num_xs), np.arange(num_zs))
    # wave_i = np.round(1 / dataset_test.speed_of_sound * dataset_test.sample_rate).astype(int) # wave length = 1m
    # wave_i = np.round(0.3 / dataset_test.speed_of_sound * dataset_test.sample_rate).astype(int) # wave length = 0.3m
    wave_lengths = [0.1, 0.3, 1, 3, 10]
    wave_ids = [np.round(dataset_test.speed_of_sound / wave_length / dataset_test.sample_rate * (dataset_test.rir_length //2 + 1)).astype(int) for wave_length in wave_lengths]
    # xyzs = np.stack([xz[0].flatten(), y * np.ones_like(xz[0].flatten()), xz[1].flatten()], axis=1)

    # xyzijs = np.stack([xz[0].flatten(), y * np.ones_like(xz[0].flatten()), xz[1].flatten(), ij[0].flatten(), ij[1].flatten()], axis=1)
    xyzijs_loose = [None] * 5
    xyzijs_loose[3] = ij[0].flatten()
    xyzijs_loose[4] = ij[1].flatten()
    # xyzijs_loose[axes[0]] = zx[0].flatten()
    xyzijs_loose[axes[0]] = xz[1].flatten()
    xyzijs_loose[axes[1]] = xz[0].flatten()
    xyzijs_loose[axes[2]] = y * np.ones_like(xyzijs_loose[axes[0]])

    xyzijs = np.stack(xyzijs_loose, axis=1)
    phase = np.zeros((len(wave_lengths), num_xs, num_zs))
    amp = np.zeros((len(wave_lengths), num_xs, num_zs))
    loudness = np.zeros((num_xs, num_zs))
    log_loudness = np.zeros((num_xs, num_zs))

    has_samples = np.zeros((num_xs + 2, num_zs + 2))
    sample_xs = ((xyzs[:, axes[1]] - x_min) / (x_max - x_min)) * (num_xs - 1)
    sample_zs = ((xyzs[:, axes[0]] - z_min) / (z_max - z_min)) * (num_zs - 1)
    sample_zxs = np.stack([sample_zs, sample_xs], axis=1)
    has_samples[sample_zxs[:, 1].astype(int) + 1, sample_zxs[:, 0].astype(int) + 1] += 1
    valid_samples = has_samples[1:-1, 1:-1] + np.min(
        [has_samples[:-2, 1:-1], has_samples[2:, 1:-1], has_samples[1:-1, :-2], has_samples[1:-1, 2:]], axis=0)

    src_mc_sampler = mc_sampler.get_sampler(source_xyz.cpu().numpy())
    for xyzij in xyzijs:
        i = np.round(xyzij[3]).astype(int)
        j = np.round(xyzij[4]).astype(int)

        if valid_samples[i, j] == 0:
            phase[:, i, j] = np.nan
            amp[:, i, j] = np.nan
            loudness[i, j] = np.nan
            log_loudness[i, j] = np.nan
            continue

        xyz = xyzij[:3]

        listener_xyz = torch.from_numpy(xyz).to(source_xyz.dtype)
        mc_samples = src_mc_sampler(xyz)
        pred_dict = rir_renderer.forward(None, None, None, None, rot, source_xyz, listener_xyz, quat, mc_samples)
        rir_pred = pred_dict['rir_full'].detach().cpu().numpy()
        rir_freq = np.fft.rfft(rir_pred)
        for wid, wave_i in enumerate(wave_ids):
            phase[wid, i, j] = np.angle(rir_freq[wave_i])
            amp[wid, i, j] = np.abs(rir_freq[wave_i])
        loudness[i, j] = np.sum(rir_pred ** 2)
        log_loudness[i, j] = np.log10(np.sum(rir_pred ** 2 + 1e-10))

    for wid, wave_length in enumerate(wave_lengths):
        
        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        im = ax[0].imshow(phase[wid], cmap='coolwarm', origin='lower', vmin=-np.pi, vmax=np.pi, interpolation='nearest')
        fig.colorbar(im, ax=ax[0])
        ax[0].set_title('Phase')
        im = ax[1].imshow(amp[wid], cmap='hot', origin='lower')
        fig.colorbar(im, ax=ax[1])
        ax[1].set_title('Amplitude')
        im = ax[2].imshow(log_loudness, cmap='hot', origin='lower')
        fig.colorbar(im, ax=ax[2])
        ax[2].set_title('Loudness')
        im = visualize_source_listener_ax_xyzs(xyzs, source_xyz, olistener_xyz, rot, path_xyzs, fig, ax[3], axes[0], axes[1], axes[2])
        plt.tight_layout()
        plt.savefig(working_dir / f'wave_dist_{name}_{wave_length}.png')
        plt.close(fig)

    fig, ax = plt.subplots(2, len(wave_lengths) + 1, figsize=(20, 10))
    for wid, wave_length in enumerate(wave_lengths):
        im = ax[0, wid].imshow(phase[wid], cmap='coolwarm', origin='lower', vmin=-np.pi, vmax=np.pi, interpolation='nearest')
        ax[0, wid].set_title(f'Phase ({wave_length}m)')
        im = ax[1, wid].imshow(amp[wid], cmap='hot', origin='lower')
        ax[1, wid].set_title(f'Amplitude ({wave_length}m)')
    im = ax[0, -1].imshow(loudness, cmap='hot', origin='lower')
    ax[0, -1].set_title('Loudness')
    im = visualize_source_listener_ax_xyzs(xyzs, source_xyz, olistener_xyz, rot, path_xyzs, fig, ax[1, -1], axes[0], axes[1], axes[2])
    plt.tight_layout()
    plt.savefig(working_dir / f'wave_dist_{name}.png')
    plt.close(fig)
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.imshow(loudness, cmap='inferno', origin='lower')
    plt.subplot(1, 2, 2)
    plt.imshow(log_loudness, cmap='inferno', origin='lower')
    plt.savefig(working_dir / f'wave_dist_{name}_loudness.png')
    plt.close()
    print(f'Inference done. Results saved at {working_dir}')

    logger.info(f'Inference done. Results saved at {working_dir}')

