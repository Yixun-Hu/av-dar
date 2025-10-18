import torch
import torch.nn as nn
import torch.optim as optim

import pathlib

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

import numpy as np

import datetime

from typing import List

# from avdar.config.base_config import BaseConfig
from avdar.core.io import build_from_config
from avdar.core.run import train_loop
from avdar.utils.seed_utils import seed_all
from avdar.core.typing import *


# import avdar.utils.logging_utils as logging
import logging

import termcolor


logger = logging.getLogger(__name__)

@hydra.main(version_base="1.2", config_path="config", config_name="base")
def main(cfg: DictConfig) -> None:

    out_dir = pathlib.Path(HydraConfig().get().run.dir)
    
    working_dir = pathlib.Path(cfg.working_dir)

    OmegaConf.resolve(cfg)
    config_yaml = OmegaConf.to_yaml(cfg)
    
    logger.info(config_yaml)
    logger.info(f"Output: {str(out_dir)}")
    
    with open(out_dir / "config.yaml", mode="w", encoding="utf-8") as f:
        f.write(config_yaml + "\n")

    seed = cfg.seed
    if seed >= 0:
        seed_all(seed)
        logger.info(termcolor.colored(f'Set seed: {seed}', 'yellow'))
    

    cache_dir = working_dir / (cfg.dataset.name + '_' + cfg.dataset.scene_name) 
    cache_dir.mkdir(parents=True, exist_ok=True)

    working_dir = cache_dir / cfg.train.exp_name / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    working_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Cache: {str(cache_dir)}")
    logger.info(f"Working: {str(working_dir)}")

    with open(working_dir / "config.yaml", mode="w", encoding="utf-8") as f:
        f.write(config_yaml + "\n")

    build_dict = build_from_config(cfg, working_dir, cache_dir, cfg.state_dict_path != '')
    dataset_train = build_dict['dataset_train']
    dataset_val = build_dict['dataset_val']
    dataset_test = build_dict['dataset_test']
    optimizer = build_dict['optimizer']

    rir_renderer = build_dict['rir_renderer']
    train_loop(cfg, rir_renderer, optimizer, dataset_train, dataset_test, working_dir, out_dir)

if   __name__ == "__main__":
    main()
    print('Done')