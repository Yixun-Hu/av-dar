import open3d as o3d

import os
import random
import numpy as np
import torch


# BUG: Not working for some reason
def seed_all(seed: int) -> int:
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    o3d.utility.random.seed(seed)

    return seed