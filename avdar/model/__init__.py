from abc import ABC, abstractmethod

from ..utils.registry_utils import Registry
from ..utils.import_utils import import_children

import torch.nn as nn

module_registry = Registry("module", nn.Module)
import_children(__file__, __name__)