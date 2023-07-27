from typing import Any, List, Optional, Tuple

import torch
from torch import Tensor

from synth.nn.spec_encoder import SpecificationEncoder
from synth.specification import PBE
from synth.task import Task

from examples.pbe.karel.karel import KarelWorld

import numpy as np


class GridEncoder(SpecificationEncoder[PBE, Tensor]):
    def __init__(
        self, output_dimension: int, reseau: torch.nn.Module
    ) -> None:
        self.output_dimension = output_dimension
        self.reseau = reseau

    # Convert an integer to a binary array
    def get_binary_array(self, num: int):
        bin_str = np.binary_repr(num, width=4)
        return np.fromstring('-'.join(bin_str[i:i+1] for i in range(len(bin_str))), dtype=int, sep='-') 

    def encode_grid(self, world: Tuple[KarelWorld, Tuple[Tuple[float]]], device: Optional[str] = None) -> Tensor:

        input, output = world

        # Get numpy grids
        input_grid = input[0].grid
        output_grid = np.asarray(output)

        # Data shape
        shape = input_grid.shape
        shape = (2,) + shape + (4,)

        # Get encoding data
        # Integers grid are converted to grid of binary arrays
        e = np.append(input_grid,output_grid).astype(int)
        e = np.array([self.get_binary_array(elem) for elem in e])
        e.shape = shape

        res = torch.FloatTensor(e).to(device)
        return res

    def encode(self, task: Task[PBE], device: Optional[str] = None) -> Tensor:
        return torch.stack(
            [
                self.reseau(self.encode_grid((ex.inputs, ex.output), device)).flatten()
                for ex in task.specification.examples
            ]
        )
