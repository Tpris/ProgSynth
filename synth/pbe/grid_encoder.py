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
        self, output_dimension: int, lexicon: List[Any], undefined: bool = True
    ) -> None:
        self.output_dimension = output_dimension
        self.lexicon = lexicon

    def encode_grid(self, world: Tuple[KarelWorld, Tuple[Tuple[float]]], device: Optional[str] = None) -> Tensor:
        # world : Tuple(KarelWorld, KW.state)

        input, output = world

        # Get numpy grids
        input_grid = input[0].grid
        output_grid = np.asarray(output)

        # Data shape
        shape = input_grid.shape
        shape = (2,) + shape

        # Get encoding data
        e = np.append(input_grid,output_grid)
        e.shape = shape

        res = torch.LongTensor(e).to(device)
        return res

    def encode(self, task: Task[PBE], device: Optional[str] = None) -> Tensor:
        return torch.stack(
            [
                self.encode_grid((ex.inputs, ex.output), device)
                for ex in task.specification.examples
            ]
        )
