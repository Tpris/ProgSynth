from typing import Any, List, Optional, Tuple

import torch
from torch import Tensor

from synth.nn.spec_encoder import SpecificationEncoder
from synth.specification import PBE
from synth.task import Task

from examples.pbe.karel.karel import KarelWorld


class GridEncoder(SpecificationEncoder[PBE, Tensor]):
    def __init__(
        self, output_dimension: int, lexicon: List[Any], undefined: bool = True
    ) -> None:
        self.output_dimension = output_dimension

        self.special_symbols = [
            "PADDING",  # padding symbol that can be used later
            "STARTING",  # start of entire sequence
            "ENDOFINPUT",  # delimits the ending of an input - we might have multiple inputs
            "STARTOFOUTPUT",  # begins the start of the output
            "ENDING",  # ending of entire sequence
            "STARTOFLIST",
            "ENDOFLIST",
        ]
        if undefined:
            self.special_symbols.append("UNDEFINED")
        self.lexicon = lexicon + self.special_symbols
        self.non_special_lexicon_size = len(lexicon)
        self.symbol2index = {symbol: index for index, symbol in enumerate(self.lexicon)}
        self._default = self.symbol2index["UNDEFINED"] if undefined else None
        self.starting_index = self.symbol2index["STARTING"]
        self.end_of_input_index = self.symbol2index["ENDOFINPUT"]
        self.start_of_output_index = self.symbol2index["STARTOFOUTPUT"]
        self.ending_index = self.symbol2index["ENDING"]
        self.start_list_index = self.symbol2index["STARTOFLIST"]
        self.end_list_index = self.symbol2index["ENDOFLIST"]
        self.pad_symbol = self.symbol2index["PADDING"]
        print("GRID "*4,'\n')
        print('*'*10,self.output_dimension,"\n", flush=True)
        print('*'*10,self.lexicon,"\n", flush=True)
        print('*'*10,self.non_special_lexicon_size,"\n", flush=True)
        print('*'*10,self.symbol2index,"\n", flush=True)

    def __encode_element__(self, x: Any, encoding: List[int]) -> None:
        # if isinstance(x, List):
        #     encoding.append(self.start_list_index)
        #     for el in x:
        #         self.__encode_element__(el, encoding)
        #     encoding.append(self.end_list_index)
        # else:
            encoding.append(self.symbol2index.get(x, self._default))  # type: ignore

    def encode_grid(self, IO: Tuple[List, Any], device: Optional[str] = None) -> Tensor:
        # world: KarelWorld, grid : Tuple[Tuple[float]]
        """
        embed a list of inputs and its associated output
        IO is of the form [[I1, I2, ..., Ik], O]world: KarelWorld, grid : Tuple[Tuple[float]]
        where I1, I2, ..., Ik are inputs and O is an output

        outputs a tensor of dimension self.output_dimension
        """
        e = [self.starting_index]
        print("e = ",e,"\n")
        inputs, output = IO
        print("****** input = ",inputs,'\n',flush=True)
        print("****** output : ", output,'\n', flush=True)
        for x in inputs:
            self.__encode_element__(x, e)
            e.append(self.end_of_input_index)
        e.append(self.start_of_output_index)
        self.__encode_element__(output, e)
        e.append(self.ending_index)
        size = len(e)
        if size > self.output_dimension:
            assert False, "IOEncoder: IO too large: {} > {} for {}".format(
                size, self.output_dimension, IO
            )
        else:
            for _ in range(self.output_dimension - size):
                e.append(self.ending_index)
        res = torch.LongTensor(e).to(device)
        return res

    def encode(self, task: Task[PBE], device: Optional[str] = None) -> Tensor:
        print("tutututututututu\n", flush=True)
        return torch.stack(
            [
                self.encode_grid((ex.inputs, ex.output), device)
                for ex in task.specification.examples
            ]
        )
