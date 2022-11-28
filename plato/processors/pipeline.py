"""
Implements a pipeline of processors for data payloads to pass through.
"""
import numpy as np
import torch

from typing import Any, List
from plato.processors import base


class Processor(base.Processor):
    """
    Pipelining a list of Processors from the configuration file.
    """
    def __init__(self, processors: List[base.Processor], *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.processors = processors

    def process(self, data: Any) -> Any:
        """
        Implementing a pipeline of Processors for data payloads.
        """
        for processor in self.processors:
            testing_data = torch.ones([2, 4])
            testing_data = processor.process(testing_data)

        return data
