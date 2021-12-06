# coding=utf-8
# Copyright 2018-2020 EVA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Dict

import pandas as pd
import torchvision
import numpy as np
import os
from random import Random

from torch import Tensor
import torch
from src.models.catalog.frame_info import FrameInfo
from src.models.catalog.properties import ColorSpace
from src.udfs.pytorch_abstract_udf import PytorchAbstractUDF
from src.configuration.dictionary import EVA_DIR
from src.utils.logging_manager import LoggingManager, LoggingLevel


class MNISTDigitDetector(PytorchAbstractUDF):
    """
    Arguments:
        threshold (float): Threshold for classifier confidence score

    """

    @property
    def name(self) -> str:
        return "mnistnn"

    def __init__(self, threshold=0.85):
        super().__init__()
        self.threshold = threshold
        custom_model_path = os.path.join(EVA_DIR, "data", "models", "mnist.pt")
        self.model = torch.load(custom_model_path)
        self.model.eval()

    @property
    def input_format(self) -> FrameInfo:
        return FrameInfo(-1, -1, 3, ColorSpace.RGB)

    @property
    def labels(self) -> List[str]:
        return [
            '0',
            '1',
            '2',
            '3',
            '4',
            '5',
            '6',
            '7',
            '8',
            '9'
        ]

    def _get_predictions(self, frames: Tensor) -> pd.DataFrame:
        """
        Performs predictions on input frames
        Arguments:
            frames (np.ndarray): Frames on which predictions need
            to be performed

        Returns:
            tuple containing predicted_classes (List[List[str]]),
            predicted_scores (List[List[float]])

        """
        frames = torch.narrow(frames, 1, 0, 1)
        frames = frames.view(frames.shape[0], -1)
        predictions = self.model(frames)
        outcome_list = []
        for prediction in predictions:
            ps = torch.exp(prediction)
            probab = list(ps.detach().numpy())
            label = str(probab.index(max(probab)))
            outcome_list.append(
                {
                    "labels": [label],
                    "scores": [max(probab)],
                })
        return pd.DataFrame.from_dict(outcome_list)
