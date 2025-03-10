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
from abc import ABC, abstractmethod
from typing import List
from src.models.storage.batch import Batch

from src.planner.abstract_plan import AbstractPlan
from src.utils.logging_manager import LoggingManager, LoggingLevel
from src.utils.trace_collector import TraceCollector


class AbstractExecutor(ABC):
    """
    An abstract class for the executor engine
    Arguments:
        node (AbstractPlan): Plan node corresponding to this executor
    """

    def __init__(self, node: AbstractPlan):
        self._node = node
        self._children = []
        self._current_batch = None
        self._current_file_url = None

    def append_child(self, child: 'AbstractExecutor'):
        """
        appends a child exector node

        Arguments:
            child {AbstractExecutor} -- child node
        """
        self._children.append(child)

    @property
    def children(self) -> List['AbstractExecutor']:
        """
        Returns the list of child executor
        Returns:
            [] -- list of children
        """
        return self._children

    @property
    def node(self) -> AbstractPlan:
        return self._node

    @abstractmethod
    def validate(self):
        NotImplementedError('Must be implemented in subclasses.')

    @abstractmethod
    def exec(self):
        """
        This method is implemented by every executor.
        Contains logic for that executor;
        For retrival based executor : It fetchs frame batches from
        child nodes and emits it to parent node.
        """
        NotImplementedError('Must be implemented in subclasses.')
    
    def _notify_process_batch(self, file_url: str, new_batch: Batch):
        # if not isinstance(self._current_batch, type(None)):
        #     TraceCollector().log_unfix(self._current_file_url, self._current_batch)
        self._current_batch = new_batch
        self._current_file_url = file_url
