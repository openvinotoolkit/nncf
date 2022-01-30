"""
 Copyright (c) 2022 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from typing import List


class PriorityQueue:
    def __init__(self, queue: List[object] = None):
        self.queue = []
        self.length = 0
        if queue is not None:
            for element in queue:
                self.add(element)

    def add(self, element: object):
        self.queue.append(element)
        self.length += 1
        self.sift_up(self.length - 1)

    def sift_up(self, i: int):
        if i == 0:
            return
        if (i % 2) == 1:
            parent = i // 2
        else:
            parent = (i // 2) - 1
        if self.queue[parent] < self.queue[i]:
            self.queue[parent], self.queue[i] = self.queue[i], self.queue[parent]
            self.sift_up(parent)

    def sift_down(self, i: int):
        left_child, right_child = i * 2 + 1, i * 2 + 2
        if left_child > self.length - 1:  # No children
            return
        elif right_child > self.length - 1:  # Only left child
            if self.queue[left_child] > self.queue[i]:
                self.queue[i], self.queue[left_child] = self.queue[left_child], self.queue[i]
        else:  # Two children
            if self.queue[left_child] > self.queue[right_child]:
                if self.queue[left_child] > self.queue[i]:
                    self.queue[i], self.queue[left_child] = self.queue[left_child], self.queue[i]
                    self.sift_down(left_child)
            elif self.queue[right_child] > self.queue[i]:
                self.queue[i], self.queue[right_child] = self.queue[right_child], self.queue[i]
                self.sift_down(right_child)

    def pop(self) -> object:
        self.queue[0], self.queue[self.length - 1] = self.queue[self.length - 1], self.queue[0]
        elem = self.queue.pop()
        self.length -= 1
        self.sift_down(0)
        return elem

    def is_empty(self) -> bool:
        return self.length == 0

    def __repr__(self):
        return str(self.queue)
