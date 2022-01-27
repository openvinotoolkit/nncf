from typing import List


class PriorityQueue:
    def __init__(self, queue: List[int] = None):
        self.queue = []
        self.length = 0
        if queue is not None:
            for element in queue:
                self.append(element)

    def append(self, element):
        self.queue.append(element)
        self.length += 1
        self.sift_up(self.length - 1)

    def sift_up(self, i):
        if i == 0:
            return
        if (i % 2) == 1:
            parent = i // 2
        else:
            parent = (i // 2) - 1
        if self.queue[parent] < self.queue[i]:
            self.queue[parent], self.queue[i] = self.queue[i], self.queue[parent]
            self.sift_up(parent)

    def sift_down(self, i):
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

    def pop(self):
        self.queue[0], self.queue[self.length - 1] = self.queue[self.length - 1], self.queue[0]
        elem = self.queue.pop()
        self.length -= 1
        self.sift_down(0)
        return elem

    def is_empty(self) -> bool:
        return self.length == 0

    def __repr__(self):
        return str(self.queue)


class CompressionBuilder:
    """
    Holds the compression algorithms and controls the compression flow under the CompressedModel
    """

    def __init__(self):
        self.algorithms = PriorityQueue()

    def add_algorithm(self, algorithm):
        self.algorithms.append(algorithm)

    def init(self, compressed_model):
        while not self.algorithms.is_empty():
            algorithm = self.algorithms.pop()
            algorithm.apply_to(compressed_model)