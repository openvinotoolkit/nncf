# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf

# Box coder types.
FASTER_RCNN = "faster_rcnn"
KEYPOINT = "keypoint"
MEAN_STDDEV = "mean_stddev"
SQUARE = "square"


class BoxCoder:
    """Abstract base class for box coder."""

    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def code_size(self):
        """Return the size of each code.

        This number is a constant and should agree with the output of the `encode`
        op (e.g. if rel_codes is the output of self.encode(...), then it should have
        shape [N, code_size()]).  This abstractproperty should be overridden by
        implementations.

        Returns:
          an integer constant
        """

    def encode(self, boxes, anchors):
        """Encode a box list relative to an anchor collection.

        Args:
          boxes: BoxList holding N boxes to be encoded
          anchors: BoxList of N anchors

        Returns:
          a tensor representing N relative-encoded boxes
        """
        with tf.name_scope("Encode"):
            return self._encode(boxes, anchors)

    def decode(self, rel_codes, anchors):
        """Decode boxes that are encoded relative to an anchor collection.

        Args:
          rel_codes: a tensor representing N relative-encoded boxes
          anchors: BoxList of anchors

        Returns:
          boxlist: BoxList holding N boxes encoded in the ordinary way (i.e.,
            with corners y_min, x_min, y_max, x_max)
        """
        with tf.name_scope("Decode"):
            return self._decode(rel_codes, anchors)

    @abstractmethod
    def _encode(self, boxes, anchors):
        """Method to be overriden by implementations.

        Args:
          boxes: BoxList holding N boxes to be encoded
          anchors: BoxList of N anchors

        Returns:
          a tensor representing N relative-encoded boxes
        """

    @abstractmethod
    def _decode(self, rel_codes, anchors):
        """Method to be overriden by implementations.

        Args:
          rel_codes: a tensor representing N relative-encoded boxes
          anchors: BoxList of anchors

        Returns:
          boxlist: BoxList holding N boxes encoded in the ordinary way (i.e.,
            with corners y_min, x_min, y_max, x_max)
        """


def batch_decode(encoded_boxes, box_coder, anchors):
    """Decode a batch of encoded boxes.

    This op takes a batch of encoded bounding boxes and transforms
    them to a batch of bounding boxes specified by their corners in
    the order of [y_min, x_min, y_max, x_max].

    Args:
      encoded_boxes: a float32 tensor of shape [batch_size, num_anchors,
        code_size] representing the location of the objects.
      box_coder: a BoxCoder object.
      anchors: a BoxList of anchors used to encode `encoded_boxes`.

    Returns:
      decoded_boxes: a float32 tensor of shape [batch_size, num_anchors,
        coder_size] representing the corners of the objects in the order
        of [y_min, x_min, y_max, x_max].

    Raises:
      ValueError: if batch sizes of the inputs are inconsistent, or if
      the number of anchors inferred from encoded_boxes and anchors are
      inconsistent.
    """
    encoded_boxes.get_shape().assert_has_rank(3)
    if encoded_boxes.get_shape()[1].value != anchors.num_boxes_static():
        msg = (
            "The number of anchors inferred from encoded_boxes"
            " and anchors are inconsistent: shape[1] of encoded_boxes"
            f" {encoded_boxes.get_shape()[1].value} should be equal to the number of anchors:"
            f" {anchors.num_boxes_static()}."
        )
        raise ValueError(msg)

    decoded_boxes = tf.stack([box_coder.decode(boxes, anchors).get() for boxes in tf.unstack(encoded_boxes)])

    return decoded_boxes
