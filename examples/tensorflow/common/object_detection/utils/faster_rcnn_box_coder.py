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

import tensorflow as tf

from examples.tensorflow.common.object_detection.utils import box_coder
from examples.tensorflow.common.object_detection.utils import box_list

EPSILON = 1e-8


class FasterRcnnBoxCoder(box_coder.BoxCoder):
    """Faster RCNN box coder."""

    def __init__(self, scale_factors=None):
        """Constructor for FasterRcnnBoxCoder.

        Args:
          scale_factors: List of 4 positive scalars to scale ty, tx, th and tw. If
            set to None, does not perform scaling. For Faster RCNN, the open-source
            implementation recommends using [10.0, 10.0, 5.0, 5.0].
        """
        if scale_factors:
            assert len(scale_factors) == 4
            for scalar in scale_factors:
                assert scalar > 0
        self._scale_factors = scale_factors

    @property
    def code_size(self):
        return 4

    def _encode(self, boxes, anchors):
        """Encode a box collection with respect to anchor collection.

        Args:
          boxes: BoxList holding N boxes to be encoded.
          anchors: BoxList of anchors.

        Returns:
          a tensor representing N anchor-encoded boxes of the format
          [ty, tx, th, tw].
        """
        # Convert anchors to the center coordinate representation.
        ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()
        ycenter, xcenter, h, w = boxes.get_center_coordinates_and_sizes()
        # Avoid NaN in division and log below.
        ha += EPSILON
        wa += EPSILON
        h += EPSILON
        w += EPSILON

        tx = (xcenter - xcenter_a) / wa
        ty = (ycenter - ycenter_a) / ha
        tw = tf.math.log(w / wa)
        th = tf.math.log(h / ha)
        # Scales location targets as used in paper for joint training.
        if self._scale_factors:
            ty *= self._scale_factors[0]
            tx *= self._scale_factors[1]
            th *= self._scale_factors[2]
            tw *= self._scale_factors[3]

        return tf.transpose(a=tf.stack([ty, tx, th, tw]))

    def _decode(self, rel_codes, anchors):
        """Decode relative codes to boxes.

        Args:
          rel_codes: a tensor representing N anchor-encoded boxes.
          anchors: BoxList of anchors.

        Returns:
          boxes: BoxList holding N bounding boxes.
        """
        ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()

        ty, tx, th, tw = tf.unstack(tf.transpose(a=rel_codes))
        if self._scale_factors:
            ty /= self._scale_factors[0]
            tx /= self._scale_factors[1]
            th /= self._scale_factors[2]
            tw /= self._scale_factors[3]
        w = tf.exp(tw) * wa
        h = tf.exp(th) * ha
        ycenter = ty * ha + ycenter_a
        xcenter = tx * wa + xcenter_a
        ymin = ycenter - h / 2.0
        xmin = xcenter - w / 2.0
        ymax = ycenter + h / 2.0
        xmax = xcenter + w / 2.0

        return box_list.BoxList(tf.transpose(a=tf.stack([ymin, xmin, ymax, xmax])))
