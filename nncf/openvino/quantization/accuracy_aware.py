"""
 Copyright (c) 2023 Intel Corporation
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
from functools import partial

import numpy as np

from openvino.tools import pot


# TODO(andrey-churkin): Should be removed after OpenVINO release.
# pylint: disable=E1101
class NMSEBasedAccuracyAware(pot.AccuracyAwareCommon):
    """
    NMSE based implementation of the Accuracy Aware algorithm from the POT.

    This implementation works the same way as implementation from the POT in
    case when it is possible to calculate the metric for one instance.
    When it is not possible NMSE metric is used to select ranking subset and
    calculate node importance.
    """

    def __init__(self, config, engine):
        super().__init__(config, engine)
        if not engine.use_original_metric:
            self._metrics_config['original_metric'].persample.clear()
            self._metrics_config['original_metric'].persample.update(
                {
                    "name": 'nmse',
                    "type": 'nmse',
                    "is_special": True,
                    "comparator": lambda a: -a,
                    "sort_fn": partial(
                        pot.algorithms.quantization.accuracy_aware_common.utils.sort_by_logit_distance,
                        distance='nmse')
                }
            )
            self._metrics_config['original_metric'].ranking.clear()
            self._metrics_config['original_metric'].ranking.update(
                {
                    "name": "nmse",
                    "type": "nmse",
                    "is_special": True,
                    "comparator": lambda a: -a,
                    "sort_fn": partial(pot.algorithms.quantization.accuracy_aware_common.utils.sort_by_logit_distance,
                                       distance='nmse')
                }
            )

    def _get_score(self, model, ranking_subset: List[int], metric_name: str) -> float:
        if self._engine.use_original_metric:
            score = super()._get_score(model, ranking_subset, metric_name)
        else:
            ranking_metric = self._metrics_config[metric_name].ranking
            original_outputs = [
                (i, self._original_per_sample_metrics[ranking_metric.name][i]) for i in ranking_subset
            ]
            _, original_outputs = zip(*sorted(original_outputs, key=lambda x: x[0]))

            self._engine.set_model(model)
            per_sample_metrics = self._engine.calculate_per_sample_metrics(ranking_subset)
            quantized_outputs = [y['result'] for y in per_sample_metrics]

            nmse_distance = lambda u, v: np.dot(u - v, u - v) / np.dot(u, u)
            distance_between_samples = [
                nmse_distance(ui.flatten(), vi.flatten()) for ui, vi in zip(original_outputs, quantized_outputs)
            ]
            score = np.mean(distance_between_samples).item()
            score = ranking_metric.comparator(score)
        return score

    def _calculate_per_sample_metrics(self, model, subset_indices):
        self._engine.set_model(model)
        return self._engine.calculate_per_sample_metrics(subset_indices)
