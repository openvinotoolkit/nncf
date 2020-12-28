"""
 Copyright (c) 2020 Intel Corporation
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

from texttable import Texttable


def convert_raw_to_printable(raw_sparsity_statistics):
    sparsity_statistics = {}
    sparsity_statistics.update(raw_sparsity_statistics)

    table = Texttable()
    header = ['Name', 'Weight\'s Shape', 'SR', '% weights']
    data = [header]

    for sparsity_info in raw_sparsity_statistics['sparsity_statistic_by_module']:
        row = [sparsity_info[h] for h in header]
        data.append(row)
    table.add_rows(data)
    sparsity_statistics['sparsity_statistic_by_module'] = table
    return sparsity_statistics


def prepare_for_tensorboard(raw_sparsity_statistics):
    sparsity_statistics = {}
    base_prefix = '2.compression/statistics/'
    detailed_prefix = '3.compression_details/statistics/'
    for key, value in raw_sparsity_statistics.items():
        if key == 'sparsity_statistic_by_module':
            for v in value:
                sparsity_statistics[detailed_prefix + v['Name'] + '/sparsity'] = v['SR']
        else:
            sparsity_statistics[base_prefix + key] = value

    return sparsity_statistics
