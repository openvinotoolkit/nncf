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

try:
    from accuracy_checker.annotation_converters.ms_coco import COCO_TO_VOC
    from accuracy_checker.annotation_converters.ms_coco import MSCocoSegmentationConverter
    from accuracy_checker.main import main
except ImportError:
    from openvino.tools.accuracy_checker.annotation_converters.ms_coco import COCO_TO_VOC
    from openvino.tools.accuracy_checker.annotation_converters.ms_coco import MSCocoSegmentationConverter
    from openvino.tools.accuracy_checker.main import main


class MSCocoSegmentationToVOCConverter(MSCocoSegmentationConverter):
    __provider__ = "mscoco_segmentation_to_voc"

    @staticmethod
    def _read_image_annotation(image, annotations, label_id_to_label):
        image_labels, is_crowd, segmentation_polygons = MSCocoSegmentationConverter._read_image_annotation(
            image, annotations, label_id_to_label
        )

        # Convert to VOC labels
        image_labels = MSCocoSegmentationToVOCConverter.convert_to_voc(image_labels)

        return image_labels, is_crowd, segmentation_polygons

    @staticmethod
    def convert_to_voc(image_labels):
        return [COCO_TO_VOC.get(label, 0) for label in image_labels]


if __name__ == "__main__":
    main()
