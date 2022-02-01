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
import os
import os.path
import sys
from typing import Optional, Callable

from pathlib import Path  # from python 3.4 we may not do pip install pathlib
from pathlib import PurePath  # from python 3.4 we may not do pip install pathlib

from torch.utils import data
from torchvision import datasets

if sys.version_info[0] == 2:
    import defusedxml.cElementTree as ET
else:
    import defusedxml.ElementTree as ET

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))


class VOCAnnotationTransform:
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Args:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Args:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx],
            coordinates are normalized
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            res += [{'bbox': bndbox, 'label_idx': label_idx, 'difficult': difficult}]

        return res

class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Args:
        root (string): path to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """
    classes = VOC_CLASSES
    name = 'voc'

    def __init__(self,
                 root: str,
                 image_sets: tuple = (('2007', 'trainval'), ('2012', 'trainval')),
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = VOCAnnotationTransform(keep_difficult=False),
                 return_image_info: bool = False,
                 ):
        super().__init__()
        self.list_image_set = []
        self.target_tranform = target_transform
        self.return_image_info = return_image_info
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = []
        self.ids_new = []

        for year, name in image_sets:
            voc_elem = datasets.VOCDetection(root, year=year,
                                             image_set=name, transform=transform, target_transform=target_transform)
            for name_path in voc_elem.images:
                self.ids_new.append((str(PurePath(name_path).parents[1]), Path(name_path).stem))
            self.list_image_set.append(voc_elem)
        self._voc_concat = data.ConcatDataset(self.list_image_set)

        for (year, name) in image_sets:
            rootpath = os.path.join(root, 'VOCdevkit', 'VOC' + year)
            with open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt'), encoding='utf8') as lines:
                for line in lines:
                    self.ids.append((rootpath, line.strip()))


    def __getitem__(self, index):
        return self._voc_concat.__getitem__(index)
        # """
        # Returns image at index in torch tensor form (RGB) and
        # corresponding normalized annotation in 2d array [[xmin, ymin, xmax, ymax, label_ind],
        #                                       ... ]
        # """
        # im, gt, h, w = self.pull_item(index)
        #
        # if self.return_image_info:
        #     return im, gt, h, w
        # return im, gt

    def __len__(self):
        return self._voc_concat.__len__()


    # def pull_item(self, index):
    #     """
    #     Returns image at index in torch tensor form (RGB),
    #     corresponding normalized annotation in 2d array [[xmin, ymin, xmax, ymax, label_ind],
    #                                                      ... ],
    #     height and width of image
    #     """
    #     img_id = self.ids[index]
    #
    #     target = ET.parse(self._annopath % img_id).getroot()
    #     img = cv2.imread(self._imgpath % img_id)
    #
    #     height, width, _ = img.shape
    #
    #     if self.target_transform is not None:
    #         target = self.target_transform(target, width, height)
    #
    #     if self.transform is not None:
    #         boxes = np.asarray([x['bbox'] for x in target])
    #         labels = np.asarray([x['label_idx'] for x in target])
    #         img, boxes, labels = self.transform(img, boxes, labels)
    #         if self.rgb:
    #             img = img[:, :, (2, 1, 0)]
    #         target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
    #     return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def pull_anno(self, index): #using in load_detection_annotation
        """Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.


        Args:
            index (int): index of img to get annotation of
        Returns:
            list:  img_id, [[bbox coords, label_idx],...]]
                eg: ['001718', [[xmin, ymin, xmax, ymax, label_ind], ... ]]
        """
        img_name = self.ids[index]
        anno = ET.parse(self._annopath % img_name).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_name[1], gt

    #function not used
    # def pull_image(self, index):
    #     img_id = self.ids[index]
    #     return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def get_img_names(self):
        img_names = []
        for id_ in self.ids:
            img_names.append(id_[1])
        return img_names

a = VOCDetection('/home/nmeshalk/Desktop/data/trainval', (('2007', 'test'),)).ids
b = VOCDetection('/home/nmeshalk/Desktop/data/trainval', (('2007', 'test'),)).ids_new
print(a)
