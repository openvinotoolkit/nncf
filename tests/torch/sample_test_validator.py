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

import json
import os
import sys
import tempfile
from abc import ABC
from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Type

import torch

from examples.torch.common.utils import get_run_name
from nncf import NNCFConfig
from nncf.common.utils.registry import Registry
from tests.cross_fw.shared.command import arg_list_from_arg_dict
from tests.cross_fw.shared.config_factory import ConfigFactory
from tests.cross_fw.shared.paths import EXAMPLES_DIR
from tests.cross_fw.shared.paths import TEST_ROOT


def create_command_line(args: Dict[str, Any], sample_type: str, main_filename: str = "main.py") -> str:
    executable = EXAMPLES_DIR.joinpath("torch", sample_type, main_filename).as_posix()
    cli_args = " ".join(key if (val is None or val is True) else "{} {}".format(key, val) for key, val in args.items())
    return f"{sys.executable} {executable} {cli_args}"


class SampleType(Enum):
    CLASSIFICATION = "classification"
    CLASSIFICATION_STAGED = "classification_staged"
    CLASSIFICATION_NAS = "classification_nas"
    CLASSIFICATION_NAS_SEARCH = "classification_nas_search"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    OBJECT_DETECTION = "object_detection"


SAMPLE_HANDLERS = Registry("sample_handlers")


class BaseSampleHandler(ABC):
    @abstractmethod
    def get_metric_value_from_checkpoint(
        self, checkpoint_save_dir: str, checkpoint_name: Optional[str] = None, config_path: Optional[Path] = None
    ):
        pass

    @abstractmethod
    def get_sample_dir_name(self) -> str:
        pass

    def get_executable(self) -> str:
        """
        :return: path to the executable main file .py
        """
        return EXAMPLES_DIR.joinpath("torch", self.get_sample_dir_name(), self._get_main_filename() + ".py")

    def get_main_location(self) -> str:
        """
        :return: path for importing main file that is an entry-point for the sample
        """
        return f"examples.torch.{self.get_sample_dir_name()}.{self._get_main_filename()}"

    def get_sample_location(self) -> str:
        """
        :return: path for importing file major sample-specific functionality. Usually it's identical to main location,
        but sometimes can be another file like for staged classification.
        """
        return self.get_main_location()

    def get_train_location(self) -> str:
        """
        :return: path for importing train function.
        """
        return self.get_sample_location() + ".train"

    @staticmethod
    def get_checkpoint_path(
        checkpoint_save_dir: str, checkpoint_name: Optional[str] = None, config_path: Optional[Path] = None
    ):
        if checkpoint_name is None:
            jconfig = NNCFConfig.from_json(config_path)
            checkpoint_name = get_run_name(jconfig)
        return os.path.join(checkpoint_save_dir, checkpoint_name + "_best.pth")

    def _get_main_filename(self) -> str:
        return "main"


@SAMPLE_HANDLERS.register(SampleType.CLASSIFICATION)
class ClassificationHandler(BaseSampleHandler):
    def get_metric_value_from_checkpoint(
        self, checkpoint_save_dir: str, checkpoint_name: Optional[str] = None, config_path: Optional[Path] = None
    ):
        checkpoint_path = self.get_checkpoint_path(checkpoint_save_dir, checkpoint_name, config_path)
        assert os.path.exists(checkpoint_path), "Path to checkpoint {} does not exist".format(checkpoint_path)
        accuracy = torch.load(checkpoint_path)["best_acc1"]
        return accuracy

    def get_sample_dir_name(self) -> str:
        return "classification"


@SAMPLE_HANDLERS.register(SampleType.CLASSIFICATION_STAGED)
class ClassificationStagedHandler(ClassificationHandler):
    def get_sample_location(self) -> str:
        return f"examples.torch.{self.get_sample_dir_name()}.staged_quantization_worker"

    def get_train_location(self):
        return self.get_sample_location() + ".train_staged"


@SAMPLE_HANDLERS.register(SampleType.CLASSIFICATION_NAS)
class ClassificationNASHandler(ClassificationHandler):
    def get_executable(self) -> str:
        return EXAMPLES_DIR.joinpath(
            "experimental", "torch", self.get_sample_dir_name(), self._get_main_filename() + ".py"
        )

    def get_main_location(self):
        return f"examples.experimental.torch.{self.get_sample_dir_name()}.{self._get_main_filename()}"

    def _get_main_filename(self):
        return "bootstrap_nas"

    def get_minimal_subnet_accuracy(self):
        pass

    def get_optimal_subnet_accuracy(self):
        pass


@SAMPLE_HANDLERS.register(SampleType.CLASSIFICATION_NAS_SEARCH)
class ClassificationNASSearchHandler(ClassificationNASHandler):
    def _get_main_filename(self):
        return "bootstrap_nas_search"


@SAMPLE_HANDLERS.register(SampleType.OBJECT_DETECTION)
class ObjectDetectionHandler(BaseSampleHandler):
    def get_metric_value_from_checkpoint(
        self, checkpoint_save_dir: str, checkpoint_name: Optional[str] = None, config_path: Optional[Path] = None
    ):
        pass

    def get_sample_dir_name(self) -> str:
        return "object_detection"


@SAMPLE_HANDLERS.register(SampleType.SEMANTIC_SEGMENTATION)
class SemanticSegmentationHandler(BaseSampleHandler):
    def get_sample_dir_name(self) -> str:
        return "semantic_segmentation"

    def get_metric_value_from_checkpoint(
        self, checkpoint_save_dir: str, checkpoint_name: Optional[str] = None, config_path: Optional[Path] = None
    ):
        pass


class BaseSampleValidator(ABC):
    @abstractmethod
    def setup_spy(self, mocker):
        pass

    @abstractmethod
    def validate_spy(self):
        pass

    @abstractmethod
    def validate_sample(self, args, mocker):
        pass

    @abstractmethod
    def get_default_args(self, tmp_path):
        pass


class BaseSampleTestCaseDescriptor(ABC):
    def __init__(self):
        self.config_name_: str = ""

        self.sample_type_: Optional[SampleType] = None
        self.sample_handler: Optional[BaseSampleHandler] = None
        self.sample_type(SampleType.CLASSIFICATION)

        self.dataset_dir: Optional[Path] = None
        self.dataset_name: str = ""
        self.batch_size_: int = 0

    @property
    @abstractmethod
    def config_directory(self) -> Path:
        pass

    @abstractmethod
    def get_validator(self) -> BaseSampleValidator:
        pass

    @property
    def config_path(self) -> Path:
        return self.config_directory.joinpath(self.config_name_)

    def config_name(self, config_name: str):
        self.config_name_ = config_name
        return self

    def batch_size(self, batch_size: int):
        self.batch_size_ = batch_size
        return self

    def sample_type(self, sample_type: SampleType):
        self.sample_type_ = sample_type
        sample_handler_cls: Type[BaseSampleHandler] = SAMPLE_HANDLERS.get(self.sample_type_)
        self.sample_handler = sample_handler_cls()
        return self

    def real_dataset(self, dataset_name: str):
        self.dataset_name = dataset_name
        return self

    def mock_dataset(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.dataset_dir = TEST_ROOT / "torch" / "data" / "mock_datasets" / dataset_name
        return self

    def __str__(self):
        return "_".join([self.config_name_, self.sample_type_.value, self.dataset_name])


class SanityTestCaseDescriptor(BaseSampleTestCaseDescriptor, ABC):
    def __init__(self):
        super().__init__()
        self.config_dict: Dict = {}

    @abstractmethod
    def get_compression_section(self):
        pass

    def get_config_update(self) -> Dict:
        sample_params = self.get_sample_params()
        return {**sample_params, "target_device": "NPU", "compression": self.get_compression_section()}

    def get_sample_params(self) -> Dict:
        return {"dataset": self.dataset_name} if self.dataset_name else {}

    def finalize(self, dataset_dir=None) -> "SanityTestCaseDescriptor":
        with self.config_path.open() as file:
            json_config = json.load(file)
            json_config.update(self.get_config_update())
            self.config_dict = json_config
        if self.dataset_dir is None:
            self.dataset_dir = Path(
                dataset_dir if dataset_dir else os.path.join(tempfile.gettempdir(), self.dataset_name)
            )
        return self


class SanitySampleValidator(BaseSampleValidator, ABC):
    def __init__(self, desc: SanityTestCaseDescriptor):
        self._desc = desc
        self._sample_handler = desc.sample_handler

    def validate_sample(self, args: Dict[str, Any], mocker):
        arg_list = arg_list_from_arg_dict(args)
        command_line = " ".join(arg_list)
        print(f"Command line arguments: {command_line}")

        import importlib

        main_location = self._sample_handler.get_main_location()
        sample = importlib.import_module(main_location)

        self.setup_spy(mocker)
        sample.main(arg_list)
        self.validate_spy()

    def get_default_args(self, tmp_path):
        config_factory = ConfigFactory(self._desc.config_dict, tmp_path / "config.json")
        args = {
            "--data": str(self._desc.dataset_dir),
            "--config": config_factory.serialize(),
            "--log-dir": tmp_path,
            "--batch-size": self._desc.batch_size_,
            "--workers": 0,  # Workaround for the PyTorch MultiProcessingDataLoader issue
        }
        if not torch.cuda.is_available():
            args["--cpu-only"] = True
        return args
