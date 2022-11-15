import os
from pathlib import Path

import pytest

from tests.common.helpers import PROJECT_ROOT, TEST_ROOT
from tests.torch.helpers import Command
from tests.torch.test_sanity_third_party import TestTransformers as _ThirdPartyTransformersEnv
from tests.torch.test_sanity_third_party import create_command_line

NNCF_CONFIG_PATH = Path(TEST_ROOT, 'torch', 'sparsity', 'movement', 'examples')


@pytest.fixture(scope='class')
def temp_folder(tmp_path_factory):
    root_folder = tmp_path_factory.mktemp('movement_third_party')
    folders = {"models": root_folder / 'models',
               "venv": root_folder / 'venv',
               "repo": root_folder / 'repo'}
    for folder in folders.values():
        Path(folder).mkdir(exist_ok=True, parents=True)
    return folders


class TransformersVirtualEnv(_ThirdPartyTransformersEnv):
    # Notes: This class handles the venv installation.
    # It now wraps TestTransformers, but we propose to extract the core parts
    # and create a more general TransformersVirtualEnv class that can be used
    # by others (including `test_sanity_third_party.py`).
    def __init__(self, temp_folder):
        self.VENV_PATH = str(temp_folder["venv"])
        self.VENV_ACTIVATE = str(". {}/bin/activate".format(self.VENV_PATH))
        self.PYTHON_EXECUTABLE = str("{}/bin/python".format(self.VENV_PATH))
        self.TRANSFORMERS_REPO_PATH = str(os.path.join(temp_folder["repo"], "transformers"))
        self.CUDA_VISIBLE_STRING = "export CUDA_VISIBLE_DEVICES=0;"
        self.PATH_TO_PATCH = str(os.path.join(PROJECT_ROOT, "third_party_integration", "huggingface_transformers",
                                              "0001-Modifications-for-NNCF-usage.patch"))

    def install_transformers(self, pip_cache_dir, torch_with_cuda11):
        self.test_install_trans_(pip_cache_dir, torch_with_cuda11)


@pytest.mark.usefixtures('temp_folder')
class TestMovementWithTransformers:
    @pytest.fixture(autouse=True)
    def setup(self, temp_folder):
        self.temp_folder = temp_folder
        self.env = TransformersVirtualEnv(temp_folder)

    @pytest.mark.dependency(name='install_transformers')
    def test_install_transformers_env(self, third_party, pip_cache_dir, torch_with_cuda11, temp_folder):
        if not third_party:
            pytest.skip('Skip tests of movement sparsity with patched transformers package '
                        'since `--third-party-sanity` is False.')
        self.env.install_transformers(pip_cache_dir, torch_with_cuda11)
        print(self.env.VENV_ACTIVATE)

    @pytest.mark.dependency(depends=['install_transformers'], name="glue_movement_train")
    def test_movement_glue_train(self, temp_folder):
        nncf_config = NNCF_CONFIG_PATH / 'bert_tiny_uncased_mrpc_movement.json'
        output_dir = Path(temp_folder["models"], nncf_config.stem)
        com_line = "examples/pytorch/text-classification/run_glue.py --model_name_or_path " \
                   "google/bert_uncased_L-2_H-128_A-2 --task_name mrpc --do_train " \
                   " --per_gpu_train_batch_size 4 --learning_rate 1e-4 --num_train_epochs 4 --max_seq_length 128 " \
                   " --max_train_samples 8 --output_dir {output_dir} --save_steps 200 --nncf_config {nncf_config} " \
            .format(output_dir=output_dir, nncf_config=nncf_config)
        runner = Command(create_command_line(com_line, self.env.VENV_ACTIVATE, self.env.PYTHON_EXECUTABLE,
                                             self.env.CUDA_VISIBLE_STRING), self.env.TRANSFORMERS_REPO_PATH)
        runner.run()
        assert Path(output_dir, "pytorch_model.bin").is_file()

    @pytest.mark.dependency(depends=['install_transformers', 'glue_movement_train'])
    def test_movement_glue_eval(self, temp_folder):
        nncf_config = NNCF_CONFIG_PATH / 'bert_tiny_uncased_mrpc_movement.json'
        model_dir = Path(temp_folder["models"], nncf_config.stem)
        output_dir = Path(temp_folder["models"], nncf_config.stem + 'eval')
        com_line = "examples/pytorch/text-classification/run_glue.py --model_name_or_path {model_dir}" \
                   " --task_name mrpc --do_eval " \
                   " --learning_rate 2e-5" \
                   " --max_seq_length 128 --output_dir {output_dir}" \
                   " --max_eval_samples 10" \
                   " --nncf_config {nncf_config}" \
            .format(model_dir=model_dir, output_dir=output_dir, nncf_config=nncf_config)
        runner = Command(create_command_line(com_line, self.env.VENV_ACTIVATE, self.env.PYTHON_EXECUTABLE,
                                             self.env.CUDA_VISIBLE_STRING), self.env.TRANSFORMERS_REPO_PATH)
        runner.run()
