from pathlib import Path

import pytest

from tests.shared.paths import TEST_ROOT
from tests.torch.helpers import Command
from tests.torch.test_sanity_third_party import TransformersVirtualEnvInstaller
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


@pytest.mark.usefixtures('temp_folder')
class TestMovementWithTransformers:
    # pylint:disable=redefined-outer-name
    @pytest.fixture(autouse=True)
    def setup(self, temp_folder):
        self.temp_folder = temp_folder
        self.env = TransformersVirtualEnvInstaller(temp_folder['venv'], temp_folder['repo'])

    @pytest.mark.dependency(name='install_transformers')
    def test_install_transformers_env(self, third_party, pip_cache_dir, torch_with_cuda11):
        if not third_party:
            pytest.skip('Skip tests of movement sparsity with patched transformers package '
                        'since `--third-party-sanity` is False.')
        self.env.install_env(pip_cache_dir, torch_with_cuda11)

    @pytest.mark.dependency(depends=['install_transformers'], name="glue_movement_train")
    def test_movement_glue_train(self):
        nncf_config = NNCF_CONFIG_PATH / 'bert_tiny_uncased_mrpc_movement.json'
        output_dir = Path(self.temp_folder["models"], nncf_config.stem)
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
    def test_movement_glue_eval(self):
        nncf_config = NNCF_CONFIG_PATH / 'bert_tiny_uncased_mrpc_movement.json'
        model_dir = Path(self.temp_folder["models"], nncf_config.stem)
        output_dir = Path(self.temp_folder["models"], nncf_config.stem + '_eval')
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
