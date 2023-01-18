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
from pathlib import Path

import pytest

from tests.shared.paths import PROJECT_ROOT
from tests.torch.helpers import Command
from tests.torch.test_sanity_third_party import TransformersVirtualEnvInstaller
from tests.torch.test_sanity_third_party import create_command_line


@pytest.fixture(scope='class')
def temp_folder(tmp_path_factory):
    root_folder = tmp_path_factory.mktemp('BootstrapNAS_third_party')
    folders = {'models': root_folder / 'models',
               'venv': root_folder / 'venv',
               'repo': root_folder / 'repo'}
    for folder in folders.values():
        Path(folder).mkdir(exist_ok=True, parents=True)
    return folders


class BootstrapNASTransformersVirtualEnvInstaller(TransformersVirtualEnvInstaller):
    def __init__(self, venv_path, repo_path):
        super().__init__(venv_path, repo_path)
        self.PATH_TO_PATCH = str(os.path.join(PROJECT_ROOT, "third_party_integration", "huggingface_transformers",
                                        "BootstrapNAS_HF", "0001-Modifications-for-BootstrapNAS-usage.patch"))

    def install_env(self, pip_cache_dir, torch_with_cuda11):
        pip_runner = super().install_env(pip_cache_dir, torch_with_cuda11)
        pip_runner.run_pip("install tensorboard")


@pytest.mark.usefixtures('temp_folder')
class TestBootstrapNASWithTransformers:
    # pylint:disable=redefined-outer-name
    @pytest.fixture(autouse=True)
    def setup(self, temp_folder):
        self.temp_folder = temp_folder
        self.env = BootstrapNASTransformersVirtualEnvInstaller(temp_folder['venv'], temp_folder['repo'])

    @pytest.mark.dependency(name='install_transformers')
    def test_install_transformers_env(self, third_party, pip_cache_dir, torch_with_cuda11):
        if not third_party:
            pytest.skip('Skip tests of BootstrapNAS with patched transformers package '
                        'since `--third-party-sanity` is False.')
        self.env.install_env(pip_cache_dir, torch_with_cuda11)

    @pytest.mark.dependency(depends=['install_transformers'], name='xnli_train')
    def test_xnli_train(self, temp_folder):
        com_line = "examples/pytorch/text-classification/run_xnli.py --model_name_or_path bert-base-chinese" \
                   " --language zh --train_language zh --do_train --do_search --per_gpu_train_batch_size 24" \
                   " --max_seq_length 128 --output_dir {} --evaluation_strategy epoch --max_eval_samples 10" \
                   " --save_steps 200 --max_train_samples 10 --metric_for_best_model accuracy" \
                   " --nncf_config nncf_bootstrapnas_config/nncf_bootstrapnas_bert_config_xnli.json" \
            .format(os.path.join(temp_folder["models"], "xnli"))
        runner = Command(create_command_line(com_line, self.env.VENV_ACTIVATE, self.env.PYTHON_EXECUTABLE,
                                             self.env.CUDA_VISIBLE_STRING), self.env.TRANSFORMERS_REPO_PATH)
        runner.run()
        assert os.path.exists(os.path.join(temp_folder["models"], "xnli", "pytorch_model.bin"))
        assert os.path.exists(os.path.join(temp_folder["models"], "xnli", "search_progression.csv"))

    @pytest.mark.dependency(depends=['install_transformers', 'xnli_train'])
    def test_xnli_eval(self, temp_folder):
        com_line = "examples/pytorch/text-classification/run_xnli.py --model_name_or_path {output}" \
                   " --language zh --do_eval --max_seq_length 128 --output_dir" \
                   " {output} --nncf_config nncf_bootstrapnas_config/nncf_bootstrapnas_bert_config_xnli.json" \
                   " --per_gpu_eval_batch_size 24 --max_eval_samples 10" \
            .format(output=os.path.join(temp_folder["models"], "xnli"))
        runner = Command(create_command_line(com_line, self.env.VENV_ACTIVATE, self.env.PYTHON_EXECUTABLE,
                                             self.env.CUDA_VISIBLE_STRING), self.env.TRANSFORMERS_REPO_PATH)
        runner.run()

    @pytest.mark.dependency(depends=['install_transformers'], name='squad_train')
    def test_squad_train(self, temp_folder):
        com_line = "examples/pytorch/question-answering/run_qa.py --model_name_or_path " \
                   "bert-large-uncased-whole-word-masking-finetuned-squad --dataset_name squad --do_train" \
                   " --do_search --max_seq_length 384 --evaluation_strategy epoch --metric_for_best_model f1" \
                   " --doc_stride 128 --output_dir {} --per_gpu_train_batch_size=1" \
                   " --save_steps=200 --max_train_samples 10 --max_eval_samples 10" \
                   " --nncf_config nncf_bootstrapnas_config/nncf_bootstrapnas_bert_config_squad.json" \
            .format(os.path.join(temp_folder["models"], "squad"))
        runner = Command(create_command_line(com_line, self.env.VENV_ACTIVATE, self.env.PYTHON_EXECUTABLE,
                                             self.env.CUDA_VISIBLE_STRING), self.env.TRANSFORMERS_REPO_PATH)
        runner.run()
        assert os.path.exists(os.path.join(temp_folder["models"], "squad", "pytorch_model.bin"))
        assert os.path.exists(os.path.join(temp_folder["models"], "squad", "search_progression.csv"))

    @pytest.mark.dependency(depends=['install_transformers', 'squad_train'])
    def test_squad_eval(self, temp_folder):
        com_line = "examples/pytorch/question-answering/run_qa.py --model_name_or_path {output}" \
                   " --do_eval --dataset_name squad  --max_eval_samples 10" \
                   " --max_seq_length 384 --doc_stride 128 --per_gpu_eval_batch_size=4 --output_dir {output} " \
                   " --nncf_config nncf_bootstrapnas_config/nncf_bootstrapnas_bert_config_squad.json" \
            .format(output=os.path.join(temp_folder["models"], "squad"))
        runner = Command(create_command_line(com_line, self.env.VENV_ACTIVATE, self.env.PYTHON_EXECUTABLE,
                                             self.env.CUDA_VISIBLE_STRING), self.env.TRANSFORMERS_REPO_PATH)
        runner.run()

    @pytest.mark.dependency(depends=['install_transformers'], name='ner_train')
    def test_ner_train(self, temp_folder):
        com_line = "examples/pytorch/token-classification/run_ner.py --model_name_or_path bert-base-uncased" \
                   " --do_train --do_search --per_gpu_train_batch_size 1" \
                   " --dataset_name conll2003 --max_eval_samples 10" \
                   " --max_train_samples 10 --evaluation_strategy epoch" \
                   " --output_dir {} --metric_for_best_model accuracy" \
                   " --nncf_config nncf_bootstrapnas_config/nncf_bootstrapnas_bert_config_conll.json" \
            .format(os.path.join(temp_folder["models"], "ner_output"))
        runner = Command(create_command_line(com_line, self.env.VENV_ACTIVATE, self.env.PYTHON_EXECUTABLE,
                                             self.env.CUDA_VISIBLE_STRING), self.env.TRANSFORMERS_REPO_PATH)
        runner.run()
        assert os.path.exists(os.path.join(temp_folder["models"], "ner_output", "pytorch_model.bin"))
        assert os.path.exists(os.path.join(temp_folder["models"], "ner_output", "search_progression.csv"))

    @pytest.mark.dependency(depends=['install_transformers', 'ner_train'])
    def test_ner_eval(self, temp_folder):
        com_line = "examples/pytorch/token-classification/run_ner.py " \
                   " --model_name_or_path {output} --do_eval " \
                   " --output_dir {output} --dataset_name conll2003" \
                   " --max_eval_samples 10" \
                   " --nncf_config nncf_bootstrapnas_config/nncf_bootstrapnas_bert_config_conll.json" \
            .format(output=os.path.join(temp_folder["models"], "ner_output"))
        runner = Command(create_command_line(com_line, self.env.VENV_ACTIVATE, self.env.PYTHON_EXECUTABLE,
                                             self.env.CUDA_VISIBLE_STRING), self.env.TRANSFORMERS_REPO_PATH)
        runner.run()

    @pytest.mark.dependency(depends=['install_transformers'], name='glue_train')
    def test_glue_train(self, temp_folder):
        com_line = "examples/pytorch/text-classification/run_glue.py --model_name_or_path" \
                   " bert-base-cased-finetuned-mrpc --task_name mrpc --do_train --do_search --output_dir {}" \
                   " --per_gpu_train_batch_size 4 --evaluation_strategy epoch --max_seq_length 128 " \
                   " --max_train_samples 10 --max_eval_samples 10 --save_steps 200 --metric_for_best_model accuracy" \
                   " --nncf_config nncf_bootstrapnas_config/nncf_bootstrapnas_bert_config_mrpc.json" \
            .format(os.path.join(temp_folder["models"], "mrpc"))
        runner = Command(create_command_line(com_line, self.env.VENV_ACTIVATE, self.env.PYTHON_EXECUTABLE,
                                             self.env.CUDA_VISIBLE_STRING), self.env.TRANSFORMERS_REPO_PATH)
        runner.run()
        assert os.path.exists(os.path.join(temp_folder["models"], "mrpc", "pytorch_model.bin"))
        assert os.path.exists(os.path.join(temp_folder["models"], "mrpc", "search_progression.csv"))

    @pytest.mark.dependency(depends=['install_transformers', 'glue_train'])
    def test_glue_eval(self, temp_folder):
        com_line = "examples/pytorch/text-classification/run_glue.py --model_name_or_path {output}" \
                   " --task_name mrpc --do_eval" \
                   " --max_seq_length 128 --output_dir {output}" \
                   " --max_eval_samples 10" \
                   " --nncf_config nncf_bootstrapnas_config/nncf_bootstrapnas_bert_config_mrpc.json" \
            .format(output=os.path.join(temp_folder["models"], "mrpc"))
        runner = Command(create_command_line(com_line, self.env.VENV_ACTIVATE, self.env.PYTHON_EXECUTABLE,
                                             self.env.CUDA_VISIBLE_STRING), self.env.TRANSFORMERS_REPO_PATH)
        runner.run()

    @pytest.mark.dependency(depends=['install_transformers'])
    def test_convert_to_onnx(self, temp_folder):
        com_line = "examples/pytorch/question-answering/run_qa.py --model_name_or_path {output} " \
                   " --do_eval" \
                   " --dataset_name squad " \
                   " --max_eval_samples 10" \
                   " --output_dir {output}" \
                   " --to_onnx {output}/model.onnx" \
                   " --nncf_config nncf_bootstrapnas_config/nncf_bootstrapnas_bert_config_squad.json" \
            .format(output=os.path.join(temp_folder["models"], "squad"))
        runner = Command(create_command_line(com_line, self.env.VENV_ACTIVATE, self.env.PYTHON_EXECUTABLE,
                                             self.env.CUDA_VISIBLE_STRING), self.env.TRANSFORMERS_REPO_PATH)
        runner.run()
        assert os.path.exists(os.path.join(temp_folder["models"], "squad", "model.onnx"))
