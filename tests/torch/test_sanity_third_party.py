# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import subprocess
import sys

import pytest

from nncf.torch import BKC_TORCH_VERSION
from tests.shared.paths import PROJECT_ROOT
from tests.torch.helpers import Command

TRANSFORMERS_COMMIT = "bd469c40659ce76c81f69c7726759d249b4aef49"
INSTALL_PATH = PROJECT_ROOT.parent
DATASET_PATH = os.path.join(PROJECT_ROOT, "tests", "torch", "data", "mock_datasets")


def create_command_line(args, venv_activate, python=sys.executable, cuda_string=""):
    line = "{venv_activate} && {cuda} {python_exe} {args}".format(
        venv_activate=venv_activate, cuda=cuda_string, args=args, python_exe=python
    )
    return line


@pytest.fixture(autouse=True, scope="session")
def skip_tests(third_party):
    if not third_party:
        pytest.skip()


@pytest.fixture(scope="session")
def temp_folder(tmp_path_factory):
    return {
        "models": str(tmp_path_factory.mktemp("models", False)),
        "venv": str(tmp_path_factory.mktemp("venv", False)),
        "repo": str(tmp_path_factory.mktemp("repo", False)),
    }


class CachedPipRunner:
    def __init__(self, venv_activation_script_path: str, cache_dir: str = None):
        self.venv_activate = venv_activation_script_path
        self.cache_dir = cache_dir

    def run_pip(self, pip_command: str, cwd: str = None, use_cache: bool = True):
        if not use_cache:
            cache_dir_entry = "--no-cache-dir"
        elif self.cache_dir is not None:
            cache_dir_entry = "--cache-dir {}".format(self.cache_dir)
        else:
            cache_dir_entry = ""
        subprocess.run(f"{self.venv_activate} && pip {cache_dir_entry} {pip_command}", check=True, shell=True, cwd=cwd)


class TransformersVirtualEnvInstaller:
    def __init__(self, venv_path, repo_path):
        self.VENV_PATH = str(venv_path)
        self.VENV_ACTIVATE = str(". {}/bin/activate".format(self.VENV_PATH))
        self.PYTHON_EXECUTABLE = str("{}/bin/python".format(self.VENV_PATH))
        self.TRANSFORMERS_REPO_PATH = str(os.path.join(repo_path, "transformers"))
        self.CUDA_VISIBLE_STRING = "export CUDA_VISIBLE_DEVICES=0;"
        self.PATH_TO_PATCH = str(
            os.path.join(
                PROJECT_ROOT,
                "third_party_integration",
                "huggingface_transformers",
                "0001-Modifications-for-NNCF-usage.patch",
            )
        )

    def install_env(self, pip_cache_dir):
        version_string = "{}.{}".format(sys.version_info[0], sys.version_info[1])
        subprocess.call("virtualenv -ppython{} {}".format(version_string, self.VENV_PATH), shell=True)
        pip_runner = CachedPipRunner(self.VENV_ACTIVATE, pip_cache_dir)
        pip_runner.run_pip("install --upgrade pip")  # cache options are available with pip > 20.2
        pip_runner.run_pip("uninstall setuptools -y")
        pip_runner.run_pip("install setuptools")
        pip_runner.run_pip("install onnx")
        torch_install_cmd = "install torch=={}".format(BKC_TORCH_VERSION)
        pip_runner.run_pip(torch_install_cmd)
        subprocess.run(
            "git clone https://github.com/huggingface/transformers {}".format(self.TRANSFORMERS_REPO_PATH),
            check=True,
            shell=True,
        )
        subprocess.run(
            "git checkout {}".format(TRANSFORMERS_COMMIT), check=True, shell=True, cwd=self.TRANSFORMERS_REPO_PATH
        )
        subprocess.run("cp {} .".format(self.PATH_TO_PATCH), check=True, shell=True, cwd=self.TRANSFORMERS_REPO_PATH)
        subprocess.run(
            "git apply 0001-Modifications-for-NNCF-usage.patch", check=True, shell=True, cwd=self.TRANSFORMERS_REPO_PATH
        )
        pip_runner.run_pip("install .", cwd=self.TRANSFORMERS_REPO_PATH)
        pip_runner.run_pip('install -e ".[testing]"', cwd=self.TRANSFORMERS_REPO_PATH)
        for sample_folder in ["question-answering", "text-classification", "language-modeling", "token-classification"]:
            pip_runner.run_pip(
                f"install -r examples/pytorch/{sample_folder}/requirements.txt", cwd=self.TRANSFORMERS_REPO_PATH
            )
        pip_runner.run_pip("install boto3", cwd=self.TRANSFORMERS_REPO_PATH)
        # WA for deleted CONLL2003 in datasets==1.11.0 (https://github.com/huggingface/datasets/issues/3582)
        pip_runner.run_pip("install -U datasets", cwd=self.TRANSFORMERS_REPO_PATH)
        pip_runner.run_pip("install -e .", cwd=PROJECT_ROOT)


class TestTransformers:
    @pytest.fixture(autouse=True)
    def setup(self, temp_folder):
        self.env = TransformersVirtualEnvInstaller(temp_folder["venv"], temp_folder["repo"])

    @pytest.mark.dependency(name="install_trans")
    def test_install_trans_(self, pip_cache_dir):
        self.env.install_env(pip_cache_dir)

    @pytest.mark.dependency(depends=["install_trans"], name="xnli_train")
    def test_xnli_train(self, temp_folder):
        com_line = (
            "examples/pytorch/text-classification/run_xnli.py --model_name_or_path bert-base-chinese"
            " --language zh --train_language zh --do_train --per_gpu_train_batch_size 24"
            " --learning_rate 5e-5 --num_train_epochs 0.0001 --max_seq_length 128 --output_dir {}"
            " --save_steps 200 --nncf_config nncf_bert_config_xnli.json".format(
                os.path.join(temp_folder["models"], "xnli")
            )
        )
        runner = Command(
            create_command_line(
                com_line, self.env.VENV_ACTIVATE, self.env.PYTHON_EXECUTABLE, self.env.CUDA_VISIBLE_STRING
            ),
            self.env.TRANSFORMERS_REPO_PATH,
        )
        runner.run()
        assert os.path.exists(os.path.join(temp_folder["models"], "xnli", "pytorch_model.bin"))

    @pytest.mark.dependency(depends=["install_trans", "xnli_train"])
    def test_xnli_eval(self, temp_folder):
        com_line = (
            "examples/pytorch/text-classification/run_xnli.py --model_name_or_path {output}"
            " --language zh --do_eval --learning_rate 5e-5 --max_seq_length 128 --output_dir"
            " {output} --nncf_config nncf_bert_config_xnli.json --per_gpu_eval_batch_size 24"
            " --max_eval_samples 10".format(output=os.path.join(temp_folder["models"], "xnli"))
        )
        runner = Command(
            create_command_line(
                com_line, self.env.VENV_ACTIVATE, self.env.PYTHON_EXECUTABLE, self.env.CUDA_VISIBLE_STRING
            ),
            self.env.TRANSFORMERS_REPO_PATH,
        )
        runner.run()

    @pytest.mark.dependency(depends=["install_trans"], name="squad_train")
    def test_squad_train(self, temp_folder):
        com_line = (
            "examples/pytorch/question-answering/run_qa.py --model_name_or_path "
            "bert-large-uncased-whole-word-masking-finetuned-squad --dataset_name squad --do_train "
            " --learning_rate 3e-5 --num_train_epochs 0.0001 --max_seq_length 384 --doc_stride 128 "
            " --output_dir {} --per_gpu_train_batch_size=1 --save_steps=200 --nncf_config"
            " nncf_bert_config_squad.json".format(os.path.join(temp_folder["models"], "squad"))
        )
        runner = Command(
            create_command_line(
                com_line, self.env.VENV_ACTIVATE, self.env.PYTHON_EXECUTABLE, self.env.CUDA_VISIBLE_STRING
            ),
            self.env.TRANSFORMERS_REPO_PATH,
        )
        runner.run()
        assert os.path.exists(os.path.join(temp_folder["models"], "squad", "pytorch_model.bin"))

    @pytest.mark.dependency(depends=["install_trans", "squad_train"])
    def test_squad_eval(self, temp_folder):
        com_line = (
            "examples/pytorch/question-answering/run_qa.py --model_name_or_path {output}"
            " --do_eval --dataset_name squad  --learning_rate 3e-5"
            " --max_seq_length 384 --doc_stride 128 --per_gpu_eval_batch_size=4 --output_dir {output} "
            " --max_eval_samples 10"
            " --nncf_config nncf_bert_config_squad.json".format(output=os.path.join(temp_folder["models"], "squad"))
        )
        runner = Command(
            create_command_line(
                com_line, self.env.VENV_ACTIVATE, self.env.PYTHON_EXECUTABLE, self.env.CUDA_VISIBLE_STRING
            ),
            self.env.TRANSFORMERS_REPO_PATH,
        )
        runner.run()

    @pytest.mark.dependency(depends=["install_trans"], name="glue_roberta_train")
    def test_glue_train(self, temp_folder):
        com_line = (
            "examples/pytorch/text-classification/run_glue.py --model_name_or_path"
            " roberta-large-mnli --task_name mnli --do_train "
            " --per_gpu_train_batch_size 4 --learning_rate 2e-5 --num_train_epochs 0.001 --max_seq_length 128 "
            " --output_dir {} --save_steps 200 --nncf_config"
            " nncf_roberta_config_mnli.json".format(os.path.join(temp_folder["models"], "roberta_mnli"))
        )
        runner = Command(
            create_command_line(
                com_line, self.env.VENV_ACTIVATE, self.env.PYTHON_EXECUTABLE, self.env.CUDA_VISIBLE_STRING
            ),
            self.env.TRANSFORMERS_REPO_PATH,
        )
        runner.run()
        assert os.path.exists(os.path.join(temp_folder["models"], "roberta_mnli", "pytorch_model.bin"))

    @pytest.mark.dependency(depends=["install_trans", "glue_roberta_train"])
    def test_glue_eval(self, temp_folder):
        com_line = (
            "examples/pytorch/text-classification/run_glue.py --model_name_or_path {output}"
            " --task_name mnli --do_eval --validation_file {}/glue/glue_data/MNLI/dev_matched.tsv "
            " --learning_rate 2e-5"
            " --max_seq_length 128 --output_dir {output}"
            " --max_eval_samples 10"
            " --nncf_config nncf_roberta_config_mnli.json".format(
                DATASET_PATH, output=os.path.join(temp_folder["models"], "roberta_mnli")
            )
        )
        runner = Command(
            create_command_line(
                com_line, self.env.VENV_ACTIVATE, self.env.PYTHON_EXECUTABLE, self.env.CUDA_VISIBLE_STRING
            ),
            self.env.TRANSFORMERS_REPO_PATH,
        )
        runner.run()

    @pytest.mark.dependency(depends=["install_trans"], name="glue_distilbert_train")
    def test_glue_distilbert_train(self, temp_folder):
        com_line = (
            "examples/pytorch/text-classification/run_glue.py --model_name_or_path"
            " distilbert-base-uncased --train_file {}/glue/glue_data/SST-2/train.tsv"
            " --task_name sst2 --do_train --max_seq_length 128 --per_gpu_train_batch_size 8"
            " --learning_rate 5e-5 --num_train_epochs 0.001"
            " --output_dir {} --save_steps 200 --nncf_config"
            " nncf_distilbert_config_sst2.json".format(
                DATASET_PATH, os.path.join(temp_folder["models"], "distilbert_output")
            )
        )
        runner = Command(
            create_command_line(
                com_line, self.env.VENV_ACTIVATE, self.env.PYTHON_EXECUTABLE, self.env.CUDA_VISIBLE_STRING
            ),
            self.env.TRANSFORMERS_REPO_PATH,
        )
        runner.run()
        assert os.path.exists(os.path.join(temp_folder["models"], "distilbert_output", "pytorch_model.bin"))

    @pytest.mark.dependency(depends=["install_trans", "glue_distilbert_train"])
    def test_glue_distilbert_eval(self, temp_folder):
        com_line = (
            "examples/pytorch/text-classification/run_glue.py --model_name_or_path {output}"
            " --task_name sst2 --do_eval --max_seq_length 128"
            " --output_dir {output} --validation_file {}/glue/glue_data/SST-2/test.tsv"
            " --max_eval_samples 10"
            " --nncf_config nncf_distilbert_config_sst2.json".format(
                DATASET_PATH, output=os.path.join(temp_folder["models"], "distilbert_output")
            )
        )
        runner = Command(
            create_command_line(
                com_line, self.env.VENV_ACTIVATE, self.env.PYTHON_EXECUTABLE, self.env.CUDA_VISIBLE_STRING
            ),
            self.env.TRANSFORMERS_REPO_PATH,
        )
        runner.run()

    @pytest.mark.dependency(depends=["install_trans"], name="lm_train")
    def test_lm_train(self, temp_folder):
        # GPT2 is loaded via torch.frombuffer which is not available in torch==1.9.1 yet
        com_line = (
            "examples/pytorch/language-modeling/run_clm.py --model_name_or_path distilgpt2"
            " --do_train --per_gpu_train_batch_size 1"
            " --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 "
            " --num_train_epochs 0.001"
            " --output_dir {} --nncf_config"
            " nncf_gpt2_config_wikitext_hw_config.json".format(os.path.join(temp_folder["models"], "lm_output"))
        )
        runner = Command(
            create_command_line(
                com_line, self.env.VENV_ACTIVATE, self.env.PYTHON_EXECUTABLE, self.env.CUDA_VISIBLE_STRING
            ),
            self.env.TRANSFORMERS_REPO_PATH,
        )
        runner.run()
        assert os.path.exists(os.path.join(temp_folder["models"], "lm_output", "pytorch_model.bin"))

    @pytest.mark.dependency(depends=["install_trans", "lm_train"])
    def test_lm_eval(self, temp_folder):
        # GPT2 is loaded via torch.frombuffer which is not available in torch==1.9.1 yet
        com_line = (
            "examples/pytorch/language-modeling/run_clm.py "
            " --model_name_or_path {output} --do_eval "
            " --output_dir {output} --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1"
            " --max_eval_samples 10"
            " --nncf_config nncf_gpt2_config_wikitext_hw_config.json".format(
                output=os.path.join(temp_folder["models"], "lm_output")
            )
        )
        runner = Command(
            create_command_line(
                com_line, self.env.VENV_ACTIVATE, self.env.PYTHON_EXECUTABLE, self.env.CUDA_VISIBLE_STRING
            ),
            self.env.TRANSFORMERS_REPO_PATH,
        )
        runner.run()

    @pytest.mark.dependency(depends=["install_trans"], name="ner_train")
    def test_ner_train(self, temp_folder):
        com_line = (
            "examples/pytorch/token-classification/run_ner.py --model_name_or_path bert-base-uncased"
            " --do_train --per_gpu_train_batch_size 1"
            " --dataset_name conll2003 "
            " --max_train_samples 10"
            " --output_dir {} "
            " --nncf_config nncf_bert_config_conll.json".format(os.path.join(temp_folder["models"], "ner_output"))
        )
        runner = Command(
            create_command_line(
                com_line, self.env.VENV_ACTIVATE, self.env.PYTHON_EXECUTABLE, self.env.CUDA_VISIBLE_STRING
            ),
            self.env.TRANSFORMERS_REPO_PATH,
        )
        runner.run()
        assert os.path.exists(os.path.join(temp_folder["models"], "ner_output", "pytorch_model.bin"))

    @pytest.mark.dependency(depends=["install_trans", "ner_train"])
    def test_ner_eval(self, temp_folder):
        com_line = (
            "examples/pytorch/token-classification/run_ner.py "
            " --model_name_or_path {output} --do_eval "
            " --output_dir {output} --dataset_name conll2003"
            " --max_eval_samples 10"
            " --nncf_config nncf_bert_config_conll.json".format(
                output=os.path.join(temp_folder["models"], "ner_output")
            )
        )
        runner = Command(
            create_command_line(
                com_line, self.env.VENV_ACTIVATE, self.env.PYTHON_EXECUTABLE, self.env.CUDA_VISIBLE_STRING
            ),
            self.env.TRANSFORMERS_REPO_PATH,
        )
        runner.run()

    @pytest.mark.dependency(depends=["install_trans"])
    def test_convert_to_onnx(self, temp_folder):
        com_line = (
            "examples/pytorch/question-answering/run_qa.py --model_name_or_path {output} "
            " --do_eval"
            " --dataset_name squad "
            " --max_eval_samples 10"
            " --output_dir {output}"
            " --to_onnx {output}/model.onnx"
            " --nncf_config nncf_bert_config_squad.json".format(output=os.path.join(temp_folder["models"], "squad"))
        )
        runner = Command(
            create_command_line(
                com_line, self.env.VENV_ACTIVATE, self.env.PYTHON_EXECUTABLE, self.env.CUDA_VISIBLE_STRING
            ),
            self.env.TRANSFORMERS_REPO_PATH,
        )
        runner.run()
        assert os.path.exists(os.path.join(temp_folder["models"], "squad", "model.onnx"))
