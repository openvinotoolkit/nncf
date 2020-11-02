"""
 Copyright (c) 2019-2020 Intel Corporation
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

# pylint:disable=no-name-in-module
import os
import pytest
import sys
import subprocess
from tests.test_sanity_sample import Command
from tests.conftest import PROJECT_ROOT


TRANSFORMERS_COMMIT = "b0892fa0e8df02d683e05e625b3903209bff362d"
MMDETECTION_COMMIT = "039ad4dd64edaa5efe69f00574a0c24240adac97"
INSTALL_PATH = PROJECT_ROOT.parent
DATASET_PATH = os.path.join(PROJECT_ROOT, "tests", "data", "mock_datasets")


def create_command_line(args, venv, python=sys.executable, cuda_string=""):
    python_path = PROJECT_ROOT.as_posix()
    line = "PYTHONPATH={path} {venv_activate}; {cuda} {python_exe} {args}"\
        .format(path=python_path, venv_activate=venv, cuda=cuda_string, args=args, python_exe=python)
    return line


@pytest.fixture(autouse=True, scope="session")
def skip_tests(third_party):
    if not third_party:
        pytest.skip()


@pytest.fixture(scope="session")
def temp_folder(tmp_path_factory):
    return {"models": str(tmp_path_factory.mktemp("models", False)),
            "venv": str(tmp_path_factory.mktemp("venv", False))}


# pylint:disable=redefined-outer-name
class TestTransformers:
    @pytest.fixture(autouse=True)
    def setup(self, temp_folder):
        self.VENV_TRANS_PATH = str(os.path.join(temp_folder["venv"], "trans"))
        self.cuda_visible_string = "export CUDA_VISIBLE_DEVICES=0;"
        self.PATH_TO_PATCH = str(os.path.join(PROJECT_ROOT, "third_party_integration", "huggingface_transformers",
                                              "0001-Modifications-for-NNCF-usage.patch"))
        self.trans_python = str("{}/bin/python".format(self.VENV_TRANS_PATH))
        self.TRANS_PATH = str(os.path.join(self.VENV_TRANS_PATH, "transformers"))
        self.activate_venv = str(". {}/bin/activate".format(self.VENV_TRANS_PATH))

    def test_install_trans_(self):
        subprocess.call("virtualenv -ppython3.6 {}".format(self.VENV_TRANS_PATH), shell=True)
        subprocess.run("{} pip uninstall setuptools -y && pip install setuptools".format(self.activate_venv),
                       check=True, shell=True)
        subprocess.run("{} && pip install torch==1.5.0".format(self.activate_venv),
                       check=True, shell=True)
        subprocess.run("{} && git clone https://github.com/huggingface/transformers".format(self.activate_venv),
                       check=True, shell=True, cwd=self.VENV_TRANS_PATH)
        subprocess.run("{} && git checkout {}".format(self.activate_venv, TRANSFORMERS_COMMIT), check=True, shell=True,
                       cwd=self.TRANS_PATH)
        subprocess.run("{} && cp {} .".format(self.activate_venv, self.PATH_TO_PATCH), check=True, shell=True,
                       cwd=self.TRANS_PATH)
        subprocess.run("{} && git apply 0001-Modifications-for-NNCF-usage.patch".format(self.activate_venv),
                       check=True, shell=True, cwd=self.TRANS_PATH)
        subprocess.run("{} && pip install .".format(self.activate_venv), check=True, shell=True, cwd=self.TRANS_PATH)
        subprocess.run("{} && pip install -e \".[testing]\"".format(self.activate_venv), check=True, shell=True,
                       cwd=self.TRANS_PATH)
        subprocess.run("{} && pip install -r examples/requirements.txt".format(self.activate_venv), check=True,
                       shell=True, cwd=self.TRANS_PATH)
        subprocess.run("{} && pip install boto3".format(self.activate_venv), check=True, shell=True,
                       cwd=self.TRANS_PATH)
        subprocess.run(
            "{} && {}/bin/python setup.py develop".format(self.activate_venv, self.VENV_TRANS_PATH), check=True,
            shell=True, cwd=PROJECT_ROOT)

    def test_xnli_train(self, temp_folder):
        com_line = "examples/text-classification/run_xnli.py --model_name_or_path bert-base-chinese" \
                   " --language zh --train_language zh --do_train --data_dir {} --per_gpu_train_batch_size 24" \
                   " --learning_rate 5e-5 --num_train_epochs 1.0 --max_seq_length 128 --output_dir {}" \
                   " --save_steps 200 --nncf_config nncf_bert_config_xnli.json"\
            .format(DATASET_PATH, os.path.join(temp_folder["models"], "xnli"))
        runner = Command(create_command_line(com_line, self.VENV_TRANS_PATH, self.trans_python,
                                             self.cuda_visible_string), self.TRANS_PATH)
        res = runner.run()
        assert res == 0
        assert os.path.exists(os.path.join(temp_folder["models"], "xnli", "pytorch_model.bin"))

    def test_xnli_eval(self, temp_folder):
        com_line = "examples/text-classification/run_xnli.py --model_name_or_path {output}" \
                   " --language zh --do_eval --data_dir {} --learning_rate 5e-5 --max_seq_length 128 --output_dir" \
                   " {output} --nncf_config nncf_bert_config_xnli.json --per_gpu_eval_batch_size 24"\
            .format(DATASET_PATH, output=os.path.join(temp_folder["models"], "xnli"))
        runner = Command(create_command_line(com_line, self.activate_venv, self.trans_python,
                                             self.cuda_visible_string), self.TRANS_PATH)
        res = runner.run()
        assert res == 0

    def test_squad_train(self, temp_folder):
        com_line = "examples/question-answering/run_squad.py --model_type bert --model_name_or_path " \
                   "bert-large-uncased-whole-word-masking-finetuned-squad --do_train --do_lower_case " \
                   "--train_file {}/squad/train-v1.1.json" \
                   " --learning_rate 3e-5 --num_train_epochs 1 --max_seq_length 384 --doc_stride 128 --output_dir " \
                   "{} --per_gpu_train_batch_size=1 --save_steps=200 --nncf_config" \
                   " nncf_bert_config_squad.json".format(DATASET_PATH, os.path.join(temp_folder["models"], "squad"))
        runner = Command(create_command_line(com_line, self.activate_venv, self.trans_python,
                                             self.cuda_visible_string), self.TRANS_PATH)
        res = runner.run()
        assert res == 0
        assert os.path.exists(os.path.join(temp_folder["models"], "squad", "pytorch_model.bin"))

    def test_squad_eval(self, temp_folder):
        com_line = "examples/question-answering/run_squad.py --model_type bert --model_name_or_path {output}" \
                   " --do_eval --do_lower_case  --predict_file {}/squad/dev-v1.1.json --learning_rate 3e-5" \
                   " --max_seq_length 384 --doc_stride 128 --per_gpu_eval_batch_size=4 --output_dir {output} " \
                   "--nncf_config nncf_bert_config_squad.json"\
            .format(DATASET_PATH, output=os.path.join(temp_folder["models"], "squad"))
        runner = Command(create_command_line(com_line, self.activate_venv, self.trans_python,
                                             self.cuda_visible_string), self.TRANS_PATH)
        res = runner.run()
        assert res == 0

    def test_glue_train(self, temp_folder):
        com_line = "examples/text-classification/run_glue.py --model_name_or_path" \
                   " roberta-large-mnli --task_name mnli --do_train --data_dir {}/glue/glue_data/MNLI" \
                   " --per_gpu_train_batch_size 4 --learning_rate 2e-5 --num_train_epochs 1.0 --max_seq_length 128 " \
                   "--output_dir {} --save_steps 200 --nncf_config" \
                   " nncf_roberta_config_mnli.json"\
            .format(DATASET_PATH, os.path.join(temp_folder["models"], "roberta_mnli"))
        runner = Command(create_command_line(com_line, self.activate_venv, self.trans_python,
                                             self.cuda_visible_string), self.TRANS_PATH)
        res = runner.run()
        assert res == 0
        assert os.path.exists(os.path.join(temp_folder["models"], "roberta_mnli", "pytorch_model.bin"))

    def test_glue_eval(self, temp_folder):
        com_line = "examples/text-classification/run_glue.py --model_name_or_path {output}" \
                   " --task_name mnli --do_eval --data_dir {}/glue/glue_data/MNLI --learning_rate 2e-5" \
                   " --num_train_epochs 1.0 --max_seq_length 128 --output_dir {output}" \
                   " --nncf_config nncf_roberta_config_mnli.json"\
            .format(DATASET_PATH, output=os.path.join(temp_folder["models"], "roberta_mnli"))
        runner = Command(create_command_line(com_line, self.activate_venv, self.trans_python,
                                             self.cuda_visible_string), self.TRANS_PATH)
        res = runner.run()
        assert res == 0

    def test_glue_distilbert_train(self, temp_folder):
        com_line = "examples/text-classification/run_glue.py --model_name_or_path" \
                   " distilbert-base-uncased" \
                   " --task_name SST-2 --do_train --max_seq_length 128 --per_gpu_train_batch_size 8" \
                   " --data_dir {}/glue/glue_data/SST-2 --learning_rate 5e-5 --num_train_epochs 3.0" \
                   " --output_dir {} --save_steps 200 --nncf_config" \
                   " nncf_distilbert_config_sst2.json".format(DATASET_PATH, os.path.join(temp_folder["models"],
                                                                                         "distilbert_output"))
        runner = Command(create_command_line(com_line, self.activate_venv, self.trans_python,
                                             self.cuda_visible_string), self.TRANS_PATH)
        res = runner.run()
        assert res == 0
        assert os.path.exists(os.path.join(temp_folder["models"], "distilbert_output", "pytorch_model.bin"))

    def test_glue_distilbert_eval(self, temp_folder):
        com_line = "examples/text-classification/run_glue.py --model_name_or_path {output}" \
                   " --task_name SST-2 --do_eval --max_seq_length 128" \
                   " --output_dir {output} --data_dir {}/glue/glue_data/SST-2" \
                   " --nncf_config nncf_distilbert_config_sst2.json"\
            .format(DATASET_PATH, output=os.path.join(temp_folder["models"], "distilbert_output"))
        runner = Command(create_command_line(com_line, self.activate_venv, self.trans_python,
                                             self.cuda_visible_string), self.TRANS_PATH)
        res = runner.run()
        assert res == 0

    def test_lm_train(self, temp_folder):
        com_line = "examples/language-modeling/run_language_modeling.py --model_type gpt2 --model_name_or_path gpt2" \
                   " --do_train --per_gpu_train_batch_size 8" \
                   " --train_data_file {}/wikitext-2-raw/wiki.train.raw " \
                   " --output_dir {} --nncf_config" \
                   " nncf_gpt2_config_wikitext_hw_config.json".format(DATASET_PATH, os.path.join(temp_folder["models"],
                                                                                                 "lm_output"))
        runner = Command(create_command_line(com_line, self.activate_venv, self.trans_python,
                                             self.cuda_visible_string), self.TRANS_PATH)
        res = runner.run()
        assert res == 0
        assert os.path.exists(os.path.join(temp_folder["models"], "lm_output", "pytorch_model.bin"))

    def test_lm_eval(self, temp_folder):
        com_line = "examples/language-modeling/run_language_modeling.py --model_type gpt2 " \
                   "--model_name_or_path {output} --do_eval " \
                   " --output_dir {output} --eval_data_file {}/wikitext-2-raw/wiki.train.raw" \
                   " --nncf_config nncf_gpt2_config_wikitext_hw_config.json" \
            .format(DATASET_PATH, output=os.path.join(temp_folder["models"], "lm_output"))
        runner = Command(create_command_line(com_line, self.activate_venv, self.trans_python,
                                             self.cuda_visible_string), self.TRANS_PATH)
        res = runner.run()
        assert res == 0

    def test_convert_to_onnx(self, temp_folder):
        com_line = "examples/question-answering/run_squad.py --model_type bert --model_name_or_path {output}" \
                   " --output_dir {output}" \
                   " --to_onnx {output}/model.onnx".format(output=os.path.join(temp_folder["models"], "squad"))
        runner = Command(create_command_line(com_line, self.activate_venv, self.trans_python,
                                             self.cuda_visible_string), self.TRANS_PATH)
        res = runner.run()
        assert res == 0
        assert os.path.exists(os.path.join(temp_folder["models"], "squad", "model.onnx"))


# pylint:disable=line-too-long
class TestMmdetection:
    @pytest.fixture(autouse=True)
    def setup(self, temp_folder):
        self.PATH_TO_PATCH = str(os.path.join(PROJECT_ROOT, "third_party_integration", "mmdetection",
                                              "0001-Modifications-for-NNCF-usage.patch"))
        self.VENV_MMDET_PATH = str(os.path.join(temp_folder["venv"], "mmdet"))
        self.activate_venv = str(". {}/bin/activate".format(self.VENV_MMDET_PATH))
        self.mmdet_python = str("{}/bin/python".format(self.VENV_MMDET_PATH))
        self.MMDET_PATH = str(os.path.join(self.VENV_MMDET_PATH, "mmdetection"))

    def test_install_mmdet(self):
        subprocess.call("virtualenv -ppython3.6 {}".format(self.VENV_MMDET_PATH), shell=True)
        subprocess.run(
            "{} && {}/bin/python setup.py develop".format(self.activate_venv, self.VENV_MMDET_PATH), check=True,
            shell=True, cwd=PROJECT_ROOT)
        subprocess.run("{} pip uninstall setuptools -y && pip install setuptools".format(self.activate_venv),
                       check=True, shell=True)
        subprocess.run("{}; git clone https://github.com/open-mmlab/mmdetection.git".format(self.activate_venv),
                       check=True, shell=True, cwd=self.VENV_MMDET_PATH)
        subprocess.run("{}; git checkout {}".format(self.activate_venv, MMDETECTION_COMMIT), check=True, shell=True,
                       cwd=self.MMDET_PATH)
        subprocess.run("{}; cp {} .".format(self.activate_venv, self.PATH_TO_PATCH), check=True, shell=True,
                       cwd=self.MMDET_PATH)
        subprocess.run("{}; git apply 0001-Modifications-for-NNCF-usage.patch".format(self.activate_venv),
                       check=True, shell=True, cwd=self.MMDET_PATH)
        subprocess.run(
            "{}; pip install mmcv-full==1.1.4+torch1.5.0+cu102 "
            "-f https://download.openmmlab.com/mmcv/dist/index.html".format(self.activate_venv), check=True, shell=True,
            cwd=self.MMDET_PATH)
        subprocess.run("{}; pip install -r requirements/build.txt".format(self.activate_venv), check=True, shell=True,
                       cwd=self.MMDET_PATH)
        subprocess.run("{}; pip install -v -e .".format(self.activate_venv), check=True, shell=True,
                       cwd=self.MMDET_PATH)
        subprocess.run("{}; pip install -U \"git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools\""
                       .format(self.activate_venv), check=True, shell=True, cwd=self.MMDET_PATH)

        subprocess.run("{}; mkdir {}".format(self.activate_venv, self.MMDET_PATH + "/data"), check=True, shell=True,
                       cwd=self.MMDET_PATH)
        subprocess.run("{}; ln -s {}/voc data/VOCdevkit".format(self.activate_venv, DATASET_PATH), check=True,
                       shell=True,
                       cwd=self.MMDET_PATH)
        subprocess.run("{}; ln -s {}/coco data/coco".format(self.activate_venv, DATASET_PATH), check=True,
                       shell=True,
                       cwd=self.MMDET_PATH)

    def test_ssd300_train(self):
        subprocess.run(
            "wget https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ssd300_voc_vgg16_caffe_240e_20190501-7160d09a.pth",
            check=True, shell=True, cwd=self.MMDET_PATH)
        comm_line = "tools/train.py configs/pascal_voc/ssd300_voc_int8.py"
        runner = Command(create_command_line(comm_line, self.activate_venv, self.mmdet_python), self.MMDET_PATH)
        res = runner.run()
        assert res == 0
        assert os.path.exists(os.path.join(self.MMDET_PATH, "work_dirs", "ssd300_voc_int8", "latest.pth"))

    def test_ssd300_eval(self):
        checkpoint = os.path.join(self.MMDET_PATH, "work_dirs", "ssd300_voc_int8", "latest.pth")
        comm_line = "tools/test.py configs/pascal_voc/ssd300_voc_int8.py {} --eval mAP".format(checkpoint)
        runner = Command(create_command_line(comm_line, self.activate_venv, self.mmdet_python), self.MMDET_PATH)
        res = runner.run()
        assert res == 0

    def test_retinanet_train(self):
        subprocess.run(
            "wget https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/retinanet_r50_fpn_2x_20190616-75574209.pth",
            check=True, shell=True, cwd=self.MMDET_PATH)
        comm_line = "tools/train.py configs/retinanet/retinanet_r50_fpn_1x_int8.py"
        runner = Command(create_command_line(comm_line, self.activate_venv, self.mmdet_python), self.MMDET_PATH)
        res = runner.run()
        assert res == 0
        assert os.path.exists(os.path.join(self.MMDET_PATH, "work_dirs", "retinanet_r50_fpn_1x_int8", "latest.pth"))

    def test_retinanet_eval(self):
        checkpoint = os.path.join(self.MMDET_PATH, "work_dirs", "retinanet_r50_fpn_1x_int8", "latest.pth")
        comm_line = "tools/test.py configs/retinanet/retinanet_r50_fpn_1x_int8.py {} --eval bbox".format(checkpoint)
        runner = Command(create_command_line(comm_line, self.activate_venv, self.mmdet_python), self.MMDET_PATH)
        res = runner.run()
        assert res == 0
