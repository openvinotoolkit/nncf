# LLM Weight Compression Tool

## Install

```bash
python3.10 -m venv env
. env/bin/activate
pip install --upgrade pip

pip install openvino==2025.0.0
pip install nncf==2.15.0
pip install "git+https://github.com/huggingface/optimum.git@v1.24.0"
pip install git+https://github.com/huggingface/optimum-intel.git@v1.22.0

# #whowhatbench
git clone --depth 1 --branch 2025.0.0.0 https://github.com/openvinotoolkit/openvino.genai.git

cd openvino.genai/tools/who_what_benchmark
pip install .
```

```bash
# For test
python run.py \
--model-id facebook/opt-125m \
--config config_optimum_cli.json \
--root-dir experiment_dir \
--dump-packages
```