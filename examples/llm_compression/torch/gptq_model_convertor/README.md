# Convert compressed model to GPTQModel format

> [!WARNING]
> It's experimental feature with limitations. The API and functionality may change in future releases.

This example demonstrates how to replace compressed linear modules to GPTQ modules.

## Prerequisites

Before running this example, ensure you have Python 3.10+ installed and set up your environment:

### 1. Create and activate a virtual environment

```bash
python3 -m venv nncf_env
source nncf_env/bin/activate  # On Windows: nncf_env\Scripts\activate.bat
```

### 2. Install NNCF and other dependencies

```bash
python3 -m pip install ../../../../ -r requirements.txt
python3 -m pip install -r requirements_extra.txt --no-build-isolation # Install gptqmodel
```

## Run Example

```bash
python main.py
```
