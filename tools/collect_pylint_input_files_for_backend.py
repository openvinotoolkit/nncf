import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def is_in_backend_directory(file_path: str, backend: str):
    return re.search(f".*{os.sep}{backend}{os.sep}.*", file_path)


def is_file_name_starts_with_backend_name(file_path: str, backend: str):
    return os.path.basename(file_path).startswith(backend)


PYTHON_FILES_EXT = ".py"
BACKENDS = ["torch", "tensorflow", "openvino", "onnx"]
COMMON_BACKEND_NAME = "common"
REPO_DIR = Path(__file__).parents[1]
IGNORED_FILES = ["docs/api/source/conf.py"]
IGNORED_DIRS = ["tools"]
CHECKS_FILE_PATH_BELONGS_TO_BACKEND = [is_in_backend_directory, is_file_name_starts_with_backend_name]


def main(target_backend: str):
    if target_backend not in BACKENDS + [COMMON_BACKEND_NAME]:
        raise RuntimeError(
            f"Wrong backend passed: {target_backend}. Please choose one of available backends: {BACKENDS}"
        )

    cmd_output = subprocess.check_output("git ls-files", shell=True, cwd=REPO_DIR).decode()
    python_file_names = [
        file_path for file_path in cmd_output.split(os.linesep) if file_path.endswith(PYTHON_FILES_EXT)
    ]
    # 1) Ignore some dirs
    python_file_names = [
        file_path
        for file_path in python_file_names
        if not any(file_path.startswith(dir_name) for dir_name in IGNORED_DIRS)
    ]

    # 2) Ignore some files
    for ignored_path in IGNORED_FILES:
        python_file_names.remove(ignored_path)

    # 3) Filter files by backend
    backend_files_map = defaultdict(list)
    for file_path in python_file_names:
        for backend in BACKENDS:
            if any(check(file_path, backend) for check in CHECKS_FILE_PATH_BELONGS_TO_BACKEND):
                backend_files_map[backend].append(file_path)
                break
        else:
            backend_files_map[COMMON_BACKEND_NAME].append(file_path)

    print(*backend_files_map[target_backend], sep=os.linesep)


if __name__ == "__main__":
    main(sys.argv[1])
