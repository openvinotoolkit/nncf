# Fuzzing tests

1. Install the required dependencies

    ```bash
    pip install -e ../../../.. -r requirements.txt
    ```

2. Run the fuzz test

    ```bash
    python fuzz_target.py -runs=10000 1>fuzz_log.log 2>&1
    ```
