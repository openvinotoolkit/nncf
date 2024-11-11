# Test examples

## Manual trigger

To manual run job use GitHub  workflow:

https://github.com/openvinotoolkit/nncf/actions/workflows/examples.yml

Parameters:

    - `pull_request_number`: The pull request number.
    - `pytest_args`: Additional pytest arguments (example `-k llm`)

## Parallel test

To set up parallel testing between jobs using `pytest-split` with the option `--splitting-algorithm=least_duration`,
you need to ensure that each test's duration is tracked correctly and stored so that future test runs can use that data
to split the tests efficiently. After run workflow get time frm artifact of job and
add new test to [.test_durations](.test_durations).
