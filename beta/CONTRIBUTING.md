# Contributing to NNCF TF

Contributions are accepted in the form of:
* Submitting issues against the current code to report bugs or request features.
* Extending NNCF TF functionality with important features (e.g. to address community requests, improve usability, implement a recently published compression algorithm, etc.).
* Adding example scripts to showcase NNCF TF usage in real training pipelines and provide the means to reproduce the reported compression results.
* Providing recipes (specific NNCF TF configurations and training hyperparameters) to obtain state-of-the-art compression using NNCF TF for existing models.
* Adding well-defined patches that integrate NNCF TF into third-party repositories.
* Reducing performance overhead of NNCF TF compression by writing specialized HW-specific kernels for compression operations or improving existing ones. 

The latter forms are accepted as pull requests from your own forks of the NNCF TF repository.

Any contributions must not violate the repository's [LICENSE](./LICENSE) requirements.

## Testing

After your pull request is submitted, the maintainer will launch a scope of CI tests against it.
The scope will depend on the impact of the commit, but will at least be equal to pre-commit scope checks.

The pre-commit scope may be run locally by executing the `pytest` command (without any additional arguments) in the root repository folder.
Please run the pre-commit testing scope locally before submitting your PR and ensure that it passes to save your time and that of the reviewing maintainer.

New feature pull requests should include all the necessary testing code. Testing is done using the `pytest` framework.
The test files should be located inside the [tests](./tests) directory and start with `test_` so that the `pytest` is able to discover them.
Any additional data that is required for tests (configuration files, mock datasets, etc.) must be stored within the [tests/data](./tests/data) folder.
The test files themselves may be grouped in arbitrary directories according to their testing purpose and common sense.

Any additional tests in the [tests](./tests) directory will be automatically added into the pre-commit CI scope.
If your testing code is more extensive than unit tests (in terms of test execution time), or would be more suited to be executed on a nightly/weekly basis instead of for each future commit, please inform the maintainers in your PR discussion thread so that our internal testing pipelines could be adjusted accordingly.


## Code style
Pylint is used throughout the project to ensure code cleanliness and quality.
A Pylint run is also done as part of the pre-commit scope - the pre-commit `pytest` scope will not be run if your code fails the Pylint checks.
The Pylint rules and exceptions for this repository are described in the standard [.pylintrc](./.pylintrc) format - make sure your local linter uses these.

## Binary files
Please refrain from adding huge binary files into the repository. If binary files have to be added, mark these to use Git LFS via the [.gitattributes](./.gitattributes) file.
