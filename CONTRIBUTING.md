# Contributing to NNCF

Contributions are accepted in the form of:

* Submitting issues against the current code to report bugs or request features
* Extending NNCF functionality with important features (e.g. to address community requests, improve usability, implement a recently published compression algorithm, etc.)
* Adding example scripts to showcase NNCF usage in real training pipelines and provide the means to reproduce the reported compression results
* Providing recipes (specific NNCF configurations and training hyperparameters) to obtain state-of-the-art compression using NNCF for existing models
* Adding well-defined patches that integrate NNCF into third-party repositories
* Reducing performance overhead of NNCF compression by writing specialized CUDA kernels for compression operations or improving existing ones.

The latter forms are accepted as pull requests from your own forks of the NNCF repository.

Any contributions must not violate the repository's [LICENSE](./LICENSE) requirements.

## Testing

After your pull request is submitted, the maintainer will launch a scope of CI tests against it.
The scope will depend on the impact of the commit, but will at least be equal to pre-commit scope checks.

The pre-commit scope may be run locally by executing the `pytest` command (without any additional arguments) in the root repository folder.
Please run the pre-commit testing scope locally before submitting your PR and ensure that it passes to conserve your own time and that of the reviewing maintainer.

New feature pull requests should include all the necessary testing code.
Testing is done using the `pytest` framework.
The test files should be located inside the [tests](./tests) directory and start with `test_` so that the `pytest` is able to discover them.
Any additional data that is required for tests (configuration files, mock datasets, etc.) must be stored within the `tests/<framework>/data` folder.
The test files themselves may be grouped in arbitrary directories according to their testing purpose and common sense.

Any additional tests in the [tests](./tests) directory will be automatically added into the pre-commit CI scope.
If your testing code is more extensive than unit tests (in terms of test execution time), or would be more suited to be executed on a nightly/weekly basis instead of for each future commit, please inform the maintainers in your PR discussion thread so that our internal testing pipelines could be adjusted accordingly.

## Code style

Changes to NNCF Python code should conform to [Python Style Guide](./docs/styleguide/PyGuide.md)

Basic code style and static checks are enforced using a `pre-commit` Github action.
The exact checks that are run are described in the corresponding [config file](./.pre-commit-config.yaml).
The checks can be run locally using `make pre-commit`.
Most of these checks do in-place fixes at the same time they are run using `make pre-commit`.
Static analysis is done via `ruff` which does not do in-place fixes by default - run `ruff --fix` locally to autofix errors for which it is possible to do so.
Developers may use other tools locally (such as `pylint`) as long as no tool-specific control code is introduced and the pre-commit checks pass.

## Binary files

Please refrain from adding huge binary files into the repository. If binary files have to be added, mark these to use Git LFS via the [.gitattributes](./.gitattributes) file.

## Copyright

1) If code was borrowed, ensure the license is checked and that no legal risks exist. For questions, contact valentina.kats@intel.com.
2) The next steps depend on the license but usually include:
    * Adding the license text to the [third-party-programs.txt](licensing/third-party-programs.txt) file.
    * Mentioning the company name and license type in the borrowed source file:

      ```python
      #
      # MIT License
      # Copyright (c) 2020 EleutherAI
      ```

    * Adding a note about where the functionality came from, for example:

      ```python
      # Modification Notes: This file was taken as is from lm-eval-harness:
      # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/wikitext/preprocess_wikitext.py
      ```

## AI Policy

Please review and follow the [AI Policy](./AI_POLICY.md) when using AI-generated code in your contributions to NNCF.
