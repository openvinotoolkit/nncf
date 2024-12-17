# Style Guide for Python Code

<details markdown="1">
  <summary>Table of Contents</summary>

- [1 Introduction](#s1-introduction)
- [2 Automating Code Formatting](#s2-auto-code-formatting)
- [3 Python Language Rules](#s3-python-language-rules)
  - [3.1 3rd party packages](#s3.1-3rd-party-packages)
  - [3.2 Global variables](#s3.2-global-variables)
  - [3.3 Nested/Local/Inner Classes and Functions](#s3.3-nested)
  - [3.4 Default Iterators and Operators](#s3.4-default-iterators-and-operators)
  - [3.5 Type Annotated Code](#s3.5-type-annotated-code)
  - [3.6 Files and Sockets](#s3.6-files-and-sockets)
  - [3.7 Abstract Classes](#s3.7-abstract-classes)
- [4 Python Style Rules](#s4-python-style-rules)
  - [4.1 Line length](#s4.1-line-length)
    - [4.2 Comments and Docstrings](#s4.2-comments-and-docstrings)
      - [4.2.1 Modules](#s4.2.1-modules)
      - [4.2.2 Functions and Methods](#s4.2.2-functions-and-methods)
      - [4.2.3 Classes](#s4.2.3-classes)
      - [4.2.4 Block and Inline Comments](#s4.2.4-block-and-inline-comments)
    - [4.3 Strings](#s4.3-strings)
    - [4.4 Logging](#s4.4-logging)
    - [4.5 Error Messages](#s4.5-error-messages)
    - [4.6 TODO Comments](#s4.6-todo-comments)
    - [4.7 Naming](#s4.7-naming)
      - [4.7.1 Names to Avoid](#s4.7.1-names-to-avoid)
      - [4.7.2 Naming Conventions](#s4.7.2-naming-conventions)
      - [4.7.3 Framework specific class naming](#s4.7.3-framework-specific-class-naming)
      - [4.7.4 File Naming](#s4.7.4-file-naming)
    - [4.8 Main](#s4.8-main)
- [5 API documentation rules](#s5-api-doc-rules)
- [6 Test suite coding rules](#s6-test-suite-coding-rules)
  - [6.1 Basic style](#61-basic-style)
  - [6.2 Parametrization](#62-parametrization)
  - [6.3 Folder structure](#63-folder-structure)
  - [6.4 Test runtime considerations](#64-test-runtime-considerations)
  - [6.5 BKC management](#65-bkc-management)
- [7 Security rules](#s7-security-rules)
  - [7.1 Symlinks](#71-symlinks)

</details>

<a id="s1-introduction"></a>
<a id="1-introduction"></a>
<a id="introduction"></a>

## 1 Introduction

This document gives coding conventions for the Python code comprising [Neural Network Compression Framework (NNCF)](../../README.md).

This style guide supplements the [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)
with a list of *dos and don'ts* for Python code. If no guidelines were found in this style guide then
the [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/) should be followed.

<a id="s2-auto-code-formatting"></a>
<a id="2-auto-code-formatting"></a>
<a id="auto-code-formatting"></a>

## 2 Automating Code Formatting

To maintain consistency and readability throughout the codebase, we use the [black](https://github.com/psf/black)
and [isort](https://github.com/PyCQA/isort) tools for formatting. Before committing any changes,
it's important to run a pre-commit command to ensure that the code is properly formatted.
You can use the following commands for this:

```bash
make pre-commit
```

Also recommend configuring your IDE to run Black and isort tools automatically when saving files.

Automatic code formatting is mandatory for all Python files, but you can disable it for specific cases if required:

- if you need a specialized order of importing modules;
- for large data structures for which autoformatting unnecessarily breaks into lines,
  e.g. reference data in tests, class lists or arguments for subprocess;
- for structures for which formatting helps understanding, such as matrix.

Example for 'isort':

```python
import c
# isort: off
import b
import a
```

Example for 'black':

```python
arr1 = [
  1, 0,
  0, 1,
]  # fmt: skip

# fmt: off
arr2 = [
  1, 0,
  0, 1,
]
# fmt: on
```

<a id="s3-python-language-rules"></a>
<a id="3-python-language-rules"></a>
<a id="python-language-rules"></a>

## 3 Python Language Rules

<a id="s3.1-3rd-party-packages"></a>
<a id="31-3rd-party-packages"></a>
<a id="3rd-party-packages"></a>

### 3.1 3rd party packages

Do not add new third-party dependencies unless absolutely necessary. All things being equal, give preference to built-in packages.

<a id="s3.2-global-variables"></a>
<a id="32-global-variables"></a>
<a id="global-variables"></a>

### 3.2 Global variables

Avoid global variables.

- Module-level constants are permitted and encouraged. For example: `MAX_HOLY_HANDGRENADE_COUNT = 3`. Constants must be
  named using all caps with underscores.
- If needed, globals should be declared at the module level and made internal to the module by prepending an `_` to the
  name. External access must be done through public module-level functions.

<a id="s3.3-nested"></a>
<a id="33-nested"></a>
<a id="nested-classes-functions"></a>

### 3.3 Nested/Local/Inner Classes and Functions

No need to overuse nested local functions or classes and inner classes.

- Nested local functions or classes are fine if it satisfy the following conditions:
  - The code becomes more readable and simpler.
  - Closing over a local variables.

  ```python
  # Correct:
  def make_scaling_fn(scale):
      def mul(x):
          return scale * x
      return mul
  ```

- Inner classes are fine when it creates more readable and simple code.

- Do not nest a function just to hide it from users of a module. Instead, prefix its name with an `_` at the module
  level so that it can still be accessed by tests.

  ```Python
  # Wrong:
  def avg(a, b, c):

      def sum(x, y):
          return x + y

      m = sum(a,b)
      m = sum(m,c)
      return m/3
  ```

  ```Python
  # Correct:
  def _sum(x, y):
      return x + y

  def avg(a, b, c):
      m = _sum(a,b)
      m = _sum(m,c)
      return m/3
  ```

<a id="s3.4-default-iterators-and-operators"></a>
<a id="34-default-iterators-and-operators"></a>
<a id="default-iterators-operators"></a>

### 3.4 Default Iterators and Operators

Use default iterators and operators for types that support them, like lists,
dictionaries, and files. The built-in types define iterator methods, too. Prefer
these methods to methods that return lists, except that you should not mutate a
container while iterating over it.

```python
# Correct:
for key in adict: ...
if key not in adict: ...
if obj in alist: ...
for line in afile: ...
for k, v in adict.items(): ...
```

```python
 # Wrong:
for key in adict.keys(): ...
if not adict.has_key(key): ...
for line in afile.readlines(): ...
for k, v in dict.iteritems(): ...
```

<a id="s3.5-type-annotated-code"></a>
<a id="35-type-annotated-code"></a>
<a id="type-annotated-code"></a>

### 3.5 Type Annotated Code

Code should be annotated with type hints according to
[PEP-484](https://www.python.org/dev/peps/pep-0484/), and type-check the code at
build time with a type checking tool like [mypy](http://www.mypy-lang.org/).

```python
def func(a: int) -> List[int]:
```

<a id="s3.6-files-and-sockets"></a>
<a id="36-files-and-sockets"></a>
<a id="files-and-sockets"></a>

### 3.6 Files and Sockets

Explicitly close files and sockets when done with them.

```python
with open("hello.txt") as hello_file:
    for line in hello_file:
        print(line)
```

Use `pathlib.Path` instead of `os.*` methods for handling paths.
It is preferrable to have `pathlib.Path` objects instead of `str` to represent file paths in the code logic if performance is not critical, converting these to `str` for external APIs that cannot handle `Path` objects.

<a id="s3.7-abstract-classes"></a>
<a id="37-abstract-classes"></a>
<a id="abstract-classes"></a>

### 3.7 Abstract Classes

When defining abstract classes, the following template should be used:

```python
from abc import ABC, abstractmethod

class C(ABC):
    @abstractmethod
    def my_abstract_method(self, ...):
        pass

    @classmethod
    @abstractmethod
    def my_abstract_classmethod(cls, ...):
        pass

    @staticmethod
    @abstractmethod
    def my_abstract_staticmethod(...):
        pass

    @property
    @abstractmethod
    def my_abstract_property(self):
        pass

    @my_abstract_property.setter
    @abstractmethod
    def my_abstract_property(self, val):
        pass

    @abstractmethod
    def _get_x(self):
        pass

    @abstractmethod
    def _set_x(self, val):
        pass
```

<a id="s4-python-style-rules"></a>
<a id="4-python-style-rules"></a>
<a id="python-style-rules"></a>

## 4 Python Style Rules

<a id="s4.1-line-length"></a>
<a id="41-line-length"></a>
<a id="line-length"></a>

### 4.1 Line length

Maximum line length is *120 characters*.

Explicit exceptions to the 120 character limit:

- Long import statements.
- URLs, pathnames, or long flags in comments.
- Long string module level constants not containing whitespace that would be
  inconvenient to split across lines such as URLs or pathnames.

<a id="s4.2-comments-and-docstrings"></a>
<a id="42-comments-and-docstrings"></a>
<a id="comments-and-docstrings"></a>

### 4.2 Comments and Docstrings

Be sure to use the right style for module, function, method docstrings and
inline comments.

<a id="s4.2.1-modules"></a>
<a id="421-modules"></a>
<a id="modules"></a>

#### 4.2.1 Modules

Every file should contain a license boilerplate, where [YYYY] should be replaced to current year.

```python
# Copyright (c) [YYYY] Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

<a id="s4.2.2-functions-and-methods"></a>
<a id="422-functions-and-methods"></a>
<a id="functions-and-methods"></a>

#### 4.2.2 Functions and Methods

In this section, "function" means a method, function, or generator.

A function must have a docstring, unless it meets all of the following criteria:

- not externally visible
- very short
- obvious

```python
def load_state(model: torch.nn.Module, state_dict_to_load: dict, is_resume: bool = False) -> int:
    """
    Used to load a checkpoint containing a compressed model into an NNCFNetwork object, but can
    be used for any PyTorch module as well. Will do matching of state_dict_to_load parameters to
    the model's state_dict parameters while discarding irrelevant prefixes added during wrapping
    in NNCFNetwork or DataParallel/DistributedDataParallel objects, and load the matched parameters
    from the state_dict_to_load into the model's state dict.

    :param model: The target module for the state_dict_to_load to be loaded to.
    :param state_dict_to_load: A state dict containing the parameters to be loaded into the model.
    :param is_resume: Determines the behavior when the function cannot do a successful parameter
        match when loading. If True, the function will raise an exception if it cannot match
        the state_dict_to_load parameters to the model's parameters (i.e. if some parameters required
        by model are missing in state_dict_to_load, or if state_dict_to_load has parameters that
        could not be matched to model parameters, or if the shape of parameters is not matching).
        If False, the exception won't be raised. Usually is_resume is specified as False when loading
        uncompressed model's weights into the model with compression algorithms already applied, and
        as True when loading a compressed model's weights into the model with compression algorithms
        applied to evaluate the model.
    :return: The number of state_dict_to_load entries successfully matched and loaded into model.
    """
```

<a id="s4.2.3-classes"></a>
<a id="423-classes"></a>
<a id="classes"></a>

#### 4.2.3 Classes

Classes should have a docstring below the class definition describing the class. If your class
has public attributes, they should be documented here follow the same formatting as a function's
params section.

```python
class ModelTransformer:
    """
    Applies transformations to the model.

    :param public_attribute: Public attribute description
    """

    def __init__(self, model: TModel, transformation_layout: TransformationLayout):
        """
        :param model: The model to be transformed
        :param transformation_layout: An instance of `TransformationLayout` that
            includes a list of transformations to be applied to the model.
        """
        self._model = model
        self._transformations = transformation_layout.transformations
        self.public_attribute = None

    def transform(self) -> TModel:
        """
        Applies transformations to the model.

        :return: The transformed model
        """
        raise NotImplementedError()
```

The `__init__` function and other magic methods in non-API classes may be left without a textual description,
if there is nothing special about this exact implementation of the magic method
(i.e. the function has no notable side effects, the implementation is done in a conventional way such as
hashing all fields as a tuple in `__hash__` or concatenating string-like objects in `__add__` etc.)

For instance, this simple `__init__` method may omit the method description in the docstring (the parameter description is, however, still required):

```python
class Klass:
    # ...
    def __init__(self, param1: int, param2: float):
        """
        :param param1: Description of param1
        :param param2: Description of param2
        """
        self.param1 = param1
        self.param2 = param2
```

while this `__init__` requires a description of external dependencies and potential side effects of creating objects of the class:

```python
class ComplexKlass(BaseClass):
    # ...
    def __init__(self, param1: ParamType, param2: AnotherParamType):
        """
        *Add a brief explanation of what happens during this particular __init__, such as :*
        The construction of this object is dependent on the value of GLOBAL_VARIABLE...
        Each object of the class after __init__ is registered in ...
        Each instantiation of an object of this class leads to a side effect in ... (explain side effect)
        If *this* and *that*, the object creation will fail with a RuntimeError.
        *... and other noteworthy stuff happening in this method.*

        :param param1: Description of param1
        :param param2: Description of param2

        :raises RuntimeError if *this* and *that*
        """
        super().__init__(param1)
        self.public_param = get_public_param_value_from_global_variable(param1, GLOBAL_VARIABLE)
        result = perform_complex_calculations_with_param1_and_param2(param1, param2)
        if result == CONSTANT_VALUE_1:
            self.another_public_param = self._do_one_thing()
        elif result == CONSTANT_VALUE_2:
            self.another_public_param = self._do_other_thing()
        else:
            raise RuntimeError()
        call_function_with_side_effects()  # such as registering this instance somewhere, or acquiring a resource, etc.
        # ... potentially more code which is not understandable at a glance
```

<a id="s4.2.4-block-and-inline-comments"></a>
<a id="424-block-and-inline-comments"></a>
<a id="block-and-inline-comments"></a>

#### 4.2.4 Block and Inline Comments

The final place to have comments is in tricky parts of the code. If you're going to have to explain it
in the future, you should comment it now. Complicated operations get a few lines of comments before
the operations commence. Non-obvious ones get comments at the end of the line.

```python
# We use a weighted dictionary search to find out where i is in
# the array.  We extrapolate position based on the largest num
# in the array and the array size and then do binary search to
# get the exact number.

if i & (i-1) == 0:  # True if i is 0 or a power of 2.
```

To improve legibility, these comments should start at least 2 spaces away from
the code with the comment character `#`, followed by at least one space before
the text of the comment itself.

On the other hand, never describe the code. Assume the person reading the code
knows Python (though not what you're trying to do) better than you do.

```python
# BAD COMMENT: Now go through the b array and make sure whenever i occurs
# the next element is i+1
```

<a id="s4.3-strings"></a>
<a id="43-strings"></a>
<a id="strings"></a>

### 4.3 Strings

```python
# Correct:

long_string = """This is fine if your use case can accept
    extraneous leading spaces."""

long_string = (
    "And this too is fine if you cannot accept\n"
    "extraneous leading spaces."
)


import textwrap

long_string = textwrap.dedent(
    """
    This is also fine, because textwrap.dedent()
    will collapse common leading spaces in each line.
    """
)
```

<a id="s4.4-logging"></a>
<a id="44-logging"></a>
<a id="logging"></a>

### 4.4 Logging

Use the logger object built into NNCF for all purposes of logging within the NNCF package code.
Do not use `print(...)` or other ways of output.

Correct:

```python
from nncf.common.logging import nncf_logger

nncf_logger.info("This is an info-level log message")
```

Wrong:

```python
print("This is an info-level log message")
```

For logging functions that expect a pattern-string (with %-placeholders) as
their first argument - consider calling them with a string literal (not an f-string!)
as their first argument with pattern-parameters as subsequent arguments, if constructing the log message takes a long
time or is otherwise hurtful to performance.

```python
import nncf
from nncf.common.logging import nncf_logger

# OK:
nncf_logger.info("Test message: %s", nncf.__version__)

# Also OK:
nncf_logger.info(f"Test message: {nncf.__version__}")

# Probably not OK:
for i in range(1000000):
    nncf_logger.info(f"Test message: {sum(range(10000000))}")
```

Use proper logging levels (https://docs.python.org/3/library/logging.html#logging-levels) when printing out a message to the logger.

DEBUG - for NNCF internal information that can be utilized during debug sessions.

INFO - for good-to-know information such as progress bar or activity indicators, short summaries or effects of non-default, user-defined configuration on execution (i.e. which parts of the model were ignored due to application of "ignored_scopes" arguments).
It should be possible to safely ignore any of the INFO messages.
This level is suitable for pretty-printing, displaying the messages that guide the user or displaying other data that could be valuable for the user, but not necessarily for the developer.

WARNING - for unexpected events during execution that the user should know about, or for non-obvious effects of the current configuration.

ERROR - for reporting failures that impair the functionality without causing a fatal exception.

CRITICAL - for logging information relevant to NNCF total failures. Currently not used since this functionality is instead achieved with the text in the exceptions.

The default logging level is INFO, meaning that the user will see INFO, WARNING, ERROR and CRITICAL messages.
At these levels the logging should be as terse as possible, and the number of NNCF-created log lines should not scale with the increase in the number of operations in the model (i.e. avoid logging every simple quantizer operation being set up by NNCF - summarize the result instead)

At all log levels the log lines should not be duplicated during execution, if possible.

For deprecation warnings, use `nncf.common.logging.logger.warning_deprecated` instead of the regular `nncf_logger.warning`.
This ensures that the deprecation warning is seen to the user at all NNCF log levels.

<a id="s4.5-error-messages"></a>
<a id="45-error-messages"></a>
<a id="error-messages"></a>

### 4.5 Error Messages

Error messages (such as: message strings on exceptions like `ValueError`, or
messages shown to the user) should follow guidelines:

- The message needs to precisely match the actual error condition.
- Interpolated pieces need to always be clearly identifiable as such.
- The message should start with a capital letter.

<a id="s4.6-todo-comments"></a>
<a id="46-todo-comments"></a>
<a id="todo-comments"></a>

### 4.6 TODO Comments

Use `TODO` comments for code that is temporary, a short-term solution, or
good-enough but not perfect.

A `TODO` comment begins with the string `TODO` in all caps and a parenthesized
name, e-mail address, or another identifier
of the person or issue with the best context about the problem. This is followed
by an explanation of what there is to do.

The purpose is to have a consistent `TODO` format that can be searched to find
out how to get more details. A `TODO` is not a commitment that the person
referenced will fix the problem. Thus when you create a
`TODO`, it is almost always your name
that is given.

```python
# TODO(kl@gmail.com): Use a "*" here for string repetition.
# TODO(Zeke) Change this to use relations.
```

If your `TODO` is of the form "At a future date do something" make sure that you
either include a very specific date ("Fix by November 2030") or a very specific
event ("Remove this code when all clients can handle XML responses.").

<a id="s4.7-naming"></a>
<a id="47-naming"></a>
<a id="naming"></a>

### 4.7 Naming

`module_name`, `package_name`, `ClassName`, `method_name`, `ExceptionName`,
`function_name`, `GLOBAL_CONSTANT_NAME`, `global_var_name`, `instance_var_name`,
`function_parameter_name`, `local_var_name`.

Function names, variable names, and filenames should be descriptive; eschew
abbreviation. In particular, do not use abbreviations that are ambiguous or
unfamiliar to readers outside your project, and do not abbreviate by deleting
letters within a word.

Always use a `.py` filename extension. Never use dashes.

<table rules="all" border="1" summary="Guidelines from Guido's Recommendations"
       cellspacing="2" cellpadding="2">

  <tr>
    <th>Type</th>
    <th>Public</th>
    <th>Internal</th>
  </tr>

  <tr>
    <td>Packages</td>
    <td><code>lower_with_under</code></td>
    <td></td>
  </tr>

  <tr>
    <td>Modules</td>
    <td><code>lower_with_under</code></td>
    <td><code>_lower_with_under</code></td>
  </tr>

  <tr>
    <td>Classes</td>
    <td><code>CapWords</code></td>
    <td><code>_CapWords</code></td>
  </tr>

  <tr>
    <td>Exceptions</td>
    <td><code>CapWords</code></td>
    <td></td>
  </tr>

  <tr>
    <td>Functions</td>
    <td><code>lower_with_under()</code></td>
    <td><code>_lower_with_under()</code></td>
  </tr>

  <tr>
    <td>Global/Class Constants</td>
    <td><code>CAPS_WITH_UNDER</code></td>
    <td><code>_CAPS_WITH_UNDER</code></td>
  </tr>

  <tr>
    <td>Global/Class Variables</td>
    <td><code>lower_with_under</code></td>
    <td><code>_lower_with_under</code></td>
  </tr>

  <tr>
    <td>Instance Variables</td>
    <td><code>lower_with_under</code></td>
    <td><code>_lower_with_under</code> (protected)</td>
  </tr>

  <tr>
    <td>Method Names</td>
    <td><code>lower_with_under()</code></td>
    <td><code>_lower_with_under()</code> (protected)</td>
  </tr>

  <tr>
    <td>Function/Method Parameters</td>
    <td><code>lower_with_under</code></td>
    <td></td>
  </tr>

  <tr>
    <td>Local Variables</td>
    <td><code>lower_with_under</code></td>
    <td></td>
  </tr>

</table>

<a id="s4.7.1-names-to-avoid"></a>
<a id="471-names-to-avoid"></a>
<a id="names-to-avoid"></a>

#### 4.7.1 Names to Avoid

- single character names, except for specifically allowed cases:
  - counters or iterators (e.g. `i`, `j`, `k`, `v`, et al.)
  - `e` as an exception identifier in `try/except` statements.
  - `f` as a file handle in `with` statements
  Please be mindful not to abuse single-character naming. Generally speaking,
  descriptiveness should be proportional to the name's scope of visibility.
  For example, `i` might be a fine name for 5-line code block but within
  multiple nested scopes, it is likely too vague.
- dashes (`-`) in any package/module name
- `__double_leading_and_trailing_underscore__` names (reserved by Python)
- offensive terms
- names that needlessly include the type of the variable (for example:
    `id_to_name_dict`)

<a id="s4.7.2-naming-conventions"></a>
<a id="472-naming-convention"></a>
<a id="naming-conventions"></a>

#### 4.7.2 Naming Conventions

- "Internal" means internal to a module, or protected or private within a
   class.
- Prepending a single underscore (`_`) has some support for protecting module
  variables and functions (linters will flag protected member access). While
  prepending a double underscore (`__` aka "dunder") to an instance variable
  or method effectively makes the variable or method private to its class
  (using name mangling); we discourage its use as it impacts readability and
  testability, and isn't *really* private.
- Place related classes and top-level functions together in a
  module.
- Use CapWords for class names, but lower\_with\_under.py for module names.
- Use the word "layer" (instead of "module") in the `nncf.common` module to
  refer to the building block of neural networks.

<a id="s4.7.3-framework-specific-class-naming"></a>
<a id="473-framework-specific-class-naming"></a>
<a id="framework-specific-class-naming"></a>

#### 4.7.3 Framework specific class naming

- `PTClassName` for Torch
- `TFClassName` for TensorFlow

<a id="s4.7.4-file-naming"></a>
<a id="474-file-naming"></a>
<a id="file-naming"></a>

#### 4.7.4 File Naming

Python filenames must have a `.py` extension and must not contain dashes (`-`).
This allows them to be imported and unit tested.

Avoid having `.py` files with names such as `utils`, `helpers` that are a "swiss army knife" containing many unrelated pieces of code used across the code base.
Instead group your new code in dedicated files/modules that are named explicitly according to the purpose of code.

Bad:

*utils.py*

```python3
def log_current_time(log_stream: LogStream):
    ...

def convert_checkpoint(ckpt: CheckpointType) -> AnotherCheckpointType:
    ...
```

Good:

*logger.py*

```python3
def log_current_time(log_stream: LogStream):
    ...
```

*checkpointing/converter.py*

```python3
class CheckpointConverter:
    # ...
    def convert(self, ckpt: CheckpointType) -> AnotherCheckpointType:
        pass
```

<a id="s4.8-main"></a>
<a id="4.8-main"></a>
<a id="main"></a>

### 4.8 Main

```python
def main():
    ...

if __name__ == "__main__":
    main()
```

<a id="s5-api-doc-rules"></a>
<a id="5-api-doc-rules"></a>
<a id="api-doc-rules"></a>

## 5 API documentation rules

All functions and classes that belong to NNCF API should be documented.
The documentation should utilize the reStructuredText (.rst) format for specifying parameters, return types and otherwise formatting the docstring, since the docstring is used as a source for generating the HTML API documentation with Sphinx.

Argument descriptions for `__init__(...)` methods of API classes should be located in the docstring of the class itself, not the docstring of the `__init__(...)` method.
This is required so that the autogenerated API documentation is rendered properly.

If the autogenerated API documentation does not show type hints for certain arguments despite the fact that the type hints are present in the object's implementation code,
or if the type hints do not refer to the API symbol's canonical alias, then the type hint should be explicitly declared in the docstring using the `:type *param_name*:` directive (or `:rtype:` for return types).

<a id="s6-test-suite-coding-rules"></a>
<a id="6-test-suite-coding-rules"></a>
<a id="test-suite-coding-rules"></a>

## 6 Test suite coding rules

Unit tests are written and executed using the [pytest](https://pytest.org) framework.

Do not test the functionality of third-party libraries.

<a id="s61-basic-style"></a>
<a id="61-basic-style"></a>
<a id="basic-style"></a>

### 6.1 Basic style

Parameters of test cases and fixtures should be type-hinted just like any other function.
Test cases should ideally all have docstrings, except for the simple ones which are sufficiently described either by their names or by the simplicity of the code.

For purposes of setting up test environment for a given case, or reusing setup/teardown code, or parametrizing complex cases the [fixture](https://docs.pytest.org/en/7.4.x/explanation/fixtures.html#about-fixtures) approach should be used.
[Fixture scoping](https://docs.pytest.org/en/7.4.x/how-to/fixtures.html#scope-sharing-fixtures-across-classes-modules-packages-or-session) should be leveraged depending on the use case - for instance, a class-scoped fixture can be utilized to provide a storage object for data gathered in individual cases inside the class, and to post-process this data after the test cases in the class have all been executed.
The fixtures that are meant to be reused in multiple test files should be defined in a corresponding `conftest.py` file, instead of importing these where they are needed from another non-`conftest.py` file where they were defined - otherwise the linters will try to remove it as an unused import and also the import itself may have an effect on how the fixture is executed based on its scope.

The xunit-style setup and teardown functions ([described here](https://docs.pytest.org/en/7.4.x/how-to/xunit_setup.html#xunitsetup)) are disallowed.
Xunit-style testing seems to be outdated ("old", as stated in pytest docs) since, as it seems, everything that is accomplished using xunit-style methods can be done via fixtures with [associated benefits](https://docs.pytest.org/en/7.4.x/explanation/fixtures.html#improvements-over-xunit-style-setup-teardown-functions).

<a id="s62-parametrization"></a>
<a id="62-parametrization"></a>
<a id="parametrization"></a>

### 6.2 Parametrization

Test cases may be parametrized either using a `@pytest.mark.parametrize` decorator (for simple parametrizations), or by using [parametrized fixtures](https://docs.pytest.org/en/7.4.x/how-to/fixtures.html#parametrizing-fixtures) in case the test requires complex parametrization, or in case the parameters have to be preprocessed in a complex fashion before being passed into the test.
The parameter set ("test structs") for each test case should be represented as a collection of type-hinted class objects (or dataclass objects).
Do not use tuples, or namedtuples, or dicts to represent an individual parameter - these cannot be typed or refactored effectively, and for complex cases it is hard to visually tell by looking at the definiton of the parameter (test struct) just what each sub-parameter means.
A set of IDs should be defined manually or by using an `idfn` for each case defined by the structs so that it is easier to distinguish visually between the test cases.
When instantiating a test struct object, specify init arguments explicitly as keywords for increased visibility.

Bad:

```python3
@pytest.mark.parametrize("param", [(42, "foo", None), (1337, "bar", RuntimeError)])
def test_case_parametrized(param):
    assert function_call() == param[0]
    assert another_object.attribute != param[1]
    if param[2] is not None:
        with pytest.raises(param[2]):
            should_raise()
```

Good:

```python3
@dataclass
class StructForTest:
    expected_return_value: int
    prohibited_attribute_value: str
    raises: Optional[Type[Exception]]


def idfn(x: StructForTest) -> str:
    return f"{x.expected_return_value}-{x.prohibited_attribute_value}-{x.raises}"


@pytest.mark.parametrize(
    "ts",
    [
        StructForTest(
            expected_return_value=42, prohibited_attribute_value="foo", raises=None
        ),
        StructForTest(
            expected_return_value=1337,
            prohibited_attribute_value="bar",
            raises=RuntimeError,
        ),
    ],
    idfn=idfn,
)
def test_case_parametrized(ts: StructForTest):
    assert function_call() == ts.expected_return_value
    assert another_object.attribute != ts.prohibited_attribute_value
    if ts.raises is not None:
        with pytest.raises(ts.raises):
            should_raise()
```

<a id="s63-folder-structure"></a>
<a id="63-folder-structure"></a>
<a id="folder-structure"></a>

### 6.3 Folder structure

Test files should be grouped into directories based on what is being tested, and in the scope of which backend.
Framework-specific tests go into framework-specific directories respectively.
Tests that check only common code functionality without backend-specific dependencies go into the `common` folder.
Test cases that reuse the same testing code for multiple frameworks or test templates should be located in the `cross_fw` folder.
Auxiliary code (non-test-case) that could be reused across the tests should be stored in the `shared` folder.

<a id="s64-test-runtime-considerations"></a>
<a id="64-test-runtime-considerations"></a>
<a id="test-runtime-considerations"></a>

### 6.4 Test runtime considerations

Test cases should strive to have a low runtime while still testing what they were meant to test.
Use mocking approach (with `unittest.mock` or `pytest-mock` functionality) to simulate behaviour of heavy/compute-intensive functionality (especially external) to save runtime.
If a test is templated to be run for many backends, and if the common tested code has a significant runtime, consider passing mocks into backend-specific interfaces instead of actually running the common code, and test common code part separately.

<a id="s65-bkc-management"></a>
<a id="65-bkc-management"></a>
<a id="bkc-management"></a>

### 6.5 BKC management

Python-level requirements for test runs are specified in `requirements.txt` files at the top levels of the corresponding test subdirectories.
The versions for each package requirement must be specified exactly.

Bad (e.g. `tests/example/requirements.txt`):

```bash
torch
```

Good:

```bash
torch==2.1.0
```

<a id="s7-security-rules"></a>
<a id="7-security-rules"></a>
<a id="security-rules"></a>

## 7 Security rules

<a id="s71-symlinks"></a>
<a id="71-symlinks"></a>
<a id="symlinks"></a>

### 7.1 Symlinks

The software attempts to access a file based on the filename, but it does not properly prevent that filename from
identifying a hard or symlinks that resolves to an unintended recourses.

Check for existence if file before opening or creating them:

- If they already exists, make sure they are neither symbolic links nor hard links, unless it is an expected requirement of the application.
- If a symlink is expected, check the target of the symlink to make sure it is pointing to an expected path before any other action.

Bad:

```python
with open(file_path) as f:
    loaded_json = json.load(f)
```

Good:

```python
from nncf.common.utils.os import safe_open
...
with safe_open(file_path) as f:
    loaded_json = json.load(f)
```

```python
from nncf.common.utils.os import fail_if_symlink
...
fail_if_symlink(file_path)
function_to_save_or_read_file(file_path)
```
