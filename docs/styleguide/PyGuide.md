# Style Guide for Python Code

<details markdown="1">
  <summary>Table of Contents</summary>

-   [1 Introduction](#s1-introduction)
-   [2 Python Language Rules](#s2-python-language-rules)
    *   [2.1 PyLint](#s2.1-pylint)
    *   [2.2 Imports](#s2.2-imports)
    *   [2.3 3rd party packages](#s2.3-3rd-party-packages)
    *   [2.4 Global variables](#s2.4-global-variables)
    *   [2.5 Nested/Local/Inner Classes and Functions](#s2.5-nested)
    *   [2.6 Default Iterators and Operators](#s2.6-default-iterators-and-operators)
    *   [2.7 Type Annotated Code](#s2.7-type-annotated-code)
    *   [2.8 Files and Sockets](#2.8-files-and-sockets)
    *   [2.9 Abstract Classes](#2.9-abstract-classes)
-   [3 Python Style Rules](#s3-python-style-rules)
    *   [3.1 Line length](#s3.1-line-length)
    *   [3.2 Indentation](#s3.2-indentation)
    *   [3.3 Blank Lines](#s3.3-blank-lines)
    *   [3.4 Whitespace](#s3.4-whitespace)
    *   [3.5 Comments and Docstrings](#s3.5-comments-and-docstrings)
        +   [3.5.1 Modules](#s3.5.1-modules)
        +   [3.5.2 Functions and Methods](#s3.5.2-functions-and-methods)
        +   [3.5.3 Classes](#s3.5.3-classes)
        +   [3.5.4 Block and Inline Comments](#s3.5.4-block-and-inline-comments)
    *   [3.6 Strings](#s3.6-strings)
    *   [3.7 Logging](#s3.7-logging) 
    *   [3.8 Error Messages](#s3.8-error-messages)
    *   [3.9 TODO Comments](#s3.9-todo-comments)
    *   [3.10 Naming](#s3.10-naming)
        +   [3.10.1 Names to Avoid](#s3.10.1-names-to-avoid)
        +   [3.10.2 Naming Conventions](#s3.10.2-naming-conventions)
        +   [3.10.3 Framework specific class naming](#s3.10.3-framework-specific-class-naming)
        +   [3.10.4 File Naming](#s3.10.4-file-naming)
    *   [3.11 Main](#s3.11-main)
</details>

<a id="s1-introduction"></a>
<a id="1-introduction"></a>
<a id="introduction"></a>
## 1 Introduction 

This document gives coding conventions for the Python code comprising [Neural Network Compression Framework (NNCF)](../../README.md). 

This style guide supplements the [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)
with a list of *dos and don'ts* for Python code. If no guidelines were found in this style guide then 
the [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/) should be followed.   

<a id="s2-python-language-rules"></a>
<a id="2-python-language-rules"></a>
<a id="python-language-rules"></a>
## 2 Python Language Rules 

<a id="s2.1-pylint"></a>
<a id="21-pylint"></a>
<a id="pylint"></a>
### 2.1 PyLint 

Run [pylint](https://github.com/PyCQA/pylint) over your code using this [pylintrc](../../.pylintrc).

- Every warning reported by [pylint](https://github.com/PyCQA/pylint) must be resolved in one of the following way:
  - *Preferred solution*: Change the code to fix the warning.
  - *Exception*: Suppress the warning if they are inappropriate so that other issues are not hidden. 
    To suppress warnings you can set a line-level comment
    ```python
    dict = 'something awful'  # Bad Idea... pylint: disable=redefined-builtin
    ```
    or update [pylintrc](../../.pylintrc) if applicable for the whole project. If the reason for the suppression 
    is not clear from the symbolic name, add an explanation.
      

<a id="s2.2-imports"></a>
<a id="22-imports"></a>
<a id="imports"></a>
### 2.2 Imports 

- Use absolute imports, as they are usually more readable and tend to be better behaved
  ```python
  # Correct:
  import mypkg.sibling
  from mypkg import sibling
  from mypkg.sibling import example
  ```
  ```python
  # Wrong:
  from . import sibling
  from .sibling import example
  ```
- Imports should usually be on separate lines:
  ```python
  # Correct:
  from nncf.api.compression import CompressionLoss
  from nncf.api.compression import CompressionScheduler
  from nncf.api.compression import CompressionAlgorithmBuilder
  from nncf.api.compression import CompressionAlgorithmController
  ```
  ```python
  # Wrong:
  from nncf.api.compression import CompressionLoss, CompressionScheduler, CompressionAlgorithmBuilder, \
      CompressionAlgorithmController
  ```
- Imports are always put at the top of the file, just after any module comments and docstrings, and before module 
  globals and constants. Imports should be grouped in the following order:
  - Standard library imports.
  - Related third party imports. 
  - Local application/library specific imports.
  
  You should put a blank line between each group of imports.
- When importing a class from a class-containing module, the preferred behavior is as follows:
  ```python
  from myclass import MyClass
  from foo.bar.yourclass import YourClass
  ```
  If this spelling causes local name clashes, then spell them explicitly:
  ```python
  import myclass
  import foo.bar.yourclass
  ```
  and use `myclass.MyClass` and `foo.bar.yourclass.YourClass`.

- Wildcard imports (`from module import *`) should be avoided, as they make it unclear which names are present 
  in the namespace, confusing both readers and many automated tools.

- For classes from the typing module. You are explicitly allowed to import multiple specific classes on one line from the typing module.
  ```python
  # Recommended:
  from typing import Any, Dict, Optional
  ```
  ```python
  # Try to avoid, but this is also applicable:
  from typing import Any 
  from typing import Dict
  from typing import Optional
  ```

<a id="s2.3-3rd-party-packages"></a>
<a id="23-3rd-party-packages"></a>
<a id="3rd-party-packages"></a>
### 2.3 3rd party packages 

Do not add new third-party dependencies unless absolutely necessary. All things being equal, give preference to built-in packages.  

<a id="s2.4-global-variables"></a>
<a id="24-global-variables"></a>
<a id="global-variables"></a>
### 2.4 Global variables 

Avoid global variables.

- Module-level constants are permitted and encouraged. For example: `MAX_HOLY_HANDGRENADE_COUNT = 3`. Constants must be 
  named using all caps with underscores.
- If needed, globals should be declared at the module level and made internal to the module by prepending an `_` to the 
  name. External access must be done through public module-level functions.

<a id="s2.5-nested"></a>
<a id="25-nested"></a>
<a id="nested-classes-functions"></a>
### 2.5 Nested/Local/Inner Classes and Functions 

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

<a id="s2.6-default-iterators-and-operators"></a>
<a id="26-default-iterators-and-operators"></a>
<a id="default-iterators-operators"></a>
### 2.6 Default Iterators and Operators 

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

<a id="s2.7-type-annotated-code"></a>
<a id="27-type-annotated-code"></a>
<a id="type-annotated-code"></a>
### 2.7 Type Annotated Code 

Code should be annotated with type hints according to
[PEP-484](https://www.python.org/dev/peps/pep-0484/), and type-check the code at
build time with a type checking tool like [mypy](http://www.mypy-lang.org/).

```python
def func(a: int) -> List[int]:
```

<a id="s2.8-files-and-sockets"></a>
<a id="28-files-and-sockets"></a>
<a id="files-and-sockets"></a>
### 2.8 Files and Sockets 

Explicitly close files and sockets when done with them.

```python
with open("hello.txt") as hello_file:
    for line in hello_file:
        print(line)
```


<a id="s2.9-abstract-classes"></a>
<a id="29-abstract-classes"></a>
<a id="abstract-classes"></a>
### 2.9 Abstract Classes 

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


<a id="s3-python-style-rules"></a>
<a id="3-python-style-rules"></a>
<a id="python-style-rules"></a>
## 3 Python Style Rules 

<a id="s3.1-line-length"></a>
<a id="31-line-length"></a>
<a id="line-length"></a>
### 3.1 Line length 

Maximum line length is *120 characters*.

Explicit exceptions to the 120 character limit:

-   Long import statements.
-   URLs, pathnames, or long flags in comments.
-   Long string module level constants not containing whitespace that would be
    inconvenient to split across lines such as URLs or pathnames.
    -   Pylint disable comments. (e.g.: `# pylint: disable=invalid-name`)

<a id="s3.2-indentation"></a>
<a id="32-indentation"></a>
<a id="indentation"></a>
### 3.2 Indentation 

Indent your code blocks with *4 spaces*.

Never use tabs or mix tabs and spaces. In cases of implied line continuation,
you should align wrapped elements either verticall; or using a hanging indent of 4 spaces,
in which case there should be nothing after the open parenthesis or bracket on
the first line.

```python
# Correct:   

# Aligned with opening delimiter
foo = long_function_name(var_one, var_two,
                         var_three, var_four)
meal = (spam,
       beans)

# Aligned with opening delimiter in a dictionary
foo = {
   long_dictionary_key: value1 +
                        value2,
   ...
}

# 4-space hanging indent; nothing on first line
foo = long_function_name(
   var_one, var_two, var_three,
   var_four)
meal = (
   spam,
   beans)

# 4-space hanging indent in a dictionary
foo = {
   long_dictionary_key:
       long_dictionary_value,
   ...
}
```

```python
# Wrong:

# Stuff on first line forbidden
foo = long_function_name(var_one, var_two,
   var_three, var_four)
meal = (spam,
   beans)

# 2-space hanging indent forbidden
foo = long_function_name(
 var_one, var_two, var_three,
 var_four)

# No hanging indent in a dictionary
foo = {
   long_dictionary_key:
   long_dictionary_value,
   ...
}
```

<a id="s3.3-blank-lines"></a>
<a id="33-blank-lines"></a>
<a id="blank-lines"></a>
### 3.3 Blank Lines 

Two blank lines between top-level definitions, be they function or class
definitions. One blank line between method definitions and between the `class`
line and the first method. No blank line following a `def` line. Use single
blank lines as you judge appropriate within functions or methods.

<a id="s3.4-whitespace"></a>
<a id="34-whitespace"></a>
<a id="whitespace"></a>
### 3.4 Whitespace 

Follow standard typographic rules for the use of spaces around punctuation.

No whitespace inside parentheses, brackets or braces.

```python
# Correct: 
spam(ham[1], {eggs: 2}, [])
```

```python
# Wrong:
spam( ham[ 1 ], { eggs: 2 }, [ ] )
```

No whitespace before a comma, semicolon, or colon. Do use whitespace after a
comma, semicolon, or colon, except at the end of the line.

```python
# Correct: 
if x == 4:
     print(x, y)
 x, y = y, x
```

```python
# Wrong:
if x == 4 :
     print(x , y)
 x , y = y , x
```

No whitespace before the open paren/bracket that starts an argument list,
indexing or slicing.

```python
# Correct: 
spam(1)
```

```python
# Wrong:
spam (1)
```

```python
# Correct: 
dict['key'] = list[index]
```

```python
# Wrong:
dict ['key'] = list [index]
```

No trailing whitespace.

Surround binary operators with a single space on either side for assignment
(`=`), comparisons (`==, <, >, !=, <>, <=, >=, in, not in, is, is not`), and
Booleans (`and, or, not`). Use your better judgment for the insertion of spaces
around arithmetic operators (`+`, `-`, `*`, `/`, `//`, `%`, `**`, `@`).

```python
# Correct: 
x == 1
```

```python
# Wrong:
x<1
```

Never use spaces around `=` when passing keyword arguments or defining a default
parameter value, with one exception:
when a type annotation is present, _do_ use spaces
around the `=` for the default parameter value.

```python
# Correct:
def complex(real, imag=0.0): return Magic(r=real, i=imag)
def complex(real, imag: float = 0.0): return Magic(r=real, i=imag)
```

```python
# Wrong:
def complex(real, imag = 0.0): return Magic(r = real, i = imag)
def complex(real, imag: float=0.0): return Magic(r = real, i = imag)
```

Don't use spaces to vertically align tokens on consecutive lines, since it
becomes a maintenance burden (applies to `:`, `#`, `=`, etc.):

```python
# Correct:
foo = 1000  # comment
long_name = 2  # comment that should not be aligned

dictionary = {
    'foo': 1,
    'long_name': 2,
}
```

```python
# Wrong:
foo       = 1000  # comment
long_name = 2     # comment that should not be aligned

dictionary = {
    'foo'      : 1,
    'long_name': 2,
}
```

<a id="s3.5-comments-and-docstrings"></a>
<a id="35-comments-and-docstrings"></a>
<a id="comments-and-docstrings"></a>
### 3.5 Comments and Docstrings 

Be sure to use the right style for module, function, method docstrings and
inline comments.

<a id="s3.5.1-modules"></a>
<a id="351-modules"></a>
<a id="modules"></a>
#### 3.5.1 Modules 

Every file should contain a license boilerplate.

```python
"""
 Copyright (c) 2020 Intel Corporation
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
```

<a id="s3.5.2-functions-and-methods"></a>
<a id="352-functions-and-methods"></a>
<a id="functions-and-methods"></a>
#### 3.5.2 Functions and Methods 

In this section, "function" means a method, function, or generator.

A function must have a docstring, unless it meets all of the following criteria:
-   not externally visible
-   very short
-   obvious

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

<a id="s3.5.3-classes"></a>
<a id="353-classes"></a>
<a id="classes"></a>
#### 3.5.3 Classes 

Classes should have a docstring below the class definition describing the class. If your class
has public attributes, they should be documented here follow the same formatting as a function's
params section.

```python
class ModelTransformer:
    """
    Applies transformations to the model.

    :param public_attribute: Public attribute description
    """

    def __init__(self, model: ModelType, transformation_layout: TransformationLayout):
        """
        Initializes Model Transformer

        :param model: The model to be transformed
        :param transformation_layout: An instance of `TransformationLayout` that
            includes a list of transformations to be applied to the model.
        """
        self._model = model
        self._transformations = transformation_layout.transformations
        self.public_attribute = None

    def transform(self) -> ModelType:
        """
        Applies transformations to the model.

        :return: The transformed model
        """
        raise NotImplementedError()
```

<a id="s3.5.4-block-and-inline-comments"></a>
<a id="354-block-and-inline-comments"></a>
<a id="block-and-inline-comments"></a>
#### 3.5.4 Block and Inline Comments 

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

<a id="s3.6-strings"></a>
<a id="36-strings"></a>
<a id="strings"></a>
### 3.6 Strings 

Use `'something'` instead of `"something"`

```python
# Correct:

long_string = '''This is fine if your use case can accept
    extraneous leading spaces.'''

long_string = ('And this is fine if you cannot accept\n' +
               'extraneous leading spaces.')

long_string = ('And this too is fine if you cannot accept\n'
               'extraneous leading spaces.')

import textwrap

long_string = textwrap.dedent('''\
    This is also fine, because textwrap.dedent()
    will collapse common leading spaces in each line.''')
```

<a id="s3.7-logging"></a>
<a id="37-logging"></a>
<a id="logging"></a>
### 3.7 Logging 

For logging functions that expect a pattern-string (with %-placeholders) as
their first argument: Always call them with a string literal (not an f-string!)
as their first argument with pattern-parameters as subsequent arguments. Some
logging implementations collect the unexpanded pattern-string as a queryable
field. It also prevents spending time rendering a message that no logger is
configured to output.

```python
# Correct:
import nncf
from nncf.common.utils.logger import logger as nncf_logger
nncf_logger.info('Test message: %s', nncf.__version__)
```

<a id="s3.8-error-messages"></a>
<a id="38-error-messages"></a>
<a id="error-messages"></a>
### 3.8 Error Messages 

Error messages (such as: message strings on exceptions like `ValueError`, or
messages shown to the user) should follow guidelines:
- The message needs to precisely match the actual error condition.
- Interpolated pieces need to always be clearly identifiable as such.
- The message should start with a capital letter.

<a id="s3.9-todo-comments"></a>
<a id="39-todo-comments"></a>
<a id="todo-comments"></a>
### 3.9 TODO Comments 

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

<a id="s3.10-naming"></a>
<a id="310-naming"></a>
<a id="naming"></a>
### 3.10 Naming 

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

<a id="s3.10.1-names-to-avoid"></a>
<a id="3101-names-to-avoid"></a>
<a id="names-to-avoid"></a>
#### 3.10.1 Names to Avoid 

-   single character names, except for specifically allowed cases:
    -   counters or iterators (e.g. `i`, `j`, `k`, `v`, et al.)
    -   `e` as an exception identifier in `try/except` statements.
    -   `f` as a file handle in `with` statements
    Please be mindful not to abuse single-character naming. Generally speaking,
    descriptiveness should be proportional to the name's scope of visibility.
    For example, `i` might be a fine name for 5-line code block but within
    multiple nested scopes, it is likely too vague.
-   dashes (`-`) in any package/module name
-   `__double_leading_and_trailing_underscore__` names (reserved by Python)
-   offensive terms
-   names that needlessly include the type of the variable (for example:
    `id_to_name_dict`)

<a id="s3.10.2-naming-conventions"></a>
<a id="3102-naming-convention"></a>
<a id="naming-conventions"></a>
#### 3.10.2 Naming Conventions 

-   "Internal" means internal to a module, or protected or private within a
    class.
-   Prepending a single underscore (`_`) has some support for protecting module
    variables and functions (linters will flag protected member access). While
    prepending a double underscore (`__` aka "dunder") to an instance variable
    or method effectively makes the variable or method private to its class
    (using name mangling); we discourage its use as it impacts readability and
    testability, and isn't *really* private.
-   Place related classes and top-level functions together in a
    module.
-   Use CapWords for class names, but lower\_with\_under.py for module names.
-   Use the word "layer" (instead of "module") in the `nncf.common` module to
    refer to the building block of neural networks.

<a id="s3.10.3-framework-specific-class-naming"></a>
<a id="3103-framework-specific-class-naming"></a>
<a id="framework-specific-class-naming"></a>
#### 3.10.3 Framework specific class naming 

- `PTClassName` for Torch
- `TFClassName` for TensorFlow

<a id="s3.10.4-file-naming"></a>
<a id="3104-file-naming"></a>
<a id="file-naming"></a>
#### 3.10.4 File Naming 

Python filenames must have a `.py` extension and must not contain dashes (`-`).
This allows them to be imported and unit tested.

<a id="s3.11-main"></a>
<a id="311-main"></a>
<a id="main"></a>
### 3.11 Main 

```python
def main():
    ...

if __name__ == '__main__':
    main()
```
