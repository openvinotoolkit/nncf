# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import patch

from nncf.common.utils.patcher import PATCHER

CORRECT_WRAPPER_STACK = "base"


def wrapper1(self, fn, *args, **kwargs):
    kwargs["wrapper_stack"] += "_wrapper1"
    return fn(*args, **kwargs)


def wrapper2(self, fn, *args, **kwargs):
    kwargs["wrapper_stack"] += "_wrapper2"
    return fn(*args, **kwargs)


def assert_wrapper_stack(wrapper_stack=None):
    assert wrapper_stack == CORRECT_WRAPPER_STACK, f"{wrapper_stack} != {CORRECT_WRAPPER_STACK}"


class TestOverrideClass:
    def assert_wrapper_stack_method(self, wrapper_stack=None):
        assert wrapper_stack == CORRECT_WRAPPER_STACK, f"{wrapper_stack} != {CORRECT_WRAPPER_STACK}"

    @staticmethod
    def assert_wrapper_stack_static(wrapper_stack=None):
        assert wrapper_stack == CORRECT_WRAPPER_STACK, f"{wrapper_stack} != {CORRECT_WRAPPER_STACK}"

    @classmethod
    def assert_wrapper_stack_class(cls, wrapper_stack=None):
        assert wrapper_stack == CORRECT_WRAPPER_STACK, f"{wrapper_stack} != {CORRECT_WRAPPER_STACK}"

    def ori_method(self):
        return "placeholder_method"


def test_patcher():
    global CORRECT_WRAPPER_STACK

    def wrapper3(self, fn, *args, **kwargs):
        kwargs["wrapper_stack"] += "_wrapper3"
        return fn(*args, **kwargs)

    test_obj = TestOverrideClass()

    # Test without patching
    assert_wrapper_stack(wrapper_stack="base")
    test_obj.assert_wrapper_stack_method(wrapper_stack="base")
    TestOverrideClass.assert_wrapper_stack_static(wrapper_stack="base")
    TestOverrideClass.assert_wrapper_stack_class(wrapper_stack="base")

    # Test non-class method
    PATCHER.patch(assert_wrapper_stack, wrapper1)
    CORRECT_WRAPPER_STACK = "base_wrapper1"
    assert_wrapper_stack(wrapper_stack="base")

    # Test single patch static method
    PATCHER.patch(TestOverrideClass.assert_wrapper_stack_static, wrapper1)
    CORRECT_WRAPPER_STACK = "base_wrapper1"
    test_obj.assert_wrapper_stack_static(wrapper_stack="base")  # doesn't work if called from class level

    # Test single patch class method
    PATCHER.patch(TestOverrideClass.assert_wrapper_stack_class, wrapper1)
    CORRECT_WRAPPER_STACK = "base_wrapper1"
    test_obj.assert_wrapper_stack_class(wrapper_stack="base")  # doesn't work if called from class level

    # Test single patch object method
    PATCHER.patch(TestOverrideClass.assert_wrapper_stack_method, wrapper1)
    CORRECT_WRAPPER_STACK = "base_wrapper1"
    test_obj.assert_wrapper_stack_method(wrapper_stack="base")

    # Test applying two nested patches
    PATCHER.patch(TestOverrideClass.assert_wrapper_stack_method, wrapper2, force=False)
    CORRECT_WRAPPER_STACK = "base_wrapper2_wrapper1"
    test_obj.assert_wrapper_stack_method(wrapper_stack="base")

    # Test applying three nested patches
    PATCHER.patch(TestOverrideClass.assert_wrapper_stack_method, wrapper3, force=False)
    CORRECT_WRAPPER_STACK = "base_wrapper3_wrapper2_wrapper1"
    test_obj.assert_wrapper_stack_method(wrapper_stack="base")

    # Test unpatching with depth = 0
    PATCHER.unpatch(TestOverrideClass.assert_wrapper_stack_method, depth=0)
    CORRECT_WRAPPER_STACK = "base"
    test_obj.assert_wrapper_stack_method(wrapper_stack="base")

    # Test unpatching with depth = 1
    PATCHER.patch(TestOverrideClass.assert_wrapper_stack_method, wrapper1, force=False)
    PATCHER.patch(TestOverrideClass.assert_wrapper_stack_method, wrapper2, force=False)
    PATCHER.unpatch(TestOverrideClass.assert_wrapper_stack_method, depth=1)
    CORRECT_WRAPPER_STACK = "base_wrapper1"
    test_obj.assert_wrapper_stack_method(wrapper_stack="base")

    # Test unpatching with depth = 2
    PATCHER.patch(TestOverrideClass.assert_wrapper_stack_method, wrapper2, force=False)
    PATCHER.unpatch(TestOverrideClass.assert_wrapper_stack_method, depth=2)
    CORRECT_WRAPPER_STACK = "base"
    test_obj.assert_wrapper_stack_method(wrapper_stack="base")

    # Test overriding patch
    PATCHER.patch(TestOverrideClass.assert_wrapper_stack_method, wrapper2, force=False)
    PATCHER.patch(TestOverrideClass.assert_wrapper_stack_method, wrapper1, force=True)
    CORRECT_WRAPPER_STACK = "base_wrapper1"
    test_obj.assert_wrapper_stack_method(wrapper_stack="base")

    # Unpatch with depth = 0
    PATCHER.unpatch(TestOverrideClass.assert_wrapper_stack_method, depth=0)
    CORRECT_WRAPPER_STACK = "base"
    test_obj.assert_wrapper_stack_method(wrapper_stack="base")

    # Test single patch instance method
    PATCHER.patch(test_obj.assert_wrapper_stack_method, wrapper1, force=False)
    CORRECT_WRAPPER_STACK = "base_wrapper1"
    test_obj.assert_wrapper_stack_method(wrapper_stack="base")


def test_attribute_error_module():
    with patch("inspect.getfullargspec", return_value=[["mock_value"]]):
        with patch.object(TestOverrideClass, "__getattribute__", side_effect=AttributeError()):
            pass


def test_attribute_error_class():
    with patch("inspect.getfullargspec", return_value=[["mock_value1, mock_value2"]]):
        with patch.object(TestOverrideClass, "__getattribute__", side_effect=AttributeError()):
            pass
