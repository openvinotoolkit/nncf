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
from collections import namedtuple
from functools import partial
from typing import Any

import pytest

from nncf.torch.nested_objects_traversal import objwalk


class ObjwalkTestClass:
    def __init__(self, field: int):
        self.field = field

    def member_fn(self, val):
        return ObjwalkTestClass(self.field + 1)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


NamedTuple = namedtuple("NamedTuple", ("field1", "field2"))

OBJWALK_INIT_VAL = 0
OBJWALK_REF_VAL = OBJWALK_INIT_VAL + 1
TEST_VS_REF_OBJECTS_TO_WALK = [
    (0, 0),
    ("foo", "foo"),
    (ObjwalkTestClass(OBJWALK_INIT_VAL), ObjwalkTestClass(OBJWALK_REF_VAL)),
    ([0, ObjwalkTestClass(OBJWALK_INIT_VAL), "bar"], [0, ObjwalkTestClass(OBJWALK_REF_VAL), "bar"]),
    (
        [ObjwalkTestClass(OBJWALK_INIT_VAL), ObjwalkTestClass(OBJWALK_INIT_VAL), (5, 8)],
        [ObjwalkTestClass(OBJWALK_REF_VAL), ObjwalkTestClass(OBJWALK_REF_VAL), (5, 8)],
    ),
    (
        {"obj1": ObjwalkTestClass(OBJWALK_INIT_VAL), "obj2": ObjwalkTestClass(OBJWALK_INIT_VAL)},
        {"obj1": ObjwalkTestClass(OBJWALK_REF_VAL), "obj2": ObjwalkTestClass(OBJWALK_REF_VAL)},
    ),
    ((ObjwalkTestClass(OBJWALK_INIT_VAL), 42), (ObjwalkTestClass(OBJWALK_REF_VAL), 42)),
    (
        [
            (ObjwalkTestClass(OBJWALK_INIT_VAL), 8),
            [ObjwalkTestClass(OBJWALK_INIT_VAL), "foo"],
            {
                "bar": ObjwalkTestClass(OBJWALK_INIT_VAL),
                "baz": (ObjwalkTestClass(OBJWALK_INIT_VAL), ObjwalkTestClass(OBJWALK_INIT_VAL)),
                "xyzzy": {1337: ObjwalkTestClass(OBJWALK_INIT_VAL), 31337: ObjwalkTestClass(OBJWALK_INIT_VAL)},
            },
        ],
        [
            (ObjwalkTestClass(OBJWALK_REF_VAL), 8),
            [ObjwalkTestClass(OBJWALK_REF_VAL), "foo"],
            {
                "bar": ObjwalkTestClass(OBJWALK_REF_VAL),
                "baz": (ObjwalkTestClass(OBJWALK_REF_VAL), ObjwalkTestClass(OBJWALK_REF_VAL)),
                "xyzzy": {1337: ObjwalkTestClass(OBJWALK_REF_VAL), 31337: ObjwalkTestClass(OBJWALK_REF_VAL)},
            },
        ],
    ),
    (
        (0, NamedTuple(field1=ObjwalkTestClass(OBJWALK_INIT_VAL), field2=-5.3), "bar"),
        (0, NamedTuple(field1=ObjwalkTestClass(OBJWALK_REF_VAL), field2=-5.3), "bar"),
    ),
]


@pytest.fixture(name="objwalk_objects", params=TEST_VS_REF_OBJECTS_TO_WALK)
def objwalk_objects_(request):
    return request.param


def test_objwalk(objwalk_objects):
    start_obj = objwalk_objects[0]
    ref_obj = objwalk_objects[1]

    def is_target_class(obj):
        return isinstance(obj, ObjwalkTestClass)

    fn_to_apply = partial(ObjwalkTestClass.member_fn, val=OBJWALK_REF_VAL)

    test_obj = objwalk(start_obj, is_target_class, fn_to_apply)

    assert test_obj == ref_obj


def assert_named_tuples_are_equal(ref_named_tuple: tuple, test_obj: Any):
    assert test_obj.__class__.__qualname__ == ref_named_tuple.__class__.__qualname__
    assert hasattr(test_obj, "_fields")
    assert all(f in test_obj._fields for f in ref_named_tuple._fields)
    assert all(f in ref_named_tuple._fields for f in test_obj._fields)


def test_objwalk_retains_named_tuple():
    named_tuple = NamedTuple(
        field1=ObjwalkTestClass(OBJWALK_INIT_VAL),
        field2=NamedTuple(field1=ObjwalkTestClass(OBJWALK_INIT_VAL), field2=-8),
    )

    def is_target_class(obj):
        return isinstance(obj, ObjwalkTestClass)

    fn_to_apply = partial(ObjwalkTestClass.member_fn, val=OBJWALK_REF_VAL)
    test_obj = objwalk(named_tuple, is_target_class, fn_to_apply)
    assert_named_tuples_are_equal(named_tuple, test_obj)
    assert_named_tuples_are_equal(named_tuple.field2, test_obj.field2)
