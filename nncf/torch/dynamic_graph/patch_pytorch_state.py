from dataclasses import dataclass


@dataclass
class PatchingState:
    """
    A class to track which pytorch components were patched by NNCF.
    """

    jit_is_wrapped: bool = False
    operators_are_wrapped: bool = False
    compile_is_wrapped: bool = False


PATCHING_STATE = PatchingState()
