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
import itertools
from functools import partial

import pytest
from torch.nn import DataParallel

import nncf
from nncf.common.logging import nncf_logger
from nncf.torch import load_state
from nncf.torch import register_default_init_args
from nncf.torch.compression_method_api import PTCompressionAlgorithmBuilder
from nncf.torch.quantization.algo import QuantizationBuilder
from tests.torch.helpers import BasicConvTestModel
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import create_ones_mock_dataloader
from tests.torch.helpers import get_empty_config
from tests.torch.helpers import register_bn_adaptation_init_args
from tests.torch.quantization.test_manual_precision_init import PrecisionInitTestDesc
from tests.torch.sparsity.rb.test_algo import get_basic_sparsity_config


@pytest.fixture()
def _nncf_caplog(caplog):
    nncf_logger.propagate = True
    yield caplog
    nncf_logger.propagate = False


LIST_MANUAL_INIT_CASES = [
    PrecisionInitTestDesc().config_with_all_inits().resume_with_the_same_config(),
    PrecisionInitTestDesc().config_with_all_inits().resume_with_the_same_config_without_init(),
]


@pytest.mark.parametrize("desc", LIST_MANUAL_INIT_CASES, ids=map(str, LIST_MANUAL_INIT_CASES))
def test_can_resume_with_manual_init(mocker, desc, _nncf_caplog):
    config = desc.config
    config_to_resume = desc.config_to_resume

    config = register_default_init_args(config, train_loader=create_ones_mock_dataloader(config))
    all_spies = desc.setup_init_spies(mocker)
    init_spy = mocker.spy(PTCompressionAlgorithmBuilder, "__init__")
    get_setup_spy = mocker.spy(QuantizationBuilder, "_get_quantizer_setup")

    _, compression_ctrl = create_compressed_model_and_algo_for_test(desc.model_creator(), config)
    desc.check_precision_init(compression_ctrl)

    for m in all_spies:
        m.assert_called()
        m.reset_mock()
    get_setup_spy.assert_called()
    get_setup_spy.reset_mock()

    compression_state = compression_ctrl.get_compression_state()
    register_bn_adaptation_init_args(config_to_resume)
    _, compression_ctrl = create_compressed_model_and_algo_for_test(
        desc.model_creator(), config_to_resume, compression_state=compression_state
    )

    if config_to_resume is not None and config_to_resume["compression"]["initializer"]:
        assert not init_spy.call_args[0][2]

    for m in all_spies:
        m.assert_not_called()
    get_setup_spy.assert_not_called()

    desc.check_precision_init(compression_ctrl)


# TODO(nlyalyus): fix algo mixing
@pytest.mark.skip("algo mixing is not supported")
@pytest.mark.parametrize("is_strict", (True, False))
def test_can_resume_with_algo_mixing(mocker, is_strict):
    desc = PrecisionInitTestDesc().config_with_all_inits()
    all_quantization_init_spies = desc.setup_init_spies(mocker)
    sparsity_config = get_basic_sparsity_config()
    sparsity_config["target_device"] = "TRIAL"
    config = desc.config
    quantization_section = config["compression"]
    config["compression"] = [{"algorithm": "const_sparsity"}, quantization_section]

    _, compression_ctrl = create_compressed_model_and_algo_for_test(desc.model_creator(), sparsity_config)
    compression_state = compression_ctrl.get_compression_state()

    config = register_default_init_args(config, train_loader=create_ones_mock_dataloader(config))
    fn = partial(
        create_compressed_model_and_algo_for_test, desc.model_creator(), config, compression_state=compression_state
    )
    if is_strict:
        with pytest.raises(nncf.InternalError):
            fn()
    else:
        _, compression_ctrl = fn()
        for m in all_quantization_init_spies:
            m.assert_called()
        desc.check_precision_init(compression_ctrl.child_ctrls[1])


QUANTIZATION = "quantization"
SPARSITY_TYPES = ["magnitude", "rb", "const"]
SPARSITY_ALGOS = {"_".join([type, "sparsity"]) for type in SPARSITY_TYPES}  # 3S

LOAD_ALGOS = list(itertools.product([QUANTIZATION], SPARSITY_ALGOS))  # Q + 3S
LOAD_ALGOS += itertools.product(SPARSITY_ALGOS, [QUANTIZATION])  # 3S + Q

SAVE_ALGOS = [[algo] for algo in SPARSITY_ALGOS]  # 3S
SAVE_ALGOS += [[QUANTIZATION]]  # Q
SAVE_ALGOS += LOAD_ALGOS  # Q , 3S, 3S + Q, Q+3S

ALGOS = list(sorted(itertools.product(SAVE_ALGOS, LOAD_ALGOS), key=lambda x: "_".join(x[0]) + "_".join(x[1])))


@pytest.fixture(
    scope="module", params=ALGOS, ids=["__".join(["save:" + "_".join(a[0]), "load:" + "_".join(a[1])]) for a in ALGOS]
)
def _algos(request):
    pair_algos = request.param
    save_algos = pair_algos[0]
    load_algos = pair_algos[1]
    resume_ok = False
    # resume expects the same list of algorithms
    if save_algos == load_algos:
        resume_ok = True

    if len(save_algos) == len(load_algos):
        for s, v in zip(save_algos, load_algos):
            if s != v and ("magnitude" in s and "const" in v or "const" in s and "magnitude" in v):
                resume_ok = True

        # Priority mechanism ensures that algo permutations are irrelevant
        if set(save_algos) == set(load_algos):
            resume_ok = True
        else:
            saved_sparsity = filter(lambda x: x != QUANTIZATION, save_algos)
            loaded_sparsity = filter(lambda x: x != QUANTIZATION, load_algos)

            for s, v in zip(saved_sparsity, loaded_sparsity):
                # resume works fine for magnitude <-> const combo, because they have similar parameters
                if s != v and ("magnitude" in s and "const" in v or "const" in s and "magnitude" in v):
                    resume_ok = True

    return {"save_algos": save_algos, "load_algos": load_algos, "is_resume_ok": resume_ok}


MODEL_WRAPPER = ["CPU", "GPU"]
WRAPPERS = list(sorted(itertools.product(MODEL_WRAPPER, MODEL_WRAPPER), key=lambda x: "_".join(x)))


@pytest.fixture(scope="function", params=WRAPPERS, ids=["_".join(["from:" + w[0], "to:" + w[1]]) for w in WRAPPERS])
def _model_wrapper(request):
    modes = request.param

    def wrap_model(mode, model):
        if mode == "GPU":
            model = DataParallel(model, [0])
        return model

    return {
        "save_model": partial(wrap_model, modes[0]),
        "resume_model": partial(wrap_model, modes[1]),
    }


@pytest.mark.parametrize("is_resume", (True, False), ids=["resume", "load_weights"])
def test_load_state_interoperability(_algos, _model_wrapper, is_resume):
    config_save = get_empty_config()
    config_save["compression"] = [{"algorithm": algo} for algo in _algos["save_algos"]]
    register_bn_adaptation_init_args(config_save)
    compressed_model_save, _ = create_compressed_model_and_algo_for_test(BasicConvTestModel(), config_save)
    model_save = _model_wrapper["save_model"](compressed_model_save)
    saved_model_state = model_save.state_dict()
    ref_num_loaded = len(saved_model_state)

    config_resume = get_empty_config()
    config_resume["compression"] = [{"algorithm": algo} for algo in _algos["load_algos"]]
    register_bn_adaptation_init_args(config_resume)
    compressed_model_resume, _ = create_compressed_model_and_algo_for_test(BasicConvTestModel(), config_resume)
    model_resume = _model_wrapper["resume_model"](compressed_model_resume)

    if not is_resume or (is_resume and _algos["is_resume_ok"]):
        act_num_loaded = load_state(model_resume, saved_model_state, is_resume)

        if (
            "magnitude_sparsity" in _algos["load_algos"] or "const_sparsity" in _algos["load_algos"]
        ) and "rb_sparsity" in _algos["save_algos"]:
            # no need to load _mask and _uniform
            ref_num_loaded -= 2
        assert act_num_loaded == ref_num_loaded
    else:
        with pytest.raises(nncf.InternalError):
            load_state(model_resume, saved_model_state, is_resume)


RESUME_ALGOS = list(itertools.product([QUANTIZATION], SPARSITY_ALGOS))  # Q + 3S
RESUME_ALGOS += [[algo] for algo in SPARSITY_ALGOS]  # 3S
RESUME_ALGOS += [[QUANTIZATION]]  # Q
RESUME_ALGOS += [["EMPTY"]]  # No Compression
RESUME_ALGOS = list(
    sorted(itertools.product(RESUME_ALGOS, RESUME_ALGOS), key=lambda x: "_".join(x[0]) + "_".join(x[1]))
)
NUM_PARAMS_PER_ALGO = {QUANTIZATION: 8, "magnitude_sparsity": 1, "const_sparsity": 1, "rb_sparsity": 3, "EMPTY": 0}


@pytest.fixture(
    scope="module",
    params=RESUME_ALGOS,
    ids=["__".join(["save:" + "_".join(a[0]), "load:" + "_".join(a[1])]) for a in RESUME_ALGOS],
)
def _resume_algos(request):
    pair_algos = request.param
    save_algos = pair_algos[0]
    load_algos = pair_algos[1]
    is_strict = True

    sparsity_on_save = SPARSITY_ALGOS.intersection(save_algos)
    sparsity_on_load = SPARSITY_ALGOS.intersection(load_algos)
    common_algos = set(save_algos).intersection(set(load_algos))
    different_algos = set(save_algos).symmetric_difference(set(load_algos))
    if different_algos:
        is_strict = False

    ref_num_compression_params = sum(map(lambda x: NUM_PARAMS_PER_ALGO[x], common_algos))
    if not SPARSITY_ALGOS.intersection(common_algos) and (sparsity_on_save and sparsity_on_load):
        ref_num_compression_params += 1

    return {
        "save_algos": save_algos,
        "load_algos": load_algos,
        "is_strict": is_strict,
        "ref_num_compression_params": ref_num_compression_params,
    }


def test_load_state__with_resume_checkpoint(_resume_algos, _model_wrapper, mocker):
    config_save = get_empty_config()
    config_save["compression"] = [{"algorithm": algo} for algo in _resume_algos["save_algos"] if algo != "EMPTY"]
    register_bn_adaptation_init_args(config_save)
    orig_model = BasicConvTestModel()
    num_model_params = len(orig_model.state_dict())
    model_save, compressed_ctrl_save = create_compressed_model_and_algo_for_test(orig_model, config_save)
    saved_model_state = model_save.state_dict()
    saved_checkpoint = compressed_ctrl_save.get_compression_state()
    ref_num_loaded = _resume_algos["ref_num_compression_params"] + num_model_params + 1  # padding_value

    config_resume = get_empty_config()
    config_resume["compression"] = [{"algorithm": algo} for algo in _resume_algos["load_algos"] if algo != "EMPTY"]
    register_bn_adaptation_init_args(config_resume)
    from nncf.torch.checkpoint_loading import KeyMatcher

    key_matcher_run_spy = mocker.spy(KeyMatcher, "run")
    model, _ = create_compressed_model_and_algo_for_test(
        BasicConvTestModel(), config_resume, compression_state=saved_checkpoint
    )
    load_state(model, saved_model_state, _resume_algos["is_strict"])
    key_matcher_run_spy.assert_called_once()
    act_num_loaded = len(key_matcher_run_spy.spy_return)
    assert act_num_loaded == ref_num_loaded


LIST_ALGOS = sorted(["", QUANTIZATION] + list(SPARSITY_ALGOS))


@pytest.mark.parametrize("is_resume", (True, False), ids=["resume", "load_weights"])
@pytest.mark.parametrize("algo", tuple(LIST_ALGOS))
def test_ordinary_load(algo, _model_wrapper, is_resume):
    config = get_empty_config()
    if algo:
        config["compression"] = {"algorithm": algo}
    register_bn_adaptation_init_args(config)

    compressed_model_save, _ = create_compressed_model_and_algo_for_test(BasicConvTestModel(), config)
    model_save = _model_wrapper["save_model"](compressed_model_save)

    compressed_model_resume, _ = create_compressed_model_and_algo_for_test(BasicConvTestModel(), config)
    model_resume = _model_wrapper["resume_model"](compressed_model_resume)

    num_loaded = load_state(model_resume, model_save.state_dict(), is_resume)

    assert num_loaded == len(model_save.state_dict())
