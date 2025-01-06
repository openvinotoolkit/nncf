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
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import pytest
import torch
from addict import Dict
from datasets import Dataset
from onnx import numpy_helper
from openvino._offline_transformations import apply_fused_names_cleanup
from openvino._offline_transformations import apply_moc_transformations
from openvino._offline_transformations import apply_pruning_transformation
from openvino.runtime import Core
from openvino.runtime import serialize
from packaging import version
from scipy.special import softmax
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from nncf.torch import create_compressed_model
from nncf.torch.checkpoint_loading import load_state
from tests.torch.helpers import PTTensorListComparator
from tests.torch.sparsity.movement.helpers import BaseMockRunRecipe
from tests.torch.sparsity.movement.helpers import BertRunRecipe
from tests.torch.sparsity.movement.helpers import ClipVisionRunRecipe
from tests.torch.sparsity.movement.helpers import Conv2dPlusLinearRunRecipe
from tests.torch.sparsity.movement.helpers import DistilBertRunRecipe
from tests.torch.sparsity.movement.helpers import LinearRunRecipe
from tests.torch.sparsity.movement.helpers import MobileBertRunRecipe
from tests.torch.sparsity.movement.helpers import SwinRunRecipe
from tests.torch.sparsity.movement.helpers import Wav2Vec2RunRecipe
from tests.torch.sparsity.movement.helpers import build_compression_trainer
from tests.torch.sparsity.movement.helpers import force_update_sparsifier_binary_masks_by_threshold
from tests.torch.sparsity.movement.helpers import initialize_sparsifier_parameters_by_linspace


@pytest.fixture(scope="function", autouse=True)
def safe_deterministic_state(_safe_deterministic_state):
    pass


class TestONNXExport:
    @pytest.mark.parametrize(
        "recipe",
        [
            BertRunRecipe(),
            Wav2Vec2RunRecipe(),
            SwinRunRecipe(),
            LinearRunRecipe(),
        ],
    )
    def test_can_export_compressed_model(self, recipe: BaseMockRunRecipe, tmp_path):
        recipe.log_dir_(tmp_path)
        compression_ctrl, _ = create_compressed_model(recipe.model(), recipe.nncf_config(), dump_graphs=False)
        onnx_path = str(tmp_path / "model.onnx")
        compression_ctrl.export_model(onnx_path)
        assert Path(onnx_path).exists()

    def test_no_weight_override_on_export(self, tmp_path):
        recipe = LinearRunRecipe(log_dir=tmp_path)
        onnx_path = str(tmp_path / "model.onnx")
        compression_ctrl, compressed_model = create_compressed_model(
            recipe.model(), recipe.nncf_config(), dump_graphs=False
        )
        for minfo in compression_ctrl.sparsified_module_info:
            initialize_sparsifier_parameters_by_linspace(minfo.operand, -1, 0.0)
            force_update_sparsifier_binary_masks_by_threshold(minfo.operand, 1.0)  # all-zero masks
        state_before = deepcopy(compressed_model.state_dict())
        compression_ctrl.export_model(onnx_path)
        state_after = compressed_model.state_dict()
        PTTensorListComparator.check_equal(list(state_before.values()), list(state_after.values()))

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse("1.12"),
        reason=f"torch {torch.__version__} is not compatible with installed transformers package. "
        f"Some tests may fail with segmentation fault",
    )
    @pytest.mark.parametrize(
        "recipe",
        [
            BertRunRecipe(),
            Wav2Vec2RunRecipe(),
            SwinRunRecipe(),
            LinearRunRecipe(),
            Conv2dPlusLinearRunRecipe(),
            SwinRunRecipe()
            .scheduler_params_(enable_structured_masking=False)
            .model_config_(
                image_size=384,
                patch_size=4,
                window_size=12,
                embed_dim=192,
                mlp_ratio=4,
                depths=(2, 2, 5, 2),
                num_heads=(6, 12, 24, 48),
                num_labels=32,
            ),
        ],
    )
    def test_same_outputs_in_torch_and_exported_onnx(self, tmp_path: Path, recipe: BaseMockRunRecipe):
        num_samples = 4
        recipe.log_dir_(tmp_path)
        compression_ctrl, compressed_model = create_compressed_model(
            recipe.model(), recipe.nncf_config(), dump_graphs=False
        )
        dataset = recipe.generate_mock_dataset(num_samples, seed=42)
        for i, minfo in enumerate(compression_ctrl.sparsified_module_info):
            initialize_sparsifier_parameters_by_linspace(minfo.operand, seed=i)
            force_update_sparsifier_binary_masks_by_threshold(minfo.operand, 0.0)

        trainer = build_compression_trainer(tmp_path, compression_ctrl, compressed_model)
        torch_outputs = trainer.predict(dataset).predictions

        compressed_model.eval()
        onnx_model_path = str(tmp_path / "model.onnx")
        compression_ctrl.export_model(onnx_model_path)
        onnx_output_dict = self._get_onnx_model_inference_outputs(onnx_model_path, dataset, recipe)
        onnx_outputs = next(iter(onnx_output_dict.values()))
        assert np.allclose(softmax(onnx_outputs, axis=-1), softmax(torch_outputs, axis=-1), atol=1e-6)

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse("1.12"),
        reason=f"torch {torch.__version__} is not compatible with installed transformers package. "
        f"Some tests may fail with segmentation fault",
    )
    @pytest.mark.parametrize(
        "desc",
        [
            Dict(
                nncf_weight_ratio=0.43,
                ov_weight_ratio=0.36,
                recipe=BertRunRecipe().model_config_(
                    max_position_embeddings=2,
                    intermediate_size=4,
                    vocab_size=2,
                    num_labels=1,
                    hidden_size=4,
                    num_attention_heads=4,
                    num_hidden_layers=1,
                ),
            ),
            Dict(
                nncf_weight_ratio=0.31,
                ov_weight_ratio=0.25,
                recipe=Wav2Vec2RunRecipe().model_config_(
                    intermediate_size=4,
                    num_labels=1,
                    num_conv_pos_embeddings=1,
                    num_attention_heads=4,
                    classifier_proj_size=1,
                ),
            ),
            Dict(
                nncf_weight_ratio=0.41,
                ov_weight_ratio=0.07,
                recipe=Wav2Vec2RunRecipe().model_config_(
                    # stride, hidden size and num heads is selected to reproduce invalid parsing of reshape:
                    # input_shape=[1, 4, 1, 2] and output shape = [4, 1, 2]. NGraph transformation eager finds extend
                    # and shrink mode in the reshape at the same time:
                    # [1, 4, 1] is mapped to [4] and [2] - to [1,2]. Such reshape is skipped from pruning and heads are
                    # not removed. But this reshape is just unsqueeze and pruning should work fine.
                    conv_stride=(5, 5),
                    hidden_size=8,
                    num_attention_heads=4,
                    # parameters below are for higher compression rate in NNCF: less parameters for not-pruned parts.
                    intermediate_size=4,
                    num_labels=1,
                    num_conv_pos_embeddings=1,
                    classifier_proj_size=3,
                ),
            ),
            Dict(
                nncf_weight_ratio=0.43,
                ov_weight_ratio=0.33,
                recipe=SwinRunRecipe().model_config_(
                    num_heads=[4],
                    num_labels=1,
                    num_channels=4,
                    window_size=1,
                ),
            ),
            Dict(
                nncf_weight_ratio=0.55,
                ov_weight_ratio=0.20,
                recipe=DistilBertRunRecipe().model_config_(),
            ),
            Dict(
                nncf_weight_ratio=0.63,
                ov_weight_ratio=0.38,
                recipe=MobileBertRunRecipe().model_config_(),
            ),
            Dict(
                nncf_weight_ratio=0.42,
                ov_weight_ratio=0.33,
                recipe=MobileBertRunRecipe()
                .model_config_()
                .algo_config_(
                    sparse_structure_by_scopes=[
                        {"mode": "block", "sparse_factors": [2, 2], "target_scopes": "{re}attention"},
                        {"mode": "per_dim", "axis": 0, "target_scopes": "{re}BertIntermediate"},
                        {"mode": "per_dim", "axis": 1, "target_scopes": "{re}FFNOutput"},
                    ],
                    ignored_scopes=["{re}embedding", "{re}pooler", "{re}classifier"],
                ),
            ),
            Dict(
                nncf_weight_ratio=0.47,
                ov_weight_ratio=0.36,
                recipe=ClipVisionRunRecipe().model_config_(),
            ),
        ],
    )
    def test_ngraph_pruning(self, tmp_path: Path, desc: dict):
        recipe: BaseMockRunRecipe = desc.recipe
        recipe.log_dir_(tmp_path)
        ctrl, _ = create_compressed_model(recipe.model(), recipe.nncf_config(), dump_graphs=False)

        # Zero out first 75% elements
        ctrl.reset_independent_structured_mask()
        groups = ctrl._structured_mask_handler._structured_mask_ctx_groups
        for group in groups:
            for ctx in group.structured_mask_contexts:
                shape = ctx.independent_structured_mask.shape
                num_zeros = shape[0]
                mask_to_override = ctx.independent_structured_mask
                if shape[1] != 1:
                    num_zeros = shape[1]
                    mask_to_override = mask_to_override[0]
                num_zeros = num_zeros * 3 // 4
                mask_to_override[:num_zeros] = 0
                mask_to_override[num_zeros:] = 1
        ctrl.resolve_structured_mask()
        ctrl.populate_structured_mask()
        compression_rate = ctrl.statistics().movement_sparsity.model_statistics.sparsity_level

        onnx_model_path = str(tmp_path / "structured.onnx")
        ctrl.export_model(onnx_model_path)
        core = Core()
        ov_model = core.read_model(onnx_model_path)

        # Convert to IR without pruning
        apply_moc_transformations(ov_model, cf=False)
        apply_fused_names_cleanup(ov_model)
        not_pruned_file = str(tmp_path / "ov_not_pruned.xml")
        serialize(ov_model, not_pruned_file)

        # Convert to IR with pruning
        apply_moc_transformations(ov_model, cf=False)
        apply_pruning_transformation(ov_model)
        apply_fused_names_cleanup(ov_model)
        pruned_file = str(tmp_path / "ov_pruned.xml")
        serialize(ov_model, pruned_file)

        not_pruned_file = Path(not_pruned_file.replace(".xml", ".bin"))
        pruned_file = Path(pruned_file.replace(".xml", ".bin"))
        pruned_file_bytes = pruned_file.stat().st_size
        not_pruned_file_bytes = not_pruned_file.stat().st_size
        file_size_ratio = 1 - pruned_file_bytes / not_pruned_file_bytes
        assert pytest.approx(compression_rate, abs=1e-2) == desc.nncf_weight_ratio
        assert (
            pytest.approx(file_size_ratio, abs=3e-2) == desc.ov_weight_ratio
        ), f"IR's size ratio: 1 - {pruned_file_bytes}/{not_pruned_file_bytes}"
        if abs(desc.ov_weight_ratio - desc.nncf_weight_ratio) >= 0.15:
            pytest.skip("Known issue in the ngraph transformation")
        assert abs(file_size_ratio - compression_rate) < 0.152

    def _get_onnx_model_inference_outputs(self, onnx_model_path: str, dataset: Dataset, recipe: BaseMockRunRecipe):
        sess = onnxruntime.InferenceSession(onnx_model_path)
        input_names = [item.name for item in sess.get_inputs()]
        output_names = [item.name for item in sess.get_outputs()]
        onnx_output_dict = defaultdict(list)
        for i in range(len(dataset)):
            item = dataset[i : i + 1]
            onnx_input = {}
            for input_name, input_info in zip(input_names, recipe.model_input_info.elements):
                onnx_input[input_name] = torch.tensor(item[input_info.keyword], dtype=input_info.type).numpy()
            outputs = sess.run(None, onnx_input)
            for name, output in zip(output_names, outputs):
                onnx_output_dict[name].append(output)
        return {name: np.concatenate(array_list) for name, array_list in onnx_output_dict.items()}

    @pytest.mark.parametrize(
        "linear_recipe",
        [LinearRunRecipe().model_config_(bias=True), Conv2dPlusLinearRunRecipe().model_config_(bias=False)],
    )
    def test_exported_onnx_has_sparsified_param(self, tmp_path: Path, linear_recipe: BaseMockRunRecipe):
        compression_ctrl, _ = create_compressed_model(
            linear_recipe.model(), linear_recipe.nncf_config(), dump_graphs=False
        )

        assert len(compression_ctrl.sparsified_module_info) == 1
        minfo = compression_ctrl.sparsified_module_info[0]
        initialize_sparsifier_parameters_by_linspace(minfo.operand)
        force_update_sparsifier_binary_masks_by_threshold(minfo.operand, 0.0)
        with torch.no_grad():
            ref_weight, ref_bias = minfo.operand(minfo.module.weight, minfo.module.bias)

        onnx_model_path = str(tmp_path / "model.onnx")
        compression_ctrl.export_model(onnx_model_path)
        onnx_model = onnx.load(onnx_model_path)

        assert ref_weight.count_nonzero() > 0
        assert self._onnx_model_has_target_linear_param(onnx_model, ref_weight.cpu().numpy())
        if ref_bias is not None:
            assert ref_bias.count_nonzero() > 0
            assert self._onnx_model_has_target_linear_param(onnx_model, ref_bias.cpu().numpy())

    def _onnx_model_has_target_linear_param(self, onnx_model: onnx.ModelProto, target_param: np.ndarray) -> bool:
        tensor_protos = []
        for item in onnx_model.graph.initializer:
            if isinstance(item, onnx.TensorProto):
                tensor_protos.append(item)
        for node in onnx_model.graph.node:
            for attribute in node.attribute:
                if attribute.HasField("t") and isinstance(attribute.t, onnx.TensorProto):
                    tensor_protos.append(attribute.t)
        for tensor_proto in tensor_protos:
            if (not tensor_proto.HasField("raw_data")) or tensor_proto.raw_data is None:
                continue
            onnx_weight = numpy_helper.to_array(tensor_proto)
            # linear weight may be transposed when do_constant_folding=True
            for weight in [onnx_weight, onnx_weight.T]:
                if weight.shape == target_param.shape and np.allclose(weight, target_param):
                    return True
        return False


class TestStateDict:
    def test_compressed_model_state_dict_follows_original_torch_model(self):
        linear_recipe = LinearRunRecipe().model_config_(bias=True)
        model = linear_recipe.model()
        original_state_dict = model.state_dict()
        ref_state_dict = {}
        for name, value in original_state_dict.items():
            ref_state_dict[f"{name}"] = value
            for keyword in ["weight", "bias"]:
                name_parts = name.split(".")
                if name_parts[-1] == keyword:
                    importance_name = ".".join([*name_parts[:-1], f"pre_ops.0.op.{keyword}_importance"])
                    ref_state_dict[importance_name] = torch.zeros_like(value, dtype=torch.float)
                    mask_name = ".".join([*name_parts[:-1], f"pre_ops.0.op.{keyword}_ctx._binary_mask"])
                    ref_state_dict[mask_name] = torch.ones_like(value, dtype=torch.float)

        _, compressed_model = create_compressed_model(model, linear_recipe.nncf_config(), dump_graphs=False)
        compressed_state_dict = compressed_model.state_dict()
        assert sorted(ref_state_dict) == sorted(compressed_state_dict)
        for name, ref_value in ref_state_dict.items():
            assert torch.allclose(compressed_state_dict[name], ref_value)

    def test_can_load_state_dict(self, tmp_path):
        recipe = LinearRunRecipe(log_dir=tmp_path)
        compression_ctrl, compressed_model = create_compressed_model(
            recipe.model(init_seed=0), recipe.nncf_config(), dump_graphs=False
        )
        for minfo in compression_ctrl.sparsified_module_info:
            initialize_sparsifier_parameters_by_linspace(minfo.operand)
            force_update_sparsifier_binary_masks_by_threshold(minfo.operand)
        state_dict = deepcopy(compressed_model.state_dict())
        # load state dict with torch api
        _, new_compressed_model = create_compressed_model(
            recipe.model(init_seed=1), recipe.nncf_config(), dump_graphs=False
        )
        new_compressed_model.load_state_dict(state_dict)
        PTTensorListComparator.check_equal(list(new_compressed_model.state_dict().values()), list(state_dict.values()))
        # load state dict with nncf api
        _, new_compressed_model = create_compressed_model(
            recipe.model(init_seed=2), recipe.nncf_config(), dump_graphs=False
        )
        assert load_state(new_compressed_model, state_dict, is_resume=True) == len(state_dict)
        PTTensorListComparator.check_equal(list(new_compressed_model.state_dict().values()), list(state_dict.values()))


class TestCompressionState:
    def test_can_load_compression_state(self, tmp_path):
        recipe = LinearRunRecipe(log_dir=tmp_path).scheduler_params_(steps_per_epoch=4)
        compression_ctrl, compressed_model = create_compressed_model(recipe.model(init_seed=1), recipe.nncf_config())
        trainer = build_compression_trainer(
            tmp_path,
            compression_ctrl,
            compressed_model,
            train_dataset=recipe.generate_mock_dataset(num_samples=16),
            batch_size=4,
            num_train_epochs=4,
        )
        trainer.train()
        ref_compression_state = compression_ctrl.get_compression_state()
        compression_ctrl, _ = create_compressed_model(
            recipe.model(init_seed=2), recipe.nncf_config(), compression_state=ref_compression_state
        )
        assert compression_ctrl.get_compression_state() == ref_compression_state

    @pytest.mark.parametrize("resume_step", [2, 7, 15, 17], ids=["epoch0", "epoch1", "epoch2_end", "epoch3"])
    @pytest.mark.parametrize("infer_steps_per_epoch", [True, False])
    @pytest.mark.parametrize("adaptive_init_threshold", [True, False])
    def test_can_resume_training_from_compression_state(
        self, tmp_path, resume_step: int, infer_steps_per_epoch: bool, adaptive_init_threshold: bool
    ):
        actual_steps_per_epoch = 5
        batch_size = 4
        num_train_epochs = 5
        recipe = BertRunRecipe(log_dir=tmp_path).model_config_(intermediate_size=6)
        recipe.scheduler_params_(
            warmup_start_epoch=1,
            warmup_end_epoch=3,
            steps_per_epoch=None if infer_steps_per_epoch else actual_steps_per_epoch,
            init_importance_threshold=None if adaptive_init_threshold else -0.01,
        )
        dataset = recipe.generate_mock_dataset(num_samples=batch_size * actual_steps_per_epoch)

        # train from beginning
        compression_ctrl, compressed_model = create_compressed_model(
            recipe.model(init_seed=1), recipe.nncf_config(), dump_graphs=False
        )
        trainer = build_compression_trainer(
            output_dir=tmp_path / "from_beginning",
            compression_ctrl=compression_ctrl,
            compressed_model=compressed_model,
            train_dataset=dataset,
            batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            save_steps=resume_step,
        )
        trainer.train()

        # resume training
        resumed_compression_ctrl, resumed_compressed_model = create_compressed_model(
            recipe.model(init_seed=2), recipe.nncf_config(), dump_graphs=False
        )
        resumed_trainer = build_compression_trainer(
            output_dir=tmp_path / "resumed",
            compression_ctrl=resumed_compression_ctrl,
            compressed_model=resumed_compressed_model,
            train_dataset=dataset,
            batch_size=batch_size,
            num_train_epochs=num_train_epochs,
        )
        resume_folder = Path(tmp_path, "from_beginning", f"{PREFIX_CHECKPOINT_DIR}-{resume_step}")
        resumed_trainer.train(str(resume_folder))

        PTTensorListComparator.check_equal(
            list(compressed_model.state_dict().values()),
            list(resumed_compressed_model.state_dict().values()),
            atol=0.001,
        )
