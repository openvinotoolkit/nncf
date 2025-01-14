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

from pathlib import Path
from typing import Tuple

import pytest
import torch
from huggingface_hub import model_info
from torch import nn
from transformers import AutoConfig

from tests.torch.models_hub_test.common import BaseTestModel
from tests.torch.models_hub_test.common import ExampleType
from tests.torch.models_hub_test.common import ModelInfo
from tests.torch.models_hub_test.common import get_model_params
from tests.torch.models_hub_test.common import idfn

MODEL_LIST_FILE = Path(__file__).parent / "hf_transformers_models.txt"


def filter_example(model: nn.Module, example: ExampleType) -> Tuple[nn.Module, ExampleType]:
    """
    Filter example of input by signature of the model.

    :param model: The model.
    :param example: Example of input.
    :return: Filtered example of input.
    """
    try:
        import inspect

        if isinstance(example, dict):
            model_params = inspect.signature(model.forward).parameters
            names_set = {p for p in model_params}
            new_example = dict()
            for k, v in example:
                if k in names_set:
                    new_example[k] = v
        return new_example
    except:  # noqa: E722
        return example


@pytest.fixture(scope="class", autouse=True)
def fixture_image(request):
    import requests
    from PIL import Image

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    request.cls.image = Image.open(requests.get(url, stream=True).raw)


class TestTransformersModel(BaseTestModel):
    def load_model(self, name: str) -> Tuple[nn.Module, ExampleType]:
        torch.manual_seed(0)

        mi = model_info(name)
        auto_processor = None
        model = None
        example = None
        try:
            auto_model = mi.transformersInfo["auto_model"]
            if "processor" in mi.transformersInfo:
                auto_processor = mi.transformersInfo["processor"]
        except:  # noqa: E722
            auto_model = None
        if "clip_vision_model" in mi.tags:
            from transformers import CLIPFeatureExtractor
            from transformers import CLIPVisionModel

            config = AutoConfig.from_pretrained(name, torchscript=True)
            model = CLIPVisionModel._from_config(config)
            preprocessor = CLIPFeatureExtractor.from_pretrained(name)
            encoded_input = preprocessor(self.image, return_tensors="pt")
            example = dict(encoded_input)
        elif "t5" in mi.tags:
            from transformers import T5Tokenizer

            tokenizer = T5Tokenizer.from_pretrained(name)
            encoder = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt")
            decoder = tokenizer("Studies show that", return_tensors="pt")
            example = (encoder.input_ids, encoder.attention_mask, decoder.input_ids, decoder.attention_mask)
        elif "hubert" in mi.tags:
            wav_input_16khz = torch.randn(1, 10000)
            example = (wav_input_16khz,)
        elif "vit-gpt2" in name:
            from transformers import VisionEncoderDecoderModel
            from transformers import ViTImageProcessor

            config = AutoConfig.from_pretrained(name, torchscript=True)
            model = VisionEncoderDecoderModel._from_config(config)
            feature_extractor = ViTImageProcessor.from_pretrained(name)
            encoded_input = feature_extractor(images=[self.image], return_tensors="pt")

            class VIT_GPT2_Model(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, x):
                    return self.model.generate(x, max_length=16, num_beams=4)

            model = VIT_GPT2_Model(model)
            example = (encoded_input.pixel_values,)
        elif "mms-lid" in name:
            # mms-lid model config does not have auto_model attribute, only direct loading available
            from transformers import AutoFeatureExtractor
            from transformers import Wav2Vec2ForSequenceClassification

            config = AutoConfig.from_pretrained(name, torchscript=True)
            model = Wav2Vec2ForSequenceClassification._from_config(config)
            processor = AutoFeatureExtractor.from_pretrained(name)
            input_values = processor(torch.randn(16000).numpy(), sampling_rate=16_000, return_tensors="pt")
            example = {"input_values": input_values.input_values}
        elif "retribert" in mi.tags:
            from transformers import RetriBertTokenizer

            text = "How many cats are there?"
            tokenizer = RetriBertTokenizer.from_pretrained(name)
            encoding1 = tokenizer("How many cats are there?", return_tensors="pt")
            encoding2 = tokenizer("Second text", return_tensors="pt")
            example = (encoding1.input_ids, encoding1.attention_mask, encoding2.input_ids, encoding2.attention_mask)
        elif "mgp-str" in mi.tags or "clip_vision_model" in mi.tags:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(name)
            encoded_input = processor(images=self.image, return_tensors="pt")
            example = (encoded_input.pixel_values,)
        elif "flava" in mi.tags:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(name)
            encoded_input = processor(
                text=["a photo of a cat", "a photo of a dog"], images=[self.image, self.image], return_tensors="pt"
            )
            example = dict(encoded_input)
        elif "vivit" in mi.tags:
            from transformers import VivitImageProcessor

            frames = list(torch.randint(0, 255, [32, 3, 224, 224]).to(torch.float32))
            processor = VivitImageProcessor.from_pretrained(name)
            encoded_input = processor(images=frames, return_tensors="pt")
            example = (encoded_input.pixel_values,)
        elif "tvlt" in mi.tags:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(name)
            num_frames = 8
            images = list(torch.rand(num_frames, 3, 224, 224))
            audio = list(torch.randn(10000))
            input_dict = processor(images, audio, sampling_rate=44100, return_tensors="pt")
            example = dict(input_dict)
        elif "gptsan-japanese" in mi.tags:
            from transformers import AutoTokenizer

            processor = AutoTokenizer.from_pretrained(name)
            text = "織田信長は、"
            encoded_input = processor(text=[text], return_tensors="pt")
            example = dict(input_ids=encoded_input.input_ids, token_type_ids=encoded_input.token_type_ids)
        elif "videomae" in mi.tags or "timesformer" in mi.tags:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(name)
            video = list(torch.randint(0, 255, [16, 3, 224, 224]).to(torch.float32))
            inputs = processor(video, return_tensors="pt")
            example = dict(inputs)
        else:
            try:
                if auto_model == "AutoModelForCausalLM":
                    from transformers import AutoModelForCausalLM
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(name)
                    config = AutoConfig.from_pretrained(name, torchscript=True)
                    model = AutoModelForCausalLM.from_config(config)
                    text = "Replace me by any text you'd like."
                    encoded_input = tokenizer(text, return_tensors="pt")
                    inputs_dict = dict(encoded_input)
                    if "facebook/incoder" in name and "token_type_ids" in inputs_dict:
                        del inputs_dict["token_type_ids"]
                    example = inputs_dict

                    # Unused input
                    if name == "RWKV/rwkv-4-169m-pile":
                        example.pop("attention_mask")

                elif auto_model == "AutoModelForMaskedLM":
                    from transformers import AutoModelForMaskedLM
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(name)
                    config = AutoConfig.from_pretrained(name, torchscript=True)
                    model = AutoModelForMaskedLM.from_config(config)
                    text = "Replace me by any text you'd like."
                    encoded_input = tokenizer(text, return_tensors="pt")
                    example = dict(encoded_input)
                elif auto_model == "AutoModelForImageClassification":
                    from transformers import AutoModelForImageClassification
                    from transformers import AutoProcessor

                    processor = AutoProcessor.from_pretrained(name)
                    config = AutoConfig.from_pretrained(name, torchscript=True)
                    model = AutoModelForImageClassification.from_config(config)
                    encoded_input = processor(images=self.image, return_tensors="pt")
                    example = dict(encoded_input)
                elif auto_model == "AutoModelForSeq2SeqLM":
                    from transformers import AutoModelForSeq2SeqLM
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(name)
                    config = AutoConfig.from_pretrained(name, torchscript=True)
                    model = AutoModelForSeq2SeqLM.from_config(config)
                    inputs = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt")
                    decoder_inputs = tokenizer(
                        "<pad> Studien haben gezeigt dass es hilfreich ist einen Hund zu besitzen",
                        return_tensors="pt",
                        add_special_tokens=False,
                    )
                    example = dict(input_ids=inputs.input_ids, decoder_input_ids=decoder_inputs.input_ids)
                elif auto_model == "AutoModelForSpeechSeq2Seq":
                    from transformers import AutoModelForSpeechSeq2Seq
                    from transformers import AutoProcessor

                    processor = AutoProcessor.from_pretrained(name)
                    config = AutoConfig.from_pretrained(name, torchscript=True)
                    model = AutoModelForSpeechSeq2Seq.from_config(config)
                    inputs = processor(torch.randn(1000).numpy(), sampling_rate=16000, return_tensors="pt")
                    example = dict(inputs)
                elif auto_model == "AutoModelForCTC":
                    from transformers import AutoModelForCTC
                    from transformers import AutoProcessor

                    processor = AutoProcessor.from_pretrained(name)
                    config = AutoConfig.from_pretrained(name, torchscript=True)
                    model = AutoModelForCTC.from_config(config)
                    input_values = processor(torch.randn(1000).numpy(), return_tensors="pt")
                    example = dict(input_values)
                elif auto_model == "AutoModelForTableQuestionAnswering":
                    import pandas as pd
                    from transformers import AutoModelForTableQuestionAnswering
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(name)
                    config = AutoConfig.from_pretrained(name, torchscript=True)
                    model = AutoModelForTableQuestionAnswering.from_config(config)
                    data = {
                        "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
                        "Number of movies": ["87", "53", "69"],
                    }
                    queries = [
                        "What is the name of the first actor?",
                        "How many movies has George Clooney played in?",
                        "What is the total number of movies?",
                    ]
                    answer_coordinates = [[(0, 0)], [(2, 1)], [(0, 1), (1, 1), (2, 1)]]
                    answer_text = [["Brad Pitt"], ["69"], ["209"]]
                    table = pd.DataFrame.from_dict(data)
                    encoded_input = tokenizer(
                        table=table,
                        queries=queries,
                        answer_coordinates=answer_coordinates,
                        answer_text=answer_text,
                        padding="max_length",
                        return_tensors="pt",
                    )
                    example = dict(
                        input_ids=encoded_input["input_ids"],
                        token_type_ids=encoded_input["token_type_ids"],
                        attention_mask=encoded_input["attention_mask"],
                    )
                else:
                    from transformers import AutoProcessor
                    from transformers import AutoTokenizer

                    text = "Replace me by any text you'd like."
                    if auto_processor is not None and "Tokenizer" not in auto_processor:
                        processor = AutoProcessor.from_pretrained(name)
                        encoded_input = processor(text=[text], images=self.image, return_tensors="pt", padding=True)
                    else:
                        tokenizer = AutoTokenizer.from_pretrained(name)
                        encoded_input = tokenizer(text, return_tensors="pt")
                    example = dict(encoded_input)
                    # Input does not used by config parameter config.type_vocab_size=0
                    if name in [
                        "regisss/bridgetower-newyorker-a100-8x",
                        "facebook/mask2former-swin-base-coco-panoptic",
                        "facebook/maskformer-swin-base-coco",
                    ]:
                        example.pop("pixel_mask")
                    elif name in ["MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", "microsoft/deberta-base"]:
                        example.pop("token_type_ids")
            except:  # noqa: E722
                pass
        if model is None:
            from transformers import AutoModel

            config = AutoConfig.from_pretrained(name, torchscript=True)
            model = AutoModel.from_config(config)
        if hasattr(model, "set_default_language"):
            model.set_default_language("en_XX")
        if example is None:
            if "encodec" in mi.tags:
                example = (torch.randn(1, 1, 100),)
            else:
                example = (torch.randint(1, 1000, [1, 100]),)
        example = filter_example(model, example)
        model.eval()
        # Uncomment to check inference original model
        # if isinstance(example, dict):
        #     model(**example)
        # else:
        #     model(*example)
        return model, example

    @pytest.mark.parametrize("model_info", get_model_params(MODEL_LIST_FILE), ids=idfn)
    def test_nncf_wrap(self, model_info: ModelInfo):
        self.nncf_wrap(model_info.model_name)
