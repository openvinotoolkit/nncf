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

# This logic is largely copied from the https://github.com/Theia-4869/CDPruner/tree/main
import numpy as np
import torch
import torch.nn.functional as F


def get_visual_similarity(image_features):
    """
    Compute the cosine similarity matrix among image features.
    """
    image_features = image_features.float()  # (B, N, D)
    image_normalized = image_features / image_features.norm(dim=-1, keepdim=True)  # (B, N, D)
    similarity = torch.matmul(image_normalized, image_normalized.transpose(1, 2))  # (B, N, N)
    return similarity


def get_relevance_score(image_embeds, text_embeds):
    """
    Compute the relevance score between image and text embeddings.
    """
    image_embeds = image_embeds.float()
    text_embeds = text_embeds.float()
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)  # (B, N, C)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)  # (M, C)

    relevance = torch.matmul(image_embeds, text_embeds.t())  # (B, N, M)
    relevance = (-relevance).mean(dim=-1)  # (B, N)
    relevance = (relevance - relevance.min(dim=1, keepdim=True)[0]) / (
        relevance.max(dim=1, keepdim=True)[0] - relevance.min(dim=1, keepdim=True)[0] + 1e-6
    )
    return relevance


def build_conditional_kernel_matrix(relevance, similarity, theta=0.5):
    """
    Build the conditional DPP kernel matrix based on relevance and visual similarity.
    """
    if theta != 1:
        alpha = theta / (2 * (1 - theta))
        relevance = torch.exp(alpha * relevance)  # (B, N)

    kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1)  # (B, N, N)
    return kernel


def conditional_dpp_map(kernel, num_keep_tokens):
    """
    Perform conditional DPP MAP inference to select a subset of tokens.
    """
    device = kernel.device

    # kernel diagonal (di2s[b, i] = kernel[b, i, i] = relevance[b, i] ** 2 * (L[i,i]=1))
    di2s = torch.diagonal(kernel, dim1=1, dim2=2).clone()

    # orthogonal directions corresponding to selected tokens (L~=CC^T)
    B, N = di2s.shape
    cis = torch.zeros((num_keep_tokens, B, N), device=device)  # (num_keep_tokens, B, N)

    keep_indices = torch.empty((num_keep_tokens, B), dtype=torch.long, device=device)
    batch_idx = torch.arange(B)
    for i in range(num_keep_tokens):
        j = torch.argmax(di2s, dim=-1)  # Select the index with highest remaining score
        keep_indices[i] = j

        # compute the orthogonalized row vector for token j
        if i == 0:
            eis = kernel[batch_idx, j] / torch.sqrt(kernel[batch_idx, j, j].unsqueeze(1) + 1e-5)  # (B, N)
        else:
            proj = torch.einsum("tb,tbn->bn", cis[:i, batch_idx, j], cis[:i])  # (B, N)
            eis = (kernel[batch_idx, j] - proj) / torch.sqrt(di2s[batch_idx, j].unsqueeze(-1) + 1e-5)

        cis[i, :, :] = eis
        di2s -= eis**2
        di2s[batch_idx, j] = -float("inf")

    keep_indices = torch.sort(keep_indices.t()).values
    return keep_indices


def get_model_kwargs(model, inputs):
    """
    Get the model keyword arguments from the model and inputs.
    """
    kwargs = {}
    if hasattr(model.config, "vision_feature_select_strategy"):
        kwargs["vision_feature_select_strategy"] = model.config.vision_feature_select_strategy
    if hasattr(model.config, "vision_feature_layer"):
        kwargs["vision_feature_layer"] = model.config.vision_feature_layer
    if hasattr(inputs, "image_sizes"):
        kwargs["image_sizes"] = inputs.image_sizes
    if hasattr(inputs, "image_grid_thw"):
        kwargs["image_grid_thw"] = inputs.image_grid_thw
    return kwargs


def get_image_features(model, inputs, **kwargs):
    """
    Extract image features from the model.
    """
    pixel_values = inputs.pixel_values
    image_num_patches = None
    if "LlavaNextForConditionalGeneration" in model.config.architectures and pixel_values.dim() == 5:
        from transformers.models.llava_next.modeling_llava_next import image_size_to_num_patches

        image_num_patches = [
            image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=model.config.image_grid_pinpoints,
                patch_size=model.config.vision_config.image_size,
            )
            for imsize in inputs.image_sizes
        ]
        # stacked if input is (batch_size, num_patches, num_channels, height, width)
        _pixel_values_list = [pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)]
        pixel_values = torch.cat(_pixel_values_list, dim=0)

    if hasattr(model, "vision_tower"):  # llava or internvl
        vision_feature_layer = kwargs.get("vision_feature_layer", -1)
        image_features = model.vision_tower(pixel_values=pixel_values, output_hidden_states=True)  # .last_hidden_state
        image_features = image_features.hidden_states[vision_feature_layer]

        if kwargs.get("vision_feature_select_strategy") == "default":
            image_features = image_features[:, 1:]  # remove CLS token
        if image_num_patches is None:  # LlavaForConditionalGeneration
            image_num_patches = image_features.shape[0]
    elif hasattr(model, "visual") or hasattr(model, "vision_tower"):  # qwen
        image_features = model.visual.patch_embed(pixel_values)
        image_num_patches = (kwargs["image_grid_thw"].prod(-1)).tolist()
    else:
        error_msg = "Unsupported visual model"
        raise NotImplementedError(error_msg)

    image_features = torch.split(image_features, image_num_patches, dim=0)

    if "LlavaNextForConditionalGeneration" in model.config.architectures:
        embed_std = 1 / np.sqrt(model.config.text_config.hidden_size)
        image_newline = torch.Tensor(
            torch.randn(image_features[0].shape[-1], dtype=image_features[0].dtype) * embed_std
        ).to(model.device)
        image_features, _ = model.pack_image_features(
            image_features,
            inputs.image_sizes,
            vision_feature_select_strategy=kwargs.get("vision_feature_select_strategy"),
            image_newline=image_newline,
        )
    elif (
        "Qwen2_5_VLForConditionalGeneration" in model.config.architectures
        or "Qwen2VLForConditionalGeneration" in model.config.architectures
    ):
        spatial_merge_size = model.visual.config.spatial_merge_size
        pooled_image_features = []
        for img_feat, (t, h, w) in zip(image_features, kwargs["image_grid_thw"]):
            num_patches, d = img_feat.shape
            assert t == 1, "Only single-frame temporal dimension supported"
            assert h * w == num_patches, f"H*W != num_patches: {h}*{w} != {num_patches}"

            # Reshape to [1, D, H, W]
            x = img_feat.view(h, w, d).permute(2, 0, 1).unsqueeze(0)

            # Apply avg pooling
            x_pooled = F.avg_pool2d(x, kernel_size=spatial_merge_size, stride=spatial_merge_size)

            # Reshape back to [num_pooled_patches, D]
            pooled = x_pooled.squeeze(0).permute(1, 2, 0).reshape(-1, d)
            pooled_image_features.append(pooled)
        image_features = pooled_image_features

    if image_features[0].dim() < 3:
        image_features = [feat_i.unsqueeze(0) for feat_i in image_features]
    return image_features


def get_cdpruner_mask(image_embeds, image_features, text_embeds, special_image_mask, num_keep_tokens, theta):
    """
    Generate a mask to retain image tokens based on fast MAP inference using Conditional DPP for token selection.
    """
    keep_indices = []
    offset = 0
    # Compute keep_indices for each image embedding
    for emb_i, feat_i in zip(image_embeds, image_features):
        rel_i = get_relevance_score(emb_i.unsqueeze(0), text_embeds)
        sim_i = get_visual_similarity(feat_i)
        kernel_i = build_conditional_kernel_matrix(rel_i, sim_i, theta)
        keep_i = conditional_dpp_map(kernel_i, num_keep_tokens)[0] + offset
        keep_indices.append(keep_i)
        offset += emb_i.shape[0]

    keep_indices = torch.cat(keep_indices, dim=0)

    # Get the positions of the selected image tokens
    image_token_positions = torch.nonzero(special_image_mask[0], as_tuple=False).squeeze(1)
    kept_positions = image_token_positions[keep_indices]

    # Build mask to keep: original text + selected image tokens
    kept_mask = ~special_image_mask
    kept_mask[0, kept_positions] = True

    return kept_mask


def get_inputs_embeds(model, inputs, num_keep_tokens=None, theta=0.5):
    """
    Get the input embeddings with optional CDPruner-based token pruning.
    """
    kwargs = get_model_kwargs(model, inputs)
    inputs_embeds = model.get_input_embeddings()(inputs.input_ids)
    B, _, emb_dim = inputs_embeds.shape
    assert B == 1

    special_image_mask = inputs_embeds == model.get_input_embeddings()(
        torch.tensor(model.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
    )  # (B, seq_len, emb_dim)
    special_image_mask = special_image_mask.all(-1)

    # Get mapped image features into the language embedding space
    image_embeds = model.get_image_features(
        pixel_values=inputs.pixel_values,
        **kwargs,
    )

    flat_image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
    exp_special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
    inputs_embeds_with_images = inputs_embeds.masked_scatter(exp_special_image_mask, flat_image_embeds)

    if num_keep_tokens is None:
        return inputs_embeds_with_images

    # Prune image tokens
    text_embeds = inputs_embeds[~special_image_mask].view(-1, emb_dim)
    image_features = get_image_features(model, inputs, **kwargs)
    kept_mask = get_cdpruner_mask(image_embeds, image_features, text_embeds, special_image_mask, num_keep_tokens, theta)

    # Apply mask to inputs_embeds directly
    kept_mask = kept_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
    pruned_embeds = inputs_embeds_with_images[kept_mask].view(B, -1, emb_dim)
    return pruned_embeds
