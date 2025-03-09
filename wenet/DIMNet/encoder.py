from typing import Optional, Tuple, List

import torch
import torch.utils.checkpoint as ckpt

from wenet.transformer.convolution import ConvolutionModule
from wenet.transformer.encoder_layer import (
    ConformerEncoderLayer,
    TransformerEncoderLayer,
)
from wenet.transformer.encoder import BaseEncoder
from wenet.utils.class_utils import (
    WENET_ACTIVATION_CLASSES,
    WENET_ATTENTION_CLASSES,
    WENET_MLP_CLASSES,
)
from wenet.utils.mask import make_pad_mask
from wenet.utils.mask import add_optional_chunk_mask
from wenet.utils.common import mask_to_bias


class SharedEncoder(BaseEncoder):

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "rel_pos",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        positionwise_conv_kernel_size: int = 1,
        macaron_style: bool = True,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        query_bias: bool = True,
        key_bias: bool = True,
        value_bias: bool = True,
        conv_bias: bool = True,
        gradient_checkpointing: bool = False,
        use_sdpa: bool = False,
        layer_norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        n_kv_head: Optional[int] = None,
        head_dim: Optional[int] = None,
        mlp_type: str = "position_wise_feed_forward",
        mlp_bias: bool = True,
        n_expert: int = 8,
        n_expert_activated: int = 2,
    ):
        super().__init__(
            input_size,
            output_size,
            attention_heads,
            linear_units,
            num_blocks,
            dropout_rate,
            positional_dropout_rate,
            attention_dropout_rate,
            input_layer,
            pos_enc_layer_type,
            normalize_before,
            static_chunk_size,
            use_dynamic_chunk,
            global_cmvn,
            use_dynamic_left_chunk,
            gradient_checkpointing,
            use_sdpa,
            layer_norm_type,
            norm_eps,
        )
        activation = WENET_ACTIVATION_CLASSES[activation_type]()

        # self-attention module definition
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
            query_bias,
            key_bias,
            value_bias,
            use_sdpa,
            n_kv_head,
            head_dim,
        )
        # feed-forward module definition
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
            mlp_bias,
            n_expert,
            n_expert_activated,
        )
        # convolution module definition
        convolution_layer_args = (
            output_size,
            cnn_module_kernel,
            activation,
            cnn_module_norm,
            causal,
            conv_bias,
        )

        mlp_class = WENET_MLP_CLASSES[mlp_type]
        self.encoders = torch.nn.ModuleList(
            [
                ConformerEncoderLayer(
                    output_size,
                    WENET_ATTENTION_CLASSES[selfattention_layer_type](
                        *encoder_selfattn_layer_args
                    ),
                    mlp_class(*positionwise_layer_args),
                    mlp_class(*positionwise_layer_args) if macaron_style else None,
                    (
                        ConvolutionModule(*convolution_layer_args)
                        if use_cnn_module
                        else None
                    ),
                    dropout_rate,
                    normalize_before,
                    layer_norm_type=layer_norm_type,
                    norm_eps=norm_eps,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            xs: padded input tensor (B, T, D)
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: torch.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        """
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks  # (B, 1, T/subsample_rate)
        chunk_masks = add_optional_chunk_mask(
            xs,
            masks,
            self.use_dynamic_chunk,
            self.use_dynamic_left_chunk,
            decoding_chunk_size,
            self.static_chunk_size,
            num_decoding_left_chunks,
            # Since we allow up to 1s(100 frames) delay, the maximum
            # chunk_size is 100 / 4 = 25.
            max_chunk_size=int(100.0 / self.embed.subsampling_rate),
        )
        if self.use_sdpa:
            chunk_masks = mask_to_bias(chunk_masks, xs.dtype)
        if self.gradient_checkpointing and self.training:
            xs, layer_feats = self.forward_layers_checkpointed(
                xs, chunk_masks, pos_emb, mask_pad
            )
        else:
            xs, layer_feats = self.forward_layers(xs, chunk_masks, pos_emb, mask_pad)
        if self.normalize_before:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks, layer_feats

    def forward_layers(
        self,
        xs: torch.Tensor,
        chunk_masks: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        layer_feats = torch.zeros_like(xs)
        for idx, layer in enumerate(self.encoders):
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
            layer_feats += xs
        return xs, layer_feats

    @torch.jit.unused
    def forward_layers_checkpointed(
        self,
        xs: torch.Tensor,
        chunk_masks: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor,
    ) -> torch.Tensor:
        layer_feats = torch.zeros_like(xs)
        for idx, layer in enumerate(self.encoders):
            xs, chunk_masks, _, _ = ckpt.checkpoint(
                layer.__call__, xs, chunk_masks, pos_emb, mask_pad, use_reentrant=False
            )
            layer_feats += xs
        return xs, layer_feats
