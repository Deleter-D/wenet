from typing import Optional, Tuple

import torch

from wenet.transformer.convolution import ConvolutionModule
from wenet.transformer.encoder_layer import ConformerEncoderLayer
from wenet.transformer.encoder import BaseEncoder
from wenet.utils.class_utils import (
    WENET_ACTIVATION_CLASSES,
    WENET_ATTENTION_CLASSES,
    WENET_MLP_CLASSES,
)
from wenet.utils.mask import make_pad_mask, add_optional_chunk_mask
from wenet.utils.common import mask_to_bias


class ExpressionHabitModule(BaseEncoder):
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
        num_classes: int = 30,
    ):
        """Construct ConformerEncoder

        Args:
            input_size to use_dynamic_chunk, see in BaseEncoder
            positionwise_conv_kernel_size (int): Kernel size of positionwise
                conv1d layer.
            macaron_style (bool): Whether to use macaron style for
                positionwise layer.
            selfattention_layer_type (str): Encoder attention layer type,
                the parameter has no effect now, it's just for configure
                compatibility.
            activation_type (str): Encoder activation function type.
            use_cnn_module (bool): Whether to use convolution module.
            cnn_module_kernel (int): Kernel size of convolution module.
            causal (bool): whether to use causal convolution or not.
            key_bias: whether use bias in attention.linear_k, False for whisper models.
        """
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

        # Linears
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(output_size, output_size // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(output_size // 2, output_size),
        )

        # Classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(output_size, output_size // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(output_size // 2, num_classes),
        )

        self.pooling = torch.nn.AdaptiveAvgPool1d(1)

        # Loss function
        weights = torch.Tensor(
            [
                0.0001443418013856813,
                0.00010340192327577293,
                9.635767970707265e-05,
                9.100837277029486e-05,
                9.258401999814833e-05,
                6.893699158968702e-05,
                5.093465084296847e-05,
                4.409948844593403e-05,
                3.898939488459139e-05,
                3.5300762496469926e-05,
                9.09008271975275e-05,
                9.255831173639393e-05,
                9.604302727621975e-05,
                9.918666931164451e-05,
                0.00010554089709762533,
                4.777830864787386e-05,
                4.1533413631266354e-05,
                4.1411297001822096e-05,
                4.2645741822679005e-05,
                4.442667377493447e-05,
                3.373022565520963e-05,
                4.844257133168629e-05,
                5.184033177812338e-05,
                5.4650781506175536e-05,
                5.352745958676801e-05,
                3.8887808671981336e-05,
                4.238186056367875e-05,
                4.526115687516973e-05,
                4.933885928557332e-05,
                5.274818018778352e-05,
            ]
        )
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        labels: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for ExpressionHabitModule

        Args:
            xs: padded input tensor (B, T, D)
            xs_lens: input length (B)
            labels (torch.Tensor, optional): Ground truth labels of shape (B).
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks

        Returns:
            expression_habit_feats (torch.Tensor): The bimodal features of dialect and text (B, T, C).
            loss (torch.Tensor, optional): Computed loss if labels are provided.
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
            xs = self.forward_layers_checkpointed(xs, chunk_masks, pos_emb, mask_pad)
        else:
            xs = self.forward_layers(xs, chunk_masks, pos_emb, mask_pad)
        if self.normalize_before:
            xs = self.after_norm(xs)

        expression_habit_feats = self.linear(xs)

        xs = xs.permute(0, 2, 1)  # (B, D, T)
        xs = self.pooling(xs).squeeze(-1)  # (B, D)
        logits = self.classifier(xs)

        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return expression_habit_feats, loss

    def forward_expression_habit(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
    ) -> torch.Tensor:
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
            xs = self.forward_layers_checkpointed(xs, chunk_masks, pos_emb, mask_pad)
        else:
            xs = self.forward_layers(xs, chunk_masks, pos_emb, mask_pad)
        if self.normalize_before:
            xs = self.after_norm(xs)

        expression_habit_feats = self.linear(xs)

        return expression_habit_feats
