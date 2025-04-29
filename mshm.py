from functools import partial
from typing import Optional, Tuple
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from monai.networks.layers.utils import get_rel_pos_embedding_layer
from monai.utils import pytorch_after
from mamba_ssm import Mamba2
from torchinfo import summary
import collections.abc
from itertools import repeat
from monai.networks.layers.factories import Pool, Conv
from ptflops import get_model_complexity_info


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ConvStem(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_dims):
        super(ConvStem, self).__init__()
        conv_type: type[nn.Conv1d | nn.Conv2d | nn.Conv3d] = Conv[
            Conv.CONV, spatial_dims
        ]

        self.conv = nn.Sequential(
            conv_type(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
            conv_type(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        return self.conv(x)


class MambaLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        expand = 2
        headdim = dim * expand // 8
        self.mamba = Mamba2(
            d_model=dim,  # Model dimension d_model
            d_state=128,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
            headdim=headdim,
        )

    def forward(self, x):
        B, C = x.shape[:2]
        residual = x
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_mamba = self.mamba(x_flat)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        out += residual
        return out


class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels, scales=[3, 5, 7], spatial_dims=3):
        super(MultiScaleConvBlock, self).__init__()

        conv_type: type[nn.Conv1d | nn.Conv2d | nn.Conv3d] = Conv[
            Conv.CONV, spatial_dims
        ]

        # Multi-scale convolutions
        self.multi_scale_convs = nn.ModuleList(
            [
                nn.Sequential(
                    conv_type(
                        in_channels,
                        in_channels // 2,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
                    nn.GroupNorm(32, in_channels // 2),
                    nn.GELU(),
                    # Multi-scale convolution (3x3, 5x5, etc.)
                    conv_type(
                        in_channels // 2,
                        in_channels // 2,
                        kernel_size=s,
                        stride=1,
                        padding=s // 2,
                        bias=False,
                    ),
                    nn.GroupNorm(32, in_channels // 2),
                    nn.GELU(),
                    # 1x1 convolution to restore original channels
                    conv_type(
                        in_channels // 2,
                        in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
                    nn.GroupNorm(32, in_channels),
                    nn.GELU(),
                )
                for s in scales
            ]
        )

        # Fusion convolution to combine multi-scale features
        self.fusion_conv = nn.Sequential(
            conv_type(in_channels * len(scales), in_channels, kernel_size=1),
            nn.GroupNorm(32, in_channels),
            nn.GELU(),
        )

    def forward(self, x):
        residual = x

        # Apply all multi-scale convolutions and concatenate the features
        features = [conv(x) for conv in self.multi_scale_convs]
        fused = torch.cat(features, dim=1)  # Concatenate along the channel dimension

        # Fusion layer to reduce the combined features back to the original channels
        fused = self.fusion_conv(fused)

        # Add residual connection
        fused = fused + residual

        return fused


class MSFEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scales=[3, 5, 7],
        conv=True,
        spatial_dims=3,
        num_blocks=2,
    ):
        super(MSFEncoder, self).__init__()

        conv_type: type[nn.Conv1d | nn.Conv2d | nn.Conv3d] = Conv[
            Conv.CONV, spatial_dims
        ]
        self.out_channels = out_channels
        self.conv = conv
        if self.conv:
            # Stack of MultiScaleConvBlocks
            self.blocks = nn.ModuleList(
                [
                    MultiScaleConvBlock(
                        in_channels=in_channels,
                        scales=scales,
                        spatial_dims=spatial_dims,
                    )
                    for i in range(num_blocks)
                ]
            )
        else:
            self.blocks = nn.Sequential(MambaLayer(dim=in_channels))

        if num_blocks == 0:
            self.blocks = nn.Sequential()

        # Downsampling layer
        self.downsample = nn.Sequential(
            conv_type(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        # Apply the stacked blocks
        for block in self.blocks:
            x = block(x)

        # Apply downsampling
        downscaled = self.downsample(x)

        return downscaled


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, spatial_dims, mamba_encoder=True):
        super(FeatureExtractor, self).__init__()
        out_channels_1 = in_channels * 2**1
        out_channels_2 = in_channels * 2**2
        out_channels_3 = in_channels * 2**3
        out_channels_4 = in_channels * 2**4

        self.msf_encoder1 = MSFEncoder(
            in_channels=in_channels,
            out_channels=out_channels_1,
            conv=True,
            spatial_dims=spatial_dims,
            num_blocks=2,
        )
        self.msf_encoder2 = MSFEncoder(
            in_channels=out_channels_1,
            out_channels=out_channels_2,
            conv=True,
            spatial_dims=spatial_dims,
            num_blocks=2,
        )
        self.msf_encoder3 = MSFEncoder(
            in_channels=out_channels_2,
            out_channels=out_channels_3,
            conv=not mamba_encoder,
            spatial_dims=spatial_dims,
            num_blocks=2 if mamba_encoder else 0,
        )
        self.msf_encoder4 = MSFEncoder(
            in_channels=out_channels_3,
            out_channels=out_channels_4,
            conv=not mamba_encoder,
            spatial_dims=spatial_dims,
            num_blocks=2 if mamba_encoder else 0,
        )
        block_avgpool = [0, 1, (1, 1), (1, 1, 1)]
        avgp_type: type[
            nn.AdaptiveAvgPool1d | nn.AdaptiveAvgPool2d | nn.AdaptiveAvgPool3d
        ] = Pool[Pool.ADAPTIVEAVG, spatial_dims]
        self.pooling = avgp_type(block_avgpool[spatial_dims])
        conv_type: type[nn.Conv1d | nn.Conv2d | nn.Conv3d] = Conv[
            Conv.CONV, spatial_dims
        ]
        self.shortcut_convs = nn.ModuleList(
            [
                conv_type(
                    in_channels,
                    out_channels_1,
                    kernel_size=1,
                    stride=2,
                    padding=0,
                    bias=False,
                ),
                conv_type(
                    out_channels_1,
                    out_channels_2,
                    kernel_size=1,
                    stride=2,
                    padding=0,
                    bias=False,
                ),
                conv_type(
                    out_channels_2,
                    out_channels_3,
                    kernel_size=1,
                    stride=2,
                    padding=0,
                    bias=False,
                ),
                conv_type(
                    out_channels_3,
                    out_channels_4,
                    kernel_size=1,
                    stride=2,
                    padding=0,
                    bias=False,
                ),
            ]
        )

    def forward(self, x):
        # Layer 1
        residual = x
        x = self.msf_encoder1(x)
        residual = self.shortcut_convs[0](residual)  # Downsample residual
        x = x + residual

        # Layer 2
        residual = x
        x = self.msf_encoder2(x)
        residual = self.shortcut_convs[1](residual)  # Downsample residual
        x = x + residual

        # Layer 3
        residual = x
        x = self.msf_encoder3(x)
        residual = self.shortcut_convs[2](residual)  # Downsample residual
        x = x + residual

        # Layer 4
        residual = x
        x = self.msf_encoder4(x)
        residual = self.shortcut_convs[3](residual)  # Downsample residual
        x = x + residual

        return x


class CrossModalAttention(nn.Module):
    """
    A cross-attention block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    One can setup relative positional embedding as described in <https://arxiv.org/abs/2112.01526>
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        hidden_input_size: int | None = None,
        context_input_size: int | None = None,
        dim_head: int | None = None,
        qkv_bias: bool = False,
        save_attn: bool = False,
        causal: bool = False,
        sequence_length: int | None = None,
        rel_pos_embedding: Optional[str] = None,
        input_size: Optional[Tuple] = None,
        attention_dtype: Optional[torch.dtype] = None,
        use_flash_attention: bool = False,
    ) -> None:
        """
        Args:
            hidden_size (int): dimension of hidden layer.
            num_heads (int): number of attention heads.
            dropout_rate (float, optional): fraction of the input units to drop. Defaults to 0.0.
            hidden_input_size (int, optional): dimension of the input tensor. Defaults to hidden_size.
            context_input_size (int, optional): dimension of the context tensor. Defaults to hidden_size.
            dim_head (int, optional): dimension of each head. Defaults to hidden_size // num_heads.
            qkv_bias (bool, optional): bias term for the qkv linear layer. Defaults to False.
            save_attn (bool, optional): to make accessible the attention matrix. Defaults to False.
            causal (bool, optional): whether to use causal attention.
            sequence_length (int, optional): if causal is True, it is necessary to specify the sequence length.
            rel_pos_embedding (str, optional): Add relative positional embeddings to the attention map. For now only
                "decomposed" is supported (see https://arxiv.org/abs/2112.01526). 2D and 3D are supported.
            input_size (tuple(spatial_dim), optional): Input resolution for calculating the relative positional
                parameter size.
            attention_dtype: cast attention operations to this dtype.
            use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
                (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if dim_head:
            inner_size = num_heads * dim_head
            self.head_dim = dim_head
        else:
            if hidden_size % num_heads != 0:
                raise ValueError("hidden size should be divisible by num_heads.")
            inner_size = hidden_size
            self.head_dim = hidden_size // num_heads

        if causal and sequence_length is None:
            raise ValueError("sequence_length is necessary for causal attention.")

        if use_flash_attention and not pytorch_after(minor=13, major=1, patch=0):
            raise ValueError(
                "use_flash_attention is only supported for PyTorch versions >= 2.0."
                "Upgrade your PyTorch or set the flag to False."
            )
        if use_flash_attention and save_attn:
            raise ValueError(
                "save_attn has been set to True, but use_flash_attention is also set"
                "to True. save_attn can only be used if use_flash_attention is False"
            )

        if use_flash_attention and rel_pos_embedding is not None:
            raise ValueError(
                "rel_pos_embedding must be None if you are using flash_attention."
            )

        self.num_heads = num_heads
        self.hidden_input_size = hidden_input_size if hidden_input_size else hidden_size
        self.context_input_size = (
            context_input_size if context_input_size else hidden_size
        )
        self.out_proj = nn.Linear(inner_size, self.hidden_input_size)
        # key, query, value projections
        self.to_q = nn.Linear(self.hidden_input_size, inner_size, bias=qkv_bias)
        self.to_k = nn.Linear(self.context_input_size, inner_size, bias=qkv_bias)
        self.to_v = nn.Linear(self.context_input_size, inner_size, bias=qkv_bias)
        self.input_rearrange = Rearrange("b h (l d) -> b l h d", l=num_heads)

        self.out_rearrange = Rearrange("b l h d -> b h (l d)")
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.dropout_rate = dropout_rate

        self.scale = self.head_dim**-0.5
        self.save_attn = save_attn
        self.attention_dtype = attention_dtype

        self.causal = causal
        self.sequence_length = sequence_length
        self.use_flash_attention = use_flash_attention

        if causal and sequence_length is not None:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(sequence_length, sequence_length)).view(
                    1, 1, sequence_length, sequence_length
                ),
            )
            self.causal_mask: torch.Tensor
        else:
            self.causal_mask = torch.Tensor()

        self.att_mat = torch.Tensor()
        self.rel_positional_embedding = (
            get_rel_pos_embedding_layer(
                rel_pos_embedding, input_size, self.head_dim, self.num_heads
            )
            if rel_pos_embedding is not None
            else None
        )
        self.input_size = input_size

        mlp_ratio = 4
        dropout_rate = 0
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            drop=dropout_rate,
            norm_layer=nn.LayerNorm,
        )

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None):
        """
        Args:
            x (torch.Tensor): input tensor. B x (s_dim_1 * ... * s_dim_n) x C
            context (torch.Tensor, optional): context tensor. B x (s_dim_1 * ... * s_dim_n) x C

        Return:
            torch.Tensor: B x (s_dim_1 * ... * s_dim_n) x C
        """
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        b, t, c = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (hidden_size)

        q = self.input_rearrange(self.to_q(x))
        kv = context if context is not None else x
        _, kv_t, _ = kv.size()
        k = self.input_rearrange(self.to_k(kv))
        v = self.input_rearrange(self.to_v(kv))

        if self.attention_dtype is not None:
            q = q.to(self.attention_dtype)
            k = k.to(self.attention_dtype)

        if self.use_flash_attention:
            x = torch.nn.functional.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                scale=self.scale,
                dropout_p=self.dropout_rate,
                is_causal=self.causal,
            )
        else:
            att_mat = torch.einsum("blxd,blyd->blxy", q, k) * self.scale
            # apply relative positional embedding if defined
            if self.rel_positional_embedding is not None:
                att_mat = self.rel_positional_embedding(x, att_mat, q)

            if self.causal:
                att_mat = att_mat.masked_fill(
                    self.causal_mask[:, :, :t, :kv_t] == 0, float("-inf")
                )

            att_mat = att_mat.softmax(dim=-1)

            if self.save_attn:
                # no gradients and new tensor;
                # https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html
                self.att_mat = att_mat.detach()

            att_mat = self.drop_weights(att_mat)
            x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)

        x = self.out_rearrange(x)
        x = self.out_proj(x)
        x = self.drop_output(x)
        x = self.mlp(x)
        return x


class MambaFusionBlock(nn.Module):
    def __init__(
        self,
        dim,
    ):
        super(MambaFusionBlock, self).__init__()
        # self.mamba = Mamba(
        #     d_model=dim,  # Model dimension d_model
        #     d_state=8,  # SSM state expansion factor
        #     d_conv=2,  # Local convolution width
        #     expand=2,  # Block expansion factor
        # )
        # causal_conv1d要求步幅(x.s arstride(0)和x.s arstride(2))为8的倍数
        # d_model * expand / headdim 是 8 的 倍数
        self.mamba = Mamba2(
            d_model=dim,  # Model dimension d_model
            d_state=128,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
            headdim=64,
        )
        mlp_ratio = 4
        dropout_rate = 0
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=dropout_rate,
            norm_layer=nn.LayerNorm,
        )

    def forward(self, x):
        x_mamba = self.mamba(x) + x
        out = x_mamba
        out = self.mlp(out)
        return out


class FusionBlock(nn.Module):
    def __init__(
        self,
        dim,
    ):
        super(FusionBlock, self).__init__()
        self.fusion = nn.Sequential()

    def forward(self, x):
        out = self.fusion(x)
        return out


class ClsHead(nn.Module):
    def __init__(self, input_dim=256, num_classes=2):
        super(ClsHead, self).__init__()
        # Pooling layer to aggregate sequence information
        self.pooling = nn.AdaptiveAvgPool1d(1)
        dropout_rate = 0
        self.dropout = nn.Dropout(dropout_rate)
        # Fully connected layers
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x: [batch_size, sequence_length, feature_dim] -> [6, 512, 256]
        # Permute and pool to reduce sequence dimension
        x = x.permute(0, 2, 1)  # [batch_size, feature_dim, sequence_length]
        x = self.pooling(x).squeeze(-1)  # [batch_size, feature_dim]
        x = self.dropout(x)
        # Classification layers
        x = self.fc(x)
        return x


class MultipleSequenceHybridMamba(nn.Module):
    def __init__(
        self,
        in_channels=1,
        stem_channels=64,
        num_classes=2,
        spatial_dims=3,
        mamba_encoder=True,
        cross_attn=True,
        mamba_fusion=True,
    ):
        super(MultipleSequenceHybridMamba, self).__init__()
        embed_dim = stem_channels * 2**4
        self.mamba_encoder = mamba_encoder
        self.cross_attn = cross_attn
        self.mamba_fusion = mamba_fusion
        # Define ConvStem for initial feature extraction
        self.conv_stem = ConvStem(
            in_channels=1,
            out_channels=stem_channels,
            spatial_dims=spatial_dims,
        )

        # Define feature extractor
        self.feature_extractor = FeatureExtractor(
            in_channels=stem_channels,
            spatial_dims=spatial_dims,
            mamba_encoder=mamba_encoder,
        )

        if self.cross_attn:
            # Cross-modal attention block (hidden size is adjustable based on encoder channels and model design)
            self.cross_modal_attn_block = CrossModalAttention(
                hidden_size=embed_dim,
                num_heads=4,
                dropout_rate=0,
                qkv_bias=True,
                use_flash_attention=True,
            )

        if mamba_fusion:
            # Mamba Fusion Block
            self.fusion_block = MambaFusionBlock(dim=embed_dim)
        else:
            self.fusion_block = FusionBlock(dim=embed_dim)

        # Classification Head
        self.cls_head = ClsHead(input_dim=embed_dim, num_classes=num_classes)

    def forward(self, x):
        t1_sample = x[:, 0:1]
        t2_sample = x[:, 1:2]
        t1c_sample = x[:, 2:3]
        # Move tensors through ConvStem
        t1_features = self.conv_stem(t1_sample)
        t2_features = self.conv_stem(t2_sample)
        t1c_features = self.conv_stem(t1c_sample)

        # Feature extraction
        t1_features = self.feature_extractor(t1_features)
        t2_features = self.feature_extractor(t2_features)
        t1c_features = self.feature_extractor(t1c_features)

        # Reshape features to (B, N, C) format for cross-attention

        t1_flat = t1_features.view(
            t1_features.size(0), t1_features.size(1), -1
        ).permute(0, 2, 1)
        t2_flat = t2_features.view(
            t2_features.size(0), t2_features.size(1), -1
        ).permute(0, 2, 1)
        t1c_flat = t1c_features.view(
            t1c_features.size(0), t1c_features.size(1), -1
        ).permute(0, 2, 1)

        if self.cross_attn:
            # Cross-modal attention
            t1_t2_attn_output = self.cross_modal_attn_block(t1_flat, context=t2_flat)
            t2_t1c_attn_output = self.cross_modal_attn_block(t2_flat, context=t1c_flat)
            t1c_t1_attn_output = self.cross_modal_attn_block(t1c_flat, context=t1_flat)
        else:
            t1_t2_attn_output = t1_flat
            t2_t1c_attn_output = t2_flat
            t1c_t1_attn_output = t1c_flat

        # Concatenate cross-attention outputs for fusion
        combined_features = torch.cat(
            [t1_t2_attn_output, t2_t1c_attn_output, t1c_t1_attn_output], dim=1
        )

        # Mamba Fusion Block
        fused_features = self.fusion_block(combined_features)

        # Classification Head
        output = self.cls_head(fused_features)

        return output


if __name__ == "__main__":
    input_size = [224, 224]
    batch_size = 1
    in_channels = 3
    num_classes = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spatial_dims = 2
    stem_channels = 64
    x = torch.randn(batch_size, in_channels, input_size[0], input_size[1]).to(device)

    # Instantiate the model
    model = MultipleSequenceHybridMamba(
        in_channels=in_channels,
        num_classes=num_classes,
        stem_channels=stem_channels,
        spatial_dims=spatial_dims,
        mamba_encoder=True,
        cross_attn=True,
        mamba_fusion=True,
    ).to(device)
    img = torch.randn(batch_size, in_channels, input_size[0], input_size[1]).to(device)
    summary(model=model, input_size=img.shape)
    preds = model(img)
    print(preds, preds[0].shape)
    for name, _ in model.named_modules():
        print(name)
    summary(model=model, input_size=(1, 3, 224, 224))
    dummy_input = torch.randn(1, 1, 3, 224, 224).to(device)
    flops, params = get_model_complexity_info(model, tuple([3, 224, 224]), as_strings=False, print_per_layer_stat=False, verbose=False)
    print(
        "the flops is {}G, the params is {}M".format(
            round(flops / (10**9), 1), round(params / (10**6), 2)
        )
    )
