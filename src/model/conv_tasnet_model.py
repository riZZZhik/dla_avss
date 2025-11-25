import torch
from torch import nn
import torch.nn.functional as F

from src.model.dprnn_model import TasNetEncoder, TasNetDecoder


def _model_str(self):
    all_parameters = sum(p.numel() for p in self.parameters())
    trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

    result_info = super(self.__class__, self).__str__()
    result_info += f"\nAll parameters: {all_parameters}"
    result_info += f"\nTrainable parameters: {trainable_parameters}"
    return result_info


class ConvTasNetBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.conv_in = nn.Conv1d(
            in_channels,
            hidden_channels,
            kernel_size=1,
            bias=False,
        )

        self.norm1 = nn.GroupNorm(1, hidden_channels)
        self.prelu1 = nn.PReLU()

        self.dw_conv = nn.Conv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            groups=hidden_channels,
            dilation=dilation,
            padding="same",
            bias=False,
        )

        self.norm2 = nn.GroupNorm(1, hidden_channels)
        self.prelu2 = nn.PReLU()

        self.res_conv = nn.Conv1d(
            hidden_channels,
            in_channels,
            kernel_size=1,
            bias=False,
        )
        self.skip_conv = nn.Conv1d(
            hidden_channels,
            in_channels,
            kernel_size=1,
            bias=False,
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x, **batch):
        residual = x  

        y = self.conv_in(x)     
        y = self.norm1(y)
        y = self.prelu1(y)

        y = self.dw_conv(y)   
        y = self.norm2(y)
        y = self.prelu2(y)

        if self.dropout is not None:
            y = self.dropout(y)

        res = self.res_conv(y)   
        skip = self.skip_conv(y) 

        out = residual + res
        return out, skip

    def __str__(self):
        all_parameters = sum(p.numel() for p in self.parameters())
        trainable_parameters = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        result_info = super().__str__()
        result_info += f"\nAll parameters: {all_parameters}"
        result_info += f"\nTrainable parameters: {trainable_parameters}"
        return result_info


class ConvTasNetSeparator(nn.Module):

    def __init__(
        self,
        encoder_channels: int,
        bottleneck_channels: int,
        n_blocks: int,
        n_repeats: int,
        kernel_size: int,
        n_src: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_src = n_src
        self.encoder_channels = encoder_channels

        blocks = []
        for _ in range(n_repeats):
            for b in range(n_blocks):
                dilation = 2 ** b
                blocks.append(
                    ConvTasNetBlock(
                        in_channels=encoder_channels,
                        hidden_channels=bottleneck_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        dropout=dropout,
                    )
                )
        self.tcn_blocks = nn.ModuleList(blocks)

        self.pre_norm = nn.GroupNorm(1, encoder_channels)
        self.pre_act = nn.PReLU()
        self.pre_conv = nn.Conv1d(
            encoder_channels,
            encoder_channels,
            kernel_size=1,
            bias=False,
        )

        self.mask_conv = nn.Conv1d(
            encoder_channels,
            n_src * encoder_channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x, **batch):

        B, N, L = x.shape

        h = self.pre_conv(self.pre_act(self.pre_norm(x)))  

        skip_sum = 0
        first = True

        for block in self.tcn_blocks:
            h, skip = block(h) 
            if first:
                skip_sum = skip
                first = False
            else:
                skip_sum = skip_sum + skip

        if first:
            skip_sum = h

        skip_sum = F.relu(skip_sum)            
        mask_logits = self.mask_conv(skip_sum) 

        mask_logits = mask_logits.view(B, self.n_src, self.encoder_channels, L)
        masks = torch.sigmoid(mask_logits)
        return masks


class ConvTasNetModel(nn.Module):

    def __init__(
        self,
        encoder_out_features: int = 512,   
        kernel: int = 16,
        bottleneck_size: int = 128,      
        n_blocks: int = 8,
        n_repeats: int = 3,
        n_src: int = 2,
        conv_dropout: float = 0.0,
    ):
        super().__init__()

        self.n_src = n_src
        self.encoder_out_features = encoder_out_features

        self.encoder = TasNetEncoder(
            in_features=1,
            out_features=encoder_out_features,
            kernel=kernel,
        )

        self.separator = ConvTasNetSeparator(
            encoder_channels=encoder_out_features,
            bottleneck_channels=bottleneck_size,
            n_blocks=n_blocks,
            n_repeats=n_repeats,
            kernel_size=kernel,
            n_src=n_src,
            dropout=conv_dropout,
        )

        self.decoder = TasNetDecoder(
            in_features=encoder_out_features,
            kernel=kernel,
        )

        self.relu = nn.ReLU()

    def forward(self, mix, **batch):

        if mix.dim() == 2:
            mix_in = mix.unsqueeze(1)
        else:
            mix_in = mix

        encoded = self.encoder(mix_in)   
        L_enc = encoded.shape[-1]

        masks = self.separator(encoded)   
        L_mask = masks.shape[-1]
        L_min = min(L_enc, L_mask)

        encoded = encoded[..., :L_min]        
        masks = masks[..., :L_min]         

        encoded_expanded = encoded.unsqueeze(1) 
        masked = masks * encoded_expanded   

        B, n_src, N, Lm = masked.shape
        masked_flat = masked.view(B * n_src, N, Lm) 

        decoded = self.decoder(masked_flat)        
        if decoded.dim() == 1:
            decoded = decoded.unsqueeze(0)

        decoded = decoded.view(B, n_src, -1)         
        decoded = self.relu(decoded)

        T_mix = mix.shape[-1]
        T_dec = decoded.shape[-1]
        T_out = min(T_mix, T_dec)
        preds = decoded[..., :T_out]             

        return {"preds": preds}

    __str__ = _model_str
