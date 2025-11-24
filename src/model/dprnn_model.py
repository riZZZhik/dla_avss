from unicodedata import bidirectional

import torch
from torch import nn
from torch.nn import Sequential


class TasNetEncoder(nn.Module):
    """
    TasNetEncoder, modified (https://arxiv.org/pdf/2002.08688)
    """

    def __init__(self, in_features, out_features, kernel=2):
        """
        Args:
            in_features (int): input's in_features.
            out_features (int): outpput's features
            kernel (int): size of the first kernel.
            stride (int): first stride
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(
                in_features, out_features, kernel_size=kernel, stride=kernel // 2
            ),
            nn.Conv1d(
                out_features,
                out_features,
                kernel_size=3,
                stride=1,
                dilation=1,
                padding=1,
            ),
            nn.PReLU(),
            nn.Conv1d(
                out_features,
                out_features,
                kernel_size=3,
                stride=1,
                dilation=2,
                padding=2,
            ),
            nn.PReLU(),
            nn.Conv1d(
                out_features,
                out_features,
                kernel_size=3,
                stride=1,
                dilation=4,
                padding=4,
            ),
            nn.PReLU(),
            nn.Conv1d(
                out_features,
                out_features,
                kernel_size=3,
                stride=1,
                dilation=8,
                padding=8,
            ),
            nn.PReLU(),
        )

    def forward(self, mix, **batch):
        """
        Model forward method.

        Args:
            mix (Tensor): mixed input.
        Returns:
            output (dict): encoded mix.
        """
        return self.net(mix)

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info


class TasNetDecoder(nn.Module):
    """
    TasNetDecoder, modified (https://arxiv.org/pdf/2002.08688)
    """

    def __init__(self, in_features, kernel=2):
        """
        Args:
            in_features (int): input's in_features.
            kernel (int): size of the last kernel.
            stride (int): last stride
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(
                in_features, in_features, kernel_size=3, stride=1, dilation=8, padding=8
            ),
            nn.PReLU(),
            nn.Conv1d(
                in_features, in_features, kernel_size=3, stride=1, dilation=4, padding=4
            ),
            nn.PReLU(),
            nn.Conv1d(
                in_features, in_features, kernel_size=3, stride=1, dilation=2, padding=2
            ),
            nn.PReLU(),
            nn.Conv1d(
                in_features, in_features, kernel_size=3, stride=1, dilation=1, padding=1
            ),
            nn.PReLU(),
            nn.ConvTranspose1d(in_features, 1, kernel_size=kernel, stride=kernel // 2),
        )

    def forward(self, separated_encoded, **batch):
        """
        Model forward method.

        Args:
            separated_encoded (Tensor): separated input to be decoded.
        Returns:
            output (dict): decoded input.
        """
        return self.net(separated_encoded).squeeze()

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info


class DPRNN_Block(nn.Module):
    """
    DPRNN block from https://arxiv.org/pdf/1910.06379
    """

    def __init__(self, in_features, hidden_features, dropout):
        """
        Args:
            in_features (int): input's features (and output size (as there are sequential dprnn blocks)).
            hidden_features (int): size of the hidden layers.
        """
        super().__init__()

        self.intra_rnn = nn.LSTM(
            in_features,
            hidden_features,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.inter_rnn = nn.LSTM(
            in_features,
            hidden_features,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        self.intra_norm = nn.GroupNorm(1, in_features)
        self.inter_norm = nn.GroupNorm(1, in_features)

        self.intra_linear = nn.Linear(2 * hidden_features, in_features)
        self.inter_linear = nn.Linear(2 * hidden_features, in_features)

    def forward(self, x, **batch):
        """
        Model forward method.

        Args:
            x (Tensor): input.
        Returns:
            output (dict): output.
        """
        # INTRA

        # BxBottleneck_SizexKxS
        B, Bottle, K, S = x.shape
        y = x.permute(0, 3, 2, 1).reshape(
            B * S, K, Bottle
        )  # BxBottleneck_SizexKxS -> (B * S)xKxBottleneck_Size

        y, _ = self.intra_rnn(y)

        y = y.reshape(
            B * S * K, 2 * Bottle
        )  # (B * S)xKxBottleneck_Size -> (B * S * K)xBottleneck_Size
        y = self.intra_linear(y)
        y = y.reshape(
            B, S, K, Bottle
        )  # (B * S * K)xBottleneck_Size -> BxSxKxBottleneck_Size

        y = y.permute(0, 3, 2, 1)  # BxSxKxBottleneck_Size -> BxBottleneck_SizexKxS

        y = self.intra_norm(y)

        y = y + x

        # INTER

        # BxBottleneck_SizexKxS
        z = y.permute(0, 2, 3, 1).reshape(
            B * K, S, Bottle
        )  # BxBottleneck_SizexKxS -> (B * K)xSxBottleneck_Size

        z, _ = self.inter_rnn(z)

        z = z.reshape(
            B * S * K, 2 * Bottle
        )  # (B * K)xSxBottleneck_Size -> (B * K * S)xBottleneck_Size
        z = self.inter_linear(z)
        z = z.reshape(
            B, K, S, Bottle
        )  # (B * K * S)xBottleneck_Size -> BxKxSxBottleneck_Size

        z = z.permute(0, 3, 1, 2)  # BxKxSxBottleneck_Size -> BxBottleneck_SizexKxS
        z = self.inter_norm(z)

        output = z + y

        return output

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info


class DPRNNModel(nn.Module):
    """
    DPRNN model from https://arxiv.org/pdf/1910.06379, modifed
    """

    def __init__(
        self,
        encoder_out_features=512,
        kernel=2,
        bottleneck_size=128,
        dprnn_hidden_features=128,
        dprnn_dropout=0,
        num_dprnn_blocks=4,
        chunk_len=200,  # 32
        sample_rate=16000,
        rec_length=2,
    ):
        """
        Args:
            length (int): input's length.
            fc_hidden (int): number of hidden features.
        """
        super().__init__()
        self.chunk_len = chunk_len
        self.encoder_out_features = encoder_out_features

        # BxL -> BxEncoder_Out_FeaturesxL
        self.encoder = TasNetEncoder(1, encoder_out_features, kernel)

        # BxEncoder_Out_FeaturesxL
        self.norm = nn.GroupNorm(1, encoder_out_features)

        # BxBottleneck_SizexL
        self.conv1d = nn.Conv1d(
            in_channels=encoder_out_features,
            out_channels=bottleneck_size,
            kernel_size=1,
        )

        # Segmentation
        # # BxBottleneck_SizexL -> BxBottleneck_SizexKxS

        # Separation block
        # BxBottleneck_SizexKxS
        self.dprnn = nn.ModuleList(
            [
                DPRNN_Block(
                    bottleneck_size,
                    dprnn_hidden_features,
                    dropout=dprnn_dropout,
                )
                for _ in range(num_dprnn_blocks)
            ]
        )

        # Prelu, Conv2D
        self.prelu = nn.PReLU()
        # BxBottleneck_SizexKxS -> Bx(Bottleneck_Size * Speakers)xKxS
        self.conv2d = nn.Conv2d(bottleneck_size, 2 * bottleneck_size, kernel_size=1)

        # OverAdd, output, output_gate
        self.output = nn.Sequential(
            nn.Conv1d(bottleneck_size, bottleneck_size, 1), nn.Tanh()
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(bottleneck_size, bottleneck_size, 1), nn.Sigmoid()
        )

        # end_conv1x1
        self.end_conv1x1 = nn.Conv1d(
            bottleneck_size, encoder_out_features, 1, bias=False
        )

        # relu
        self.relu = nn.ReLU()

        # BxEncoder_Out_FeaturesxL -> Bx1xL
        self.decoder = TasNetDecoder(encoder_out_features, kernel)

    def forward(self, mix, **batch):
        """
        Model forward method.

        Args:
            mix (Tensor): mixed input.
        Returns:
            output (dict): output dict containing predictions.
        """
        x_encoded = self.encoder(mix.unsqueeze(1))
        x_encoded_len = x_encoded.shape[-1]

        x = self.norm(x_encoded)

        x = self.conv1d(x)

        x = self.pad(x, self.chunk_len)
        x = self.segment(x, self.chunk_len)
        B, Bottle, K, S = x.shape

        for dprnn_block in self.dprnn:
            x = dprnn_block(x)

        x = self.prelu(x)
        x = self.conv2d(x)

        x = x.reshape(2 * B, Bottle, K, S)

        x = self.overlap_add(x, x_encoded_len, self.chunk_len)
        x = self.output(x) * self.output_gate(x)

        x = self.end_conv1x1(x)

        x = x.reshape(B, 2, self.encoder_out_features, x_encoded_len)
        x = self.relu(x)

        x = [x[:, speaker] * x_encoded for speaker in range(2)]
        preds = torch.stack([self.decoder(x[speaker]) for speaker in range(2)], dim=0)
        preds = preds.transpose(0, 1)

        return {"preds": preds}

    def pad(self, x, chunk_len):
        """
        Model padding method.

        Args:
            x (Tensor): input to pad.
            chunk_len (int): length of future chunks
        Returns:
            output (Tensor): padded input.
        """
        # BxEncoder_Out_FeaturesxL
        B, EOutF, L = x.shape
        K = chunk_len
        P = K // 2

        if K - (P + L % K) % K > 0:
            x = torch.cat(
                [
                    x,
                    torch.zeros(
                        (B, EOutF, K - (P + L % K) % K), dtype=x.dtype, device=x.device
                    ),
                ],
                dim=2,
            )

        P_pad = torch.zeros((B, EOutF, P), dtype=x.dtype, device=x.device)
        x = torch.cat([P_pad, x, P_pad], dim=2)

        return x

    def segment(self, x, chunk_len):
        """
        Model segmentation method.

        Args:
            x (Tensor): input to be segmented.
            chunk_len (int): length of chunks
        Returns:
            output (Tensor): segmented input input.
        """
        B, EOutF, L_new = x.shape
        K = chunk_len
        P = K // 2
        x_pre = x[..., :-P].reshape(B, EOutF, -1, K)
        x_post = x[:, :, P:].reshape(B, EOutF, -1, K)

        x = torch.cat([x_pre, x_post], dim=3).reshape(B, EOutF, -1, K).transpose(2, 3)

        return x

    def overlap_add(self, x, x_orig_len, chunk_len):
        """
        Model overlap-addind method.

        Args:
            x (Tensor): input to be segmented.
            x_orig_len (int): original input length
            chunk_len (int): length of chunks
        Returns:
            output (Tensor): united input (B * Speakers)xBottlexL.
        """
        # (B * Speakers)xBottlexKxS
        B_Sps, Bottle, K, S = x.shape
        P = chunk_len // 2
        x = x.transpose(2, 3).reshape(B_Sps, Bottle, -1, K * 2)

        x_pre = x[..., :K].reshape(B_Sps, Bottle, -1)
        x_pre = x_pre[:, :, P:]
        x_post = x[..., K:].reshape(B_Sps, Bottle, -1)
        x_post = x_post[:, :, :-P]
        x = x_pre + x_post

        return x[..., :x_orig_len] if x.shape[-1] > x_orig_len else x

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
