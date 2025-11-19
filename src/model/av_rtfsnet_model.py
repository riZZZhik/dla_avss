import torch
import torch.nn as nn


def _model_str(self):
    """
    Writes number of params
    """
    all_parameters = sum(p.numel() for p in self.parameters())
    trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

    result_info = super(self.__class__, self).__str__()
    result_info += f"\nAll parameters: {all_parameters}"
    result_info += f"\nTrainable parameters: {trainable_parameters}"
    return result_info


class SequenceCore(nn.Module):
    """
    Selectable core for RNN (LSTM or MHS)
    """
    def __init__(
        self,
        dim: int,
        num_layers: int = 2,
        mode: str = "lstm",
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mode = mode

        if mode == "lstm":
            self.rnn = nn.LSTM(
                input_size=dim,
                hidden_size=dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.proj = nn.Linear(2 * dim, dim)

        elif mode == "mhs":
            self.layers = nn.ModuleList()
            self.norms = nn.ModuleList()
            for _ in range(num_layers):
                self.layers.append(
                    nn.MultiheadAttention(
                        embed_dim=dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        batch_first=True,
                    )
                )
                self.norms.append(nn.LayerNorm(dim))
            self.ff = nn.Sequential(
                nn.Linear(dim, 4 * dim),
                nn.ReLU(),
                nn.Linear(4 * dim, dim),
            )
        else:
            raise ValueError(f"Unknown core mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
 
        if self.mode == "lstm":
            y, _ = self.rnn(x)
            y = self.proj(y)
            return y
        elif mode == "mhs":
            out = x
            for attn, norm in zip(self.layers, self.norms):
                attn_out, _ = attn(out, out, out)
                out = norm(out + attn_out)
                ff = self.ff(out)
                out = norm(out + ff)
            return out
        else:
            raise ValueError(f"Unknown core mode: {mode}")


class RTFSBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        core_mode: str = "lstm",
        num_layers_freq: int = 1,
        num_layers_time: int = 1,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)

        self.freq_core = SequenceCore(
            dim=hidden_channels,
            num_layers=num_layers_freq,
            mode=core_mode,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.time_core = SequenceCore(
            dim=hidden_channels,
            num_layers=num_layers_time,
            mode=core_mode,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.out_proj = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.in_proj(x)  

        B, C, T, F = x.shape

        xf = x.permute(0, 2, 3, 1).reshape(B * T, F, C)
        xf = self.freq_core(xf)
        xf = xf.reshape(B, T, F, C).permute(0, 3, 1, 2) 

        # 2) по времени: длина = T, батч = B*F
        xt = xf.permute(0, 3, 2, 1).reshape(B * F, T, C)
        xt = self.time_core(xt)
        xt = xt.reshape(B, F, T, C).permute(0, 3, 2, 1)

        out = self.out_proj(xt)
        out = self.norm(out)
        out = self.act(out + residual)
        return out


class AVFusionFiLM(nn.Module):

    def __init__(self, audio_channels: int, visual_dim: int):
        super().__init__()
        self.gamma = nn.Linear(visual_dim, audio_channels)
        self.beta = nn.Linear(visual_dim, audio_channels)

    def forward(self, audio: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma(visual).unsqueeze(-1).unsqueeze(-1)  
        beta = self.beta(visual).unsqueeze(-1).unsqueeze(-1)   
        return gamma * audio + beta


class STFTEncoder(nn.Module):
    """
    Basic encoder based on STFT 
    """
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        base_channels: int = 64,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

        self.conv = nn.Conv2d(1, base_channels, kernel_size=1)

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )

    def forward(self, mix: torch.Tensor):
        mix_stft = self.stft(mix)   
        mag = mix_stft.abs()    

        feat = mag.permute(0, 2, 1).unsqueeze(1) 
        feat = self.conv(feat)                   
        return feat, mix_stft

    __str__ = _model_str


class ISTFTDecoder(nn.Module):
    """
    Basic dencoder based on STFT
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def istft(self, X: torch.Tensor, length: int) -> torch.Tensor:
        return torch.istft(
            X,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            length=length,
        )

    def forward(self, est_stfts, length: int):
        wavs = [self.istft(X, length=length) for X in est_stfts] 
        return torch.stack(wavs, dim=1) 

    __str__ = _model_str


class AVRTFSNetModel(nn.Module):
    """
    RTFS-NET model based on https://arxiv.org/pdf/2309.17189, modified 
    """
    def __init__(
        self,
        n_src: int = 2,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        base_channels: int = 64,
        num_blocks: int = 4,
        core_mode: str = "lstm",    # lstm/mhs
        num_layers_freq: int = 1,
        num_layers_time: int = 1,
        num_heads: int = 4,
        visual_dim: int = 96,     
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_src = n_src

        self.encoder = STFTEncoder(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            base_channels=base_channels,
        )
        self.decoder = ISTFTDecoder(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )

        self.blocks = nn.ModuleList([
            RTFSBlock(
                in_channels=base_channels,
                hidden_channels=base_channels,
                core_mode=core_mode,
                num_layers_freq=num_layers_freq,
                num_layers_time=num_layers_time,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])

        self.av_fusion = AVFusionFiLM(
            audio_channels=base_channels,
            visual_dim=2 * visual_dim,  
        )

        self.mask_conv = nn.Conv2d(base_channels, n_src, kernel_size=1)

    def forward(self, mix: torch.Tensor, **batch):

        B, T = mix.shape

        vis_s1 = batch["vis_s1"]  
        vis_s2 = batch["vis_s2"]  

        x, mix_stft = self.encoder(mix)

        for block in self.blocks:
            x = block(x)

        vis = torch.cat([vis_s1, vis_s2], dim=-1) 
        x = self.av_fusion(x, vis)     

        masks = torch.sigmoid(self.mask_conv(x))  
        masks = masks.permute(0, 1, 3, 2)  

        est_stfts = []
        for s in range(self.n_src):
            mask_s = masks[:, s]    
            est_stfts.append(mix_stft * mask_s)  

        preds = self.decoder(est_stfts, length=T)  

        return {"preds": preds}

    __str__ = _model_str
