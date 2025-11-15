from torch import nn
from torch.nn import Sequential


class BaselineModel(nn.Module):
    """
    Simple MLP
    """

    def __init__(self, length, fc_hidden=512):
        """
        Args:
            length (int): input's length.
            fc_hidden (int): number of hidden features.
        """
        super().__init__()

        self.net = Sequential(
            # people say it can approximate any function...
            nn.Linear(in_features=length, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=2 * length),
        )

    def forward(self, mix, **batch):
        """
        Model forward method.

        Args:
            mix (Tensor): mixed input.
        Returns:
            output (dict): output dict containing predictions.
        """
        output = self.net(mix)
        output = output.view((mix.shape[0], 2, mix.shape[1]))
        return {"preds": output}

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
