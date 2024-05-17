from typing import List
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision
from s2igan.utils import set_non_grad
from torchvision import models 

class SpeechEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 40,
        cnn_dim: List[int] = [64, 128],
        kernel_size: int = 6,
        stride: int = 2,
        rnn_dim: int = 512,
        rnn_num_layers: int = 2,
        rnn_type: str = "gru",
        rnn_dropout: float = 0.1,
        rnn_bidirectional: bool = True,
        attn_heads: int = 1,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        assert rnn_type in ["lstm", "gru"]
        
        self.vgg16 = models.vgg16(pretrained=False).features

        self.kernel_size = kernel_size
        self.stride = stride
        
        rnn_kwargs = dict(
            input_size=cnn_dim[1], #512
            hidden_size=rnn_dim,
            num_layers=rnn_num_layers,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=rnn_bidirectional,
        )
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(**rnn_kwargs)
        else:
            self.rnn = nn.GRU(**rnn_kwargs)
        self.output_dim = rnn_dim * (int(rnn_bidirectional) + 1)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.output_dim,
            num_heads=attn_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        
    def get_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, mel_spec,mel_spec_len):
        """
        mel_spec (-1, 40, len)
        output (-1, len, output_dim)
        """
        mel_spec = mel_spec.unsqueeze(1).repeat(1, 3, 1, 1)
        cnn_out = self.vgg16(mel_spec)
        bs,dim,h,w = cnn_out.size()
        cnn_out = cnn_out.view(bs,dim,h*w)
        cnn_out = cnn_out.permute(0, 2, 1)
        out, hidden_state = self.rnn(cnn_out)

        out, weights = self.self_attention(out, out, out)
        out = out.mean(dim=1)  
        out = torch.nn.functional.normalize(out)
        return out