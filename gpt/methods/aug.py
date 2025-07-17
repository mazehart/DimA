import torch
from transformers.activations import ACT2FN


class AugAtt(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.aug_dim = config.aug_dim
        self.n_heads = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.att = torch.nn.Linear(config.n_embd, self.aug_dim * 2)

    def forward(self, x, query_layer, key_layer):
        bsz, seq, dim = x.size()
        x = self.att(x)
        y, z = x.split(self.aug_dim, dim=-1)
        query_layer = torch.cat([query_layer, y.view(bsz, seq, -1, self.n_heads).permute(0, 3, 1, 2)], dim=-1)
        key_layer = torch.cat([key_layer, z.view(bsz, seq, -1, self.n_heads).permute(0, 3, 1, 2)], dim=-1)
        return query_layer, key_layer


class AugMlp(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.aug_dim = config.aug_dim
        self.line_dim = self.aug_dim if not config.line_m else config.line_m * self.aug_dim
        self.line = torch.nn.Sequential(torch.nn.Linear(config.n_embd, config.aug_dim),
                                        ACT2FN[config.activation_function],
                                        torch.nn.Linear(config.aug_dim, config.n_embd, bias=config.use_bias))

    def forward(self, x):
        return self.line(x)


class AugV(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.aug_dim = config.aug_dim
        self.n_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.v = torch.nn.Linear(config.hidden_size, self.aug_dim)

    def forward(self, x, value_layer):
        bsz, seq, dim = x.size()
        y = self.v(x)
        value_layer = torch.cat([value_layer, y.view(bsz, seq, -1, self.n_heads).permute(0, 3, 1, 2)], dim=-1)
        return value_layer

    def transform(self, context, new_context_layer_shape):
        x = context[..., :self.head_dim]
        y = context[..., self.head_dim:]
        context = torch.cat((x.contiguous().view(new_context_layer_shape), y.contiguous().view(new_context_layer_shape)), dim=-1)
        return context


class AugO(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.aug_dim = config.aug_dim
        self.o = torch.nn.Linear(self.aug_dim, config.hidden_size, bias=False)

    def forward(self, x):
        y = self.o(x[..., -self.aug_dim:])
        return y
