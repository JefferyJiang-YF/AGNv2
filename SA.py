import torch
from torch import nn
class SelfAttention(nn.Module):
    def __init__(self, units=None, return_attention=False, is_residual=True, activation=None,
                 use_bias=True, dropout_rate=0.0):
        super(SelfAttention, self).__init__()
        # Automatically determine the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.units = units
        self.return_attention = return_attention
        self.is_residual = is_residual
        self.activation = activation if activation is not None else torch.relu
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate

        # Initialize dense layers later in build method
        self.q_dense = None
        self.k_dense = None
        self.v_dense = None
        self.o_dense = None
        self.glu = None
        self.layernorm = None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        units = feature_dim if self.units is None else self.units

        # Initialize layers and move them to the appropriate device
        self.q_dense = nn.Linear(feature_dim, units, bias=self.use_bias).to(self.device)
        self.k_dense = nn.Linear(feature_dim, units, bias=self.use_bias).to(self.device)
        self.v_dense = nn.Linear(feature_dim, units, bias=self.use_bias).to(self.device)
        self.o_dense = nn.Linear(units, feature_dim, bias=self.use_bias).to(self.device)

        if self.is_residual:
            self.glu = GatedLinearUnit(feature_dim).to(self.device)  # Assuming GatedLinearUnit is defined
            self.layernorm = nn.LayerNorm(feature_dim).to(self.device)

    def forward(self, x, mask=None):
        x = x.to(self.device)  # Ensure input tensor is on the right device

        if self.q_dense is None:
            self.build(x.shape)

        q = self.q_dense(x)
        k = self.k_dense(x)
        v = self.v_dense(x)

        qk = torch.bmm(q, k.transpose(1, 2))
        scale = q.shape[-1] ** 0.5
        qk /= scale

        if mask is not None:
            mask = mask.to(self.device)  # Ensure mask is on the right device
            qk = qk.masked_fill(mask == 0, float('-inf'))

        a = F.softmax(qk, dim=-1)

        if self.dropout:
            a = self.dropout(a)

        out = torch.bmm(a, v)
        out = self.o_dense(out)

        if self.is_residual:
            out += self.glu(q)
            out = self.layernorm(out)

        if self.return_attention:
            return out, a

        return out
