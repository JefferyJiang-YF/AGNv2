import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.nn import LayerNorm, Dropout
from transformers import BertModel, BertConfig

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def sequence_masking(x, mask=None, value='-inf', axis=None):
    """Mask sequence.

    Args:
        x (torch.Tensor): Input tensor.
        mask (torch.Tensor, optional): Mask tensor. Defaults to None.
        value (str or float, optional): Fill value, could be '-inf', 'inf' or any float. Defaults to '-inf'.
        axis (int, optional): The dimension to apply masking. Defaults to None.

    Returns:
        torch.Tensor: Masked tensor.
    """
    if mask is None:
        return x
    if isinstance(value, str):
        if value == '-inf':
            value = float('-inf')
        elif value == 'inf':
            value = float('inf')
        else:
            raise ValueError('If value is a string, it must be `-inf` or `inf`')

    # Ensure mask is a boolean tensor
    if mask.dtype != torch.bool:
        mask = mask.to(torch.bool)

    # Cast the mask to the same data type as x
    if mask.dtype != x.dtype:
        mask = mask.to(x.dtype)

    # Handle the axis parameter
    if axis is None:
        axis = 1
    elif axis < 0:
        axis = x.ndim + axis

    if axis <= 0 or axis >= x.ndim:
        raise ValueError('axis must be between 1 and x.ndim - 1')

    # Ensure the mask is aligned with the dimensions of x
    expand_shape = list(x.shape)
    expand_shape[axis] = 1
    mask = mask.expand(expand_shape)

    # Apply the mask
    masked_x = torch.where(mask, x, torch.tensor(value, dtype=x.dtype, device=x.device))

    return masked_x





class WithVAELoss(nn.Module):
    def forward(self, inputs):
        z_mean, z_log_var, y_true, y_pred = inputs

        # Reconstruction loss
        reconstruction_loss = F.binary_cross_entropy(y_pred, y_true, reduction='mean')

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=1)
        kl_loss = kl_loss.mean()  # Ensure this is a scalar

        # Total loss
        loss = reconstruction_loss + kl_loss

        return loss


class Sampling(nn.Module):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def __init__(self):
        super(Sampling, self).__init__()

    def forward(self, z_mean, z_log_var):
        epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64, hidden_dim=128, epochs=1, batch_size=64):
        super(VariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        # Encoder
        self.encoder_hidden = nn.Linear(input_dim, hidden_dim)
        self.z_mean = nn.Linear(hidden_dim, latent_dim)
        self.z_log_var = nn.Linear(hidden_dim, latent_dim)
        self.sampling = Sampling()

        # Decoder
        self.decoder_output = nn.Linear(latent_dim, input_dim)

        # Loss module
        self.loss_module = WithVAELoss()

    def forward(self, x):
        x = x.float()
        hidden = F.relu(self.encoder_hidden(x))
        z_mean = self.z_mean(hidden)
        z_log_var = self.z_log_var(hidden)
        z = self.sampling(z_mean, z_log_var)
        reconstruction = torch.sigmoid(self.decoder_output(z))
        # Now return all necessary components for loss calculation
        return reconstruction, z_mean, z_log_var, x

    def fit(self, data, learning_rate=1e-3):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=True)
        self.to(('cuda' if torch.cuda.is_available() else 'cpu'))
        self.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_idx, x in enumerate(dataloader):

                x = torch.Tensor(x).to('cuda' if torch.cuda.is_available() else 'cpu')

                optimizer.zero_grad()
                reconstruction, z_mean, z_log_var, x_input = self.forward(x)
                loss = self.loss_module((z_mean, z_log_var, x_input, reconstruction))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f'Epoch {epoch + 1}: Average Loss: {total_loss / len(dataloader)}')

    def encode(self, x):
        """Encodes the input using the VAE encoder and returns the latent space representation."""
        self.eval()
        with torch.no_grad():
            hidden = F.relu(self.encoder_hidden(x))
            z_mean = self.z_mean(hidden)
            z_log_var = self.z_log_var(hidden)
            z = self.sampling(z_mean, z_log_var)
        return z



class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64, hidden_dim=128, epochs=1, batch_size=64):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size

        # Encoder
        self.encoder_hidden = nn.Linear(input_dim, hidden_dim)
        self.encoder_output = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder_output = nn.Linear(latent_dim, input_dim)

        # Loss module (assuming MSE loss for simplicity)
        self.loss_module = nn.MSELoss()

    def forward(self, x):
        x = x.float()
        hidden = F.relu(self.encoder_hidden(x))
        encoded = self.encoder_output(hidden)
        decoded = torch.sigmoid(self.decoder_output(encoded))
        return decoded

    def fit(self, data, learning_rate=1e-3):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=True)

        self.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_idx, x in enumerate(dataloader):
                x = x.float()
                optimizer.zero_grad()
                reconstruction = self.forward(x)
                loss = self.loss_module(reconstruction, x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f'Epoch {epoch + 1}: Average Loss: {total_loss / len(dataloader)}')

    def encode(self, x):
        """Encodes the input using the AE encoder and returns the latent space representation."""
        self.eval()
        with torch.no_grad():
            hidden = F.relu(self.encoder_hidden(x))
            encoded = self.encoder_output(hidden)
        return encoded


class GatedLinearUnit(nn.Module):
    def __init__(self, units, kernel_initializer=None):
        super(GatedLinearUnit, self).__init__()
        self.units = units

        # PyTorch doesn't use the same initializers as Keras, but we can define a similar behavior
        if kernel_initializer == 'glorot_normal':
            initializer = nn.init.xavier_normal_
        else:
            initializer = None  # Default PyTorch initializer

        # Define layers
        self.linear = nn.Linear(units, units)
        self.sigmoid = nn.Linear(units, units)

        # Apply the initializer to weights if specified
        if initializer:
            initializer(self.linear.weight)
            initializer(self.sigmoid.weight)

    def forward(self, x):
        return self.linear(x) * torch.sigmoid(self.sigmoid(x))


class SelfAttention(nn.Module):
    def __init__(self, units=None, return_attention=False, is_residual=True, activation=None, kernel_initializer=None,
                 use_bias=True, dropout_rate=0.0):
        super(SelfAttention, self).__init__()

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
        self.dropout = Dropout(dropout_rate) if dropout_rate > 0 else None

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        units = feature_dim if self.units is None else self.units

        self.q_dense = nn.Linear(feature_dim, units, bias=self.use_bias)
        self.k_dense = nn.Linear(feature_dim, units, bias=self.use_bias)
        self.v_dense = nn.Linear(feature_dim, units, bias=self.use_bias)
        self.o_dense = nn.Linear(units, feature_dim, bias=self.use_bias)

        if self.is_residual:
            self.glu = GatedLinearUnit(feature_dim)  # GatedLinearUnit assumed to be defined earlier
            self.layernorm = LayerNorm(feature_dim)

    def forward(self, x, mask=None):
        if self.q_dense is None:
            self.build(x.shape)

        q = self.q_dense(x)
        k = self.k_dense(x)
        v = self.v_dense(x)

        qk = torch.bmm(q, k.transpose(1, 2))
        scale = q.shape[-1] ** 0.5
        qk /= scale

        if mask is not None:
            qk = qk.masked_fill(mask == 0, float('-inf'))

        a = F.softmax(qk, dim=-1)

        if self.dropout:
            a = self.dropout(a)

        out = torch.bmm(a, v)
        out = self.o_dense(out)

        if self.is_residual:
            out += self.glu(q)  # Assuming GatedLinearUnit modifies q appropriately
            out = self.layernorm(out)

        if self.return_attention:
            return out, a

        return out



class AGN(nn.Module):
    def __init__(self, feature_size, activation='swish', dropout_rate=0.1, valve_rate=0.3, dynamic_valve=False):
        super(AGN, self).__init__()
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.valve_rate = valve_rate
        self.dynamic_valve = dynamic_valve
        self.valve_transform = nn.Linear(feature_size, feature_size, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)
        self.attn = SelfAttention(feature_size, return_attention=True, activation=self.activation, dropout_rate=dropout_rate)

    def forward(self, X, gi):
        valve = self.sigmoid(self.valve_transform(X))
        if self.dynamic_valve:
            valve = nn.Dropout(1.0 - self.valve_rate)(valve)
        else:
            valve_mask = (valve > 0.5 - self.valve_rate) & (valve < 0.5 + self.valve_rate)
            valve = valve * valve_mask.float()

        enhanced = X + valve * gi
        enhanced = self.dropout(enhanced)
        out, attn_weights = self.attn(enhanced)
        return out, attn_weights


class AGNModel(nn.Module):
    def __init__(self, config, task='clf'):
        super(AGNModel, self).__init__()
        self.config = config
        self.task = task
        self.bert_model = BertModel.from_pretrained(config['pretrained_model_dir'])

        self.feature_size = self.bert_model.config.hidden_size
        self.agn = AGN(self.feature_size, activation='swish', dropout_rate=config.get('dropout_rate', 0.1),
                       valve_rate=config.get('valve_rate', 0.3), dynamic_valve=config.get('use_dynamic_valve', False))
        self.gi_dense = nn.Linear(config["ae_latent_dim"], self.feature_size)
        self.gi_dropout = nn.Dropout(config.get('dropout_rate', 0.1))
        self.register_buffer("gi", torch.zeros(1, 1, self.feature_size))

        if self.task == 'clf':
            self.output_layer = nn.Sequential(
                nn.Linear(self.feature_size, config.get('hidden_size', 256)),
                nn.ReLU(),
                nn.Dropout(config.get('dropout_rate', 0.1)),
                nn.Linear(config.get('hidden_size', 256), config['output_size']),
                nn.Softmax(dim=-1)
            )
        elif self.task == 'ner':
            self.output_layer = nn.Linear(self.feature_size, config['output_size'])
        elif self.task == 'sts':
            pass

    def forward(self, inputs):
        input_ids, token_type_ids, _ = inputs
        attention_mask = (input_ids != 0).long()

        outputs = self.bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state

        self.update_gi()
        agn_output, attn_weights = self.agn(sequence_output, self.gi)

        if self.task == 'clf':
            # Apply global max pooling for classification tasks
            pooled_output = torch.max(agn_output, dim=1)[0]
            output = self.output_layer(pooled_output)
        elif self.task in ['ner', 'sts']:
            # Directly use the sequence output for NER or STS tasks
            output = self.output_layer(agn_output)

        return output, attn_weights


    def update_gi(self, new_gi=None):
        # Update the `gi` buffer based on new_gi or some internal logic
        if new_gi is not None:
            processed_gi = self.gi_dense(new_gi)
            processed_gi = self.gi_dropout(processed_gi)
            self.gi = processed_gi.unsqueeze(1)  # Ensure gi is correctly shaped

    def configure_optimizers(self):
        if self.config['optimizer'] == 'adamw':
            optimizer = optim.AdamW(self.parameters(), lr=self.config['learning_rate'])
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.config['learning_rate'])
        return optimizer
