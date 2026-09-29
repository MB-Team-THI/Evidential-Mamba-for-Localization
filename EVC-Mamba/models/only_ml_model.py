#Issues in model is normalization for dec_query input
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba
import math



class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)#Kind of keeping them as buffer parameters. It wont be part of model.parameters() and optimizer wont update it

    def forward(self, x):
        return self.pe[:, :x.size(1)]

####################################### Mamba architecture ###########################################
class only_ml_block(nn.Module):
    def __init__(self,
                 enc_in,
                 seq_len,
                 enc_out,
                 d_model,
                 enc_layers,
                 evidential,
                 dropout=0.05,
                 activation='gelu',
                 device=torch.device('cuda:0')):
        super().__init__()

        self.device = device
        self.evidential = evidential
        # Input projection: project input features -> model dimension
        self.input_projection = nn.Sequential(
            nn.Linear(enc_in, d_model),
            nn.GELU())

        self.input_norm = nn.LayerNorm(d_model)
        # Mamba model
        self.mamba_layers = nn.ModuleList([Mamba(d_model=d_model) for _ in range(enc_layers)])
        # Normalization after Mamba
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(enc_layers)
        ])
        # Optional dropout
        self.dropout = nn.Dropout(dropout)
        # Final projection
        if self.evidential==True:
            self.output_proj = nn.Linear(d_model, enc_out*4) #Each NIG has 4 parameters
        else:
            self.output_proj = nn.Linear(d_model, enc_out)
        # Move entire model to device
        self.to(self.device)


    def forward(self, enc_in):

        x = self.input_projection(enc_in)
        x = self.input_norm(x)

        # Mamba forward
        for mamba, norm in zip(self.mamba_layers, self.layer_norms):
            residual = x
            x = mamba(x)
            x = self.dropout(x)
            x = norm(x + residual)

        # Last timestep
        x = x[:, -100:, :]

        out = self.output_proj(x)

        if self.evidential:
            # reshape → (batch, time, variables, 4)
            out = out.view(*out.shape[:-1], -1, 4)

            mu = out[..., 0]
            lambda_ = F.softplus(out[..., 1]) + 1e-6
            alpha = F.softplus(out[..., 2]) + 1.0 + 1e-6
            beta = F.softplus(out[..., 3]) + 1e-6

            return mu, lambda_, alpha, beta
        return out




###############  Transformer based architecture ##############################
# class only_ml_block(nn.Module):
#     def __init__(self,
#                  enc_in,
#                  seq_len,
#                  enc_out,
#                  d_model,
#                  enc_layers,
#                  evidential,
#                  dropout=0.05,
#                  activation='gelu',
#                  device=torch.device('cuda:0')):
#         super().__init__()
#
#         self.device = device
#         self.evidential = evidential
#         self.seq_len = seq_len
#
#         # Input projection: enc_in -> d_model
#         self.input_projection = nn.Sequential(
#             nn.Linear(enc_in, d_model),
#             nn.GELU()
#         )
#
#         self.input_norm = nn.LayerNorm(d_model)
#
#
#         self.pos_encoder = PositionalEmbedding(d_model, max_len=seq_len)
#
#         # Transformer encoder stack (replaces Mamba)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=8,
#             dim_feedforward=4 * d_model,
#             dropout=dropout,
#             activation=activation,
#             batch_first=True,
#             norm_first=True
#         )
#
#         self.encoder_layers = nn.TransformerEncoder(
#             encoder_layer,
#             num_layers=enc_layers
#         )
#         self.dropout = nn.Dropout(dropout)
#         # Output projection
#         if self.evidential:
#             self.output_proj = nn.Linear(d_model, enc_out * 4)
#         else:
#             self.output_proj = nn.Linear(d_model, enc_out)
#
#         self.to(self.device)
#
#     def forward(self, enc_in):
#
#         x = self.input_projection(enc_in)
#         x = self.input_norm(x)
#
#         # Add positional encoding
#         x = x + self.pos_encoder(x)
#
#         # Transformer encoding
#
#         x = self.encoder_layers(x)
#         x = self.dropout(x)
#         # Last 100 timesteps (unchanged logic from your original)
#         x = x[:, -100:, :]
#
#         out = self.output_proj(x)
#
#         if self.evidential:
#             out = out.view(*out.shape[:-1], -1, 4)
#             mu = out[..., 0]
#             lambda_ = F.softplus(out[..., 1]) + 1e-6
#             alpha = F.softplus(out[..., 2]) + 1.0 + 1e-6
#             beta = F.softplus(out[..., 3]) + 1e-6
#
#             return mu, lambda_, alpha, beta
#         return out

# class only_ml_block(nn.Module):
#     def __init__(self,enc_in,seq_len,enc_out,enc_layers,d_model=512,n_head=8,dropout=0.0,activation="gelu",
#                  device=torch.device('cuda:0')):
#         super(only_ml_block,self).__init__()
#
#         self.device = device
#         self.d_model = d_model
#
#         self.pos_encoder = PositionalEmbedding(d_model, max_len=seq_len)
#         self.input_projection = nn.Sequential(nn.Linear(enc_in, d_model),nn.GELU()) #From OBD length to D_Model
#
#         encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,nhead=n_head, dropout=dropout,activation=activation,batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer,num_layers=enc_layers)
#         self.output_proj = nn.Linear(d_model,enc_out)
#
#
#
#     def forward(self,enc_in):
#         B = enc_in.size(0)
#
#         x = self.input_projection(enc_in)
#         pos = self.pos_encoder(x)
#         x = x + pos
#
#         x = self.transformer_encoder(x)
#         pseudo_m = self.output_proj(x[:,-1,:])
#         return pseudo_m



########################################################GRU Comparision#######################################################
# class only_ml_block(nn.Module):
#     def __init__(self, enc_in, seq_len,seq_len_dec, enc_out, enc_layers, d_model=512, dropout=0.0,
#                  activation="gelu", device=torch.device('cuda:0')):
#         super(only_ml_block, self).__init__()
#
#         self.device = device
#         self.d_model = 150 #Described from thier paper
#         self.seq_len = seq_len
#         self.dec_len = seq_len_dec
#         self.enc_layers = enc_layers
#
#         # Input projection: project input feature dimension -> model dimension
#         self.input_projection = nn.Sequential(
#             nn.Linear(enc_in,self.d_model),
#             nn.GELU()
#         )
#
#         # GRU encoder-decoder (can stack multiple layers)
#         self.gru = nn.GRU(
#             input_size=self.d_model,
#             hidden_size=self.d_model,
#             num_layers=enc_layers,
#             batch_first=True,
#             dropout=dropout if enc_layers > 1 else 0.0
#         )
#
#         # Output projection: project from model dimension -> output dimension
#         self.output_proj = nn.Linear(self.d_model, enc_out)
#
#     def forward(self, enc_in):
#         """
#         Args:
#             enc_in: Input tensor of shape (B, seq_len, enc_in)
#         Returns:
#             pseudo_m: Regressed quantities of shape (B, seq_len, enc_out)
#         """
#         # Project input features
#         x = self.input_projection(enc_in)   # (B, seq_len, d_model)
#
#         # Pass through GRU
#         # x_out contains all hidden states for each time step
#         # h_n is the last hidden state for each layer
#         x_out, h_n = self.gru(x)
#
#         # Regression at each time step
#         pseudo_m = self.output_proj(x_out)  # (B, seq_len, enc_out)
#
#         return pseudo_m[:,-self.dec_len:,:]

########################################################LSTM Comparision#######################################################

# class only_ml_block(nn.Module):
#     def __init__(self, enc_in, seq_len, seq_len_dec, enc_out, enc_layers,
#                  d_model=512, dropout=0.0, activation="relu", device=torch.device('cuda:0')):
#         super(only_ml_block, self).__init__()
#
#         self.device = device
#         self.d_model = 128  # As described in the referenced paper
#         self.seq_len = seq_len
#         self.dec_len = seq_len_dec
#         self.enc_layers = enc_layers
#
#         # Input projection MLP: enc_in -> d_model
#         self.input_projection = nn.Sequential(
#             nn.Linear(enc_in, 64),
#             nn.ReLU(),
#             nn.Linear(64, self.d_model),
#             nn.ReLU()
#         )
#
#         # LSTM encoder-decoder
#         self.lstm = nn.LSTM(
#             input_size=self.d_model,
#             hidden_size=self.d_model,
#             num_layers=enc_layers,
#             batch_first=True,
#             dropout=dropout if enc_layers > 1 else 0.0
#         )
#
#         # Output projection MLP: d_model -> enc_out
#         self.output_proj = nn.Sequential(
#             nn.Linear(self.d_model, 64),
#             nn.ReLU(),
#             nn.Linear(64, enc_out)
#         )
#
#     def forward(self, enc_in):
#         """
#         Args:
#             enc_in: Input tensor of shape (B, seq_len, enc_in)
#         Returns:
#             pseudo_m: Regressed quantities of shape (B, seq_len_dec, enc_out)
#         """
#         # Project input features
#         x = self.input_projection(enc_in)  # (B, seq_len, d_model)
#
#         # Pass through LSTM
#         x_out, (h_n, c_n) = self.lstm(x)
#
#         # Regression at each time step
#         pseudo_m = self.output_proj(x_out)  # (B, seq_len, enc_out)
#
#         return pseudo_m[:, -self.dec_len:, :]

########################################################GRU Attention Comparision#######################################################


# class only_ml_block(nn.Module):
#     def __init__(self, enc_in, seq_len, seq_len_dec, enc_out, enc_layers,
#                  d_model=128, dropout=0.0, activation="relu", device=torch.device('cuda:0')):
#         super(only_ml_block, self).__init__()
#
#         self.device = device
#         self.d_model = 128
#         self.seq_len = seq_len
#         self.dec_len = seq_len_dec
#         self.enc_layers = 1
#
#         # ---- Input projection MLP: enc_in → d_model ----
#         self.input_projection = nn.Linear(enc_in,self.d_model)
#         self.y_tilde_proj = nn.Linear(self.d_model+self.d_model, self.d_model)
#
#         # ---- Encoder GRU ----
#         self.encoder_gru = nn.GRU(
#             input_size=self.d_model,
#             hidden_size=self.d_model,
#             num_layers=enc_layers,
#             batch_first=True,
#             dropout=dropout if enc_layers > 1 else 0.0
#         )
#
#         # ---- Attention parameters ----
#         self.Wd = nn.Linear(self.d_model, self.d_model, bias=False)
#         self.Ud = nn.Linear(self.d_model, self.d_model, bias=False)
#         self.vd = nn.Linear(self.d_model, 1, bias=False)
#
#         # ---- Decoder GRU ----
#         self.decoder_gru = nn.GRU(
#             input_size=self.d_model,
#             hidden_size=self.d_model,
#             num_layers=self.enc_layers,
#             batch_first=True,
#             dropout=dropout if enc_layers > 1 else 0.0
#         )
#
#         # ---- Output projection MLP ----
#         self.output_proj = nn.Linear(self.d_model * 2, enc_out)
#
#     def attention(self, decoder_hidden, encoder_outputs):
#         """
#         Compute attention weights and context vector.
#         Args:
#             decoder_hidden: (1, B, d_model)
#             encoder_outputs: (B, seq_len, d_model)
#         Returns:
#             context: (B, d_model)
#         """
#         # Ensure everything is on the same device
#         device = encoder_outputs.device
#         decoder_hidden = decoder_hidden.to(device)
#
#         # --- Extract the top layer hidden state (B, d_model)
#         # decoder_hidden shape: (num_layers, B, d_model)
#         dec_last = decoder_hidden[-1]  # (B, d_model)
#
#         # --- Expand for attention computation
#         # Repeat hidden state across the time dimension
#         seq_len = encoder_outputs.size(1)
#         dec_hidden_expanded = dec_last.unsqueeze(1).expand(-1, seq_len, -1)  # (B, seq_len, d_model)
#
#         # --- Apply linear projections safely
#         Wd_out = self.Wd(dec_hidden_expanded)  # (B, seq_len, d_model)
#         Ud_out = self.Ud(encoder_outputs)  # (B, seq_len, d_model)
#
#         # --- Use in-place ops sparingly; create new tensor for safety
#         energy_input = torch.tanh(Wd_out + Ud_out)  # (B, seq_len, d_model)
#
#         # --- Compute energy and attention weights
#         energy = self.vd(energy_input)  # (B, seq_len, 1)
#         attn_weights = F.softmax(energy, dim=1)  # (B, seq_len, 1)
#
#         # --- Compute context vector: weighted sum of encoder outputs
#         context = torch.sum(attn_weights * encoder_outputs, dim=1)  # (B, d_model)
#
#         return context, attn_weights
#
#     def forward(self, enc_in):
#         """
#         Args:
#             enc_in: Input tensor (B, seq_len, enc_in)
#         Returns:
#             pseudo_m: Predicted outputs (B, seq_len_dec, enc_out)
#         """
#         B = enc_in.size(0)
#
#         # 1️⃣ Input projection
#         x = self.input_projection(enc_in)  # (B, seq_len, d_model)
#
#         # 2️⃣ Encode sequence
#         encoder_outputs, hidden_enc = self.encoder_gru(x)  # encoder_outputs: (B, seq_len, d_model)
#
#         # Initialize decoder
#         decoder_hidden = hidden_enc[-self.decoder_gru.num_layers:].contiguous().to(self.device)
#         decoder_input = torch.zeros(B, 1, self.d_model, device=self.device)  # start token (zeros)
#         preds = []
#
#         # 3️⃣ Decode sequence (autoregressive)
#         for t in range(self.seq_len):
#             # Attention context
#             context, _ = self.attention(decoder_hidden, encoder_outputs)
#
#             # Combine previous output and context → ỹ_t
#             y_concat = torch.cat([decoder_input.squeeze(1), context], dim=-1)
#             y_tilde = self.y_tilde_proj(y_concat).unsqueeze(1)
#
#             # Decoder GRU step
#             dec_out, decoder_hidden = self.decoder_gru(y_tilde, decoder_hidden)
#
#             # Concatenate decoder output and context for final prediction
#             concat = torch.cat([dec_out.squeeze(1), context], dim=-1)  # (B, 2*d_model)
#             y_pred = self.output_proj(concat)  # (B, enc_out)
#
#             preds.append(y_pred.unsqueeze(1))
#             decoder_input = dec_out
#
#         # 4️⃣ Collect all decoder outputs
#         pseudo_m = torch.cat(preds, dim=1)  # (B, seq_len_dec, enc_out)
#
#         return pseudo_m[:, -self.dec_len:, :]