import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import os
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



#Basic configuration for the model
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

seed_torch(20)
#Change things here for running code
class config():
    def __init__(self):
        self.batch_size = 1000
        self.root_path = "./data/OBD_ADMA/"
        #self.data_path = "Informer_dataset_file_fivehourdataset_new.csv"
        self.data_path = "stanford_combined_removed_massive_standstill.csv"
        self.size=[100, 5] #Input and output size
        self.features="MS"
        self.target="Correvit_slip_angle_COG_corrvittiltcorrected"
        self.inverse=False
        self.cols=False #All columns are used for the dataset.
        #self.input_dim = 9 #OBD
        self.input_dim = 8 #stanford
        self.hidden_dim = 100
        self.output_dim = 1
        self.output_length = 5
        self.num_layers = 2
        self.sequence_length = 100
        self.num_workers = 0
        self.optimizer = "adam"
        self.learning_rate = 0.0005

config_file = config()

folder_path = "data/rev-sted"
train_csv_files = [f'Recording_{i}_ReVSTED.csv' for i in range(1,6,1)]
test_csv_files = [f'Recording_{i}_ReVSTED.csv' for i in range(6,8,1)]
train_df_list = [pd.read_csv(os.path.join(folder_path,file)) for file in train_csv_files]
test_df_list = [pd.read_csv(os.path.join(folder_path,file)) for file in test_csv_files]
train_df = pd.concat(train_df_list, ignore_index=True)
test_df = pd.concat(test_df_list, ignore_index=True)
interested_columns = ["speedo_obd","LatAcc_obd",'SW_pos_obd','Yawrate_obd', 'VelFR_obd','VelFL_obd','VelRR_obd','VelRL_obd','Correvit_vehiclesideslip_angle_COG']

train_dataset = train_df[interested_columns].dropna().reset_index(drop=True)
test_dataset = test_df[interested_columns].dropna().reset_index(drop=True)

#Standard scaler on training features
scaler = StandardScaler()
scaler.fit(train_dataset.values)

class SequenceForecastDataset(Dataset):
    def __init__(self, full_df, input_window=100, output_window=5, scaler=None, target_col='Correvit_vehiclesideslip_angle_COG'):
        self.input_window = input_window
        self.output_window = output_window
        self.target_col = target_col
        self.columns = full_df.columns.tolist()
        self.target_idx = self.columns.index(target_col)

        # Normalize all features including the target
        self.features = full_df.values.astype(np.float32)
        if scaler is not None:
            self.features = scaler.transform(self.features)
        self.num_samples = len(full_df) - input_window - output_window + 1
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        X_seq = self.features[idx: idx + self.input_window]
        X_seq = np.delete(X_seq, self.target_idx, axis=1)
        y_seq = self.features[idx + self.input_window: idx + self.input_window + self.output_window, self.target_idx]
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32)


class lstm_encoder(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,dropout=0.0):
        super(lstm_encoder,self).__init__()
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,dropout=dropout)

    def forward(self,x):
        _,(hidden,cell) = self.lstm(x)
        return hidden,cell


class lstm_decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers,dropout):
        super(lstm_decoder, self).__init__()
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True,dropout=dropout)
        self.fc = nn.Linear(hidden_size,output_size)
    def forward(self, decoder_input,hidden,cell):
        output, (hidden, cell) = self.lstm(decoder_input, (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell


class LSTMEncoderDecoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers,target_length,dropout=0.0):
        super(LSTMEncoderDecoder, self).__init__()
        self.encoder = lstm_encoder(input_size, hidden_size, num_layers,dropout)
        self.decoder = lstm_decoder(output_size, hidden_size, num_layers,dropout)
        self.output_length = target_length
        self.output_size = output_size

    def forward(self, encoder_input):
        batch_size = encoder_input.size(0)
        hidden, cell = self.encoder(encoder_input)
        decoder_input = torch.zeros(batch_size, 1, self.output_size).to(device)
        outputs = []

        for t in range(self.output_length):
            prediction, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs.append(prediction)
            decoder_input = prediction  # Feed the prediction back as input

        # Stack predictions: (batch_size, target_length, output_size)
        return torch.cat(outputs, dim=1).squeeze(-1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMEncoderDecoder(config_file.input_dim, config_file.output_dim, config_file.hidden_dim,config_file.num_layers,config_file.output_length,dropout=0.2).to(device)
criterion = nn.MSELoss()
#optimizer = torch.optim.SGD(model.parameters(),config_file.learning_rate,momentum=0.9, weight_decay=1e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=config_file.learning_rate, weight_decay=1e-4)
n_epochs = 50
early_stopping_patience = 10
best_val_loss = float('inf')
patience_counter = 0
model_save_dir = os.path.join(os.getcwd(), "results/LSTM")
model_save_path = os.path.join(model_save_dir, "lstm_model_adam_ours.pth")
# Ensure the directory exists
os.makedirs(model_save_dir, exist_ok=True)
history = []

train_data = SequenceForecastDataset(train_dataset, input_window=100, output_window=5, scaler=scaler, target_col='Correvit_vehiclesideslip_angle_COG')
test_data = SequenceForecastDataset(test_dataset, input_window=100, output_window=5, scaler=scaler, target_col='Correvit_vehiclesideslip_angle_COG')

# Optionally split train_data further for validation
from torch.utils.data import random_split

train_size = int(0.9 * len(train_data))
val_size = len(train_data) - train_size
train_data, val_data = random_split(train_data, [train_size, val_size])

# Dataloaders
train_loader = DataLoader(train_data, batch_size=1000, shuffle=True, drop_last=True)
vali_loader = DataLoader(val_data, batch_size=1000, shuffle=False, drop_last=False)
test_loader = DataLoader(test_data, batch_size=1000, shuffle=False, drop_last=False)


#Training
# for epoch in range(n_epochs):
#     print(epoch)
#     iter_count = 0
#     train_loss = []
#     model.train()
#     for i, (batch_x, batch_y) in enumerate(train_loader):
#         iter_count += 1
#         if iter_count % 10 == 0:
#             print(f"Iteration {iter_count}")
#         batch_x = batch_x.to(device).float()
#         batch_y = batch_y.to(device).float()
#         pred = model(batch_x[:,:,:].float()) #Feeding OBD data and preventing slip angle
#         true = batch_y[:, :].float() #Taking the correvit slip angle
#         loss = criterion(pred, true)
#         train_loss.append(loss.item())
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     model.eval()
#     val_loss = []
#     for i , (batch_x, batch_y) in enumerate(vali_loader):
#         batch_x = batch_x.to(device).float()
#         batch_y = batch_y.to(device).float()
#         pred = model(batch_x[:, :,:].float())
#         true = batch_y[:,:].float()
#         loss = criterion(pred, true)
#         val_loss.append(loss.item())
#     val_loss = np.average(val_loss)
#     history.append(val_loss)
#     print(val_loss)
#
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         patience_counter = 0
#         print('Validation loss improved.Saving model..')
#         torch.save(model.state_dict(),model_save_path)
#     else:
#         patience_counter += 1
#         print(f'No improvement in validation loss for {patience_counter}')
#
#     if patience_counter >= early_stopping_patience:
#         print('Early stopping is triggered. Training stoppped')
#         break


#Test
print("Testing mode for the model")
model.load_state_dict(torch.load(model_save_path))
model.eval()
preds = []
trues = []

def inverse_transform_target(scaled_target_tensor, scaler, target_col_idx):
    true_np = scaled_target_tensor.detach().cpu().numpy()
    shape = true_np.shape
    true_flat = true_np.reshape(-1, 1)
    num_features = scaler.mean_.shape[0]

    full_data = np.zeros((true_flat.shape[0], num_features))
    full_data[:, target_col_idx] = true_flat[:, 0]

    full_inv = scaler.inverse_transform(full_data)
    target_inv = full_inv[:, target_col_idx].reshape(shape)
    return torch.tensor(target_inv)

target_idx = test_dataset.columns.get_loc('Correvit_vehiclesideslip_angle_COG')
for i , (batch_x, batch_y) in enumerate(test_loader):
    batch_x = batch_x.to(device).float()
    batch_y = batch_y.to(device).float()
    pred = model(batch_x[:, :, :].float())
    true = batch_y[:, :].float()

    # This part of the code was modified to reproject the data back to slip angles (inversion of the conversion process)
    true_rescaled = inverse_transform_target(true, scaler, target_idx).unsqueeze(-1)
    pred_rescaled = inverse_transform_target(pred, scaler, target_idx).unsqueeze(-1)

    preds.append(pred_rescaled[:, 0].detach().cpu().numpy())
    trues.append(true_rescaled[:, 0].detach().cpu().numpy())

preds = np.concatenate(preds, axis=0)
trues = np.concatenate(trues, axis=0)

print('test shape:', preds.shape, trues.shape)
plt.figure(figsize=(12, 6))
plt.plot(trues, label='True Vehicle Sideslip Angle', color='blue')
plt.plot(preds, label='Predicted Sideslip Angle', color='red', alpha=0.7)
plt.xlabel('Sample Index')
plt.ylabel('Slip Angle (deg)')
plt.title('Predicted vs True Slip Angle (0th Output Step)')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()

# 2. Save to CSV
results_df = pd.DataFrame({
    'Correvit_SlipAngle': trues.flatten(),
    'Predicted_SlipAngle': preds.flatten()
})

results_df.to_csv('OBD_VSA_predictions.csv', index=False)


