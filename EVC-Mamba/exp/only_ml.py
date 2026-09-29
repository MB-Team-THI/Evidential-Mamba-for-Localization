from datasets.dataset_loader import KalmanAwareDataset
from datasets.preprocessing import load_config,preprocess_dataset

from thop import profile
import torch
from exp.kalman_ml_basic import exp_kalman_ml_basic
from models.only_ml_model import only_ml_block

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metric import metric
import io
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings

warnings.filterwarnings('ignore')

class EvidentialLoss(nn.Module):
    def __init__(self, coeff=1e-2):
        super().__init__()
        self.coeff = coeff

    def forward(self, outputs, y):
        mu, v, alpha, beta = outputs  # unpack

        error = y - mu

        # NLL
        nll = (
            0.5 * torch.log(torch.pi / v)
            - alpha * torch.log(2 * beta * (1 + v))
            + (alpha + 0.5) * torch.log(v * error**2 + 2 * beta * (1 + v))
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )
        # Regularizer
        reg = error**2 * (2 * v + alpha)  # smoother than abs
        loss = nll + self.coeff * reg
        return loss.mean()

class Exp_only_ml(exp_kalman_ml_basic):
    def __init__(self, config,args):
        super().__init__(config, args)
        self.config = config
        self.args = args

    def _build_only_ml_model(self):

        cfg = self.config["model"]
        model = only_ml_block(
                enc_in=cfg["enc_in"],
                seq_len=cfg["seq_len"],
                enc_out=cfg["enc_out"],
                #seq_len_dec=cfg["seq_len_dec"],
                enc_layers=cfg["enc_layers"],
                d_model=cfg["d_model"],
                dropout=cfg["dropout"],
                activation=cfg["activation"],
                evidential = cfg["evidential"],
            )

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        df = preprocess_dataset(self.config)
        data_set = KalmanAwareDataset(df,self.config,flag)
        print(flag, len(data_set))

        if flag == 'test' or flag == "val":
            shuffle_flag = False;
            drop_last = True;
            batch_size = self.args.batch_size;

        else:
            shuffle_flag = True;
            drop_last = True;
            batch_size = self.args.batch_size;

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=self.args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = torch.optim.AdamW(self.only_ml_model.parameters(), lr=self.config["training"]["learning_rate"], weight_decay=0.01)
        return model_optim


    def _select_criterion(self):
        if self.config["model"]["evidential"]:
            return EvidentialLoss(coeff=1e-2)
        else:
            return nn.MSELoss()



    def vali(self, vali_data, vali_loader, criterion):
        self.only_ml_model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_p) in enumerate(vali_loader):
            device = torch.device(f"cuda:{self.args.gpu}" if torch.cuda.is_available() else "cpu")
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_p = batch_p.float().to(device)
            pred, true = self._process_one_batch_ML(vali_data, batch_x, batch_y, batch_p)
            if self.config["model"]["evidential"]:
                pred = tuple(t.detach().cpu() for t in pred)
                true = true.detach().cpu()
                loss = criterion(pred,true)
            else:
                loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.only_ml_model.train()
        return total_loss

    def train(self, setting):
        torch.autograd.set_detect_anomaly(True)
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        train_iters = 0  # Variable created for logging.
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        print("Check weighted loss implementation")
        criterion = self._select_criterion()

        total_params = sum(p.numel() for p in self.only_ml_model.parameters() if p.requires_grad)
        print(f'Total parameters: {total_params}')

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.only_ml_model.train()
            epoch_time = time.time()
            device = torch.device(f"cuda:{self.args.gpu}" if torch.cuda.is_available() else "cpu")
            for i, (batch_x, batch_y, batch_p) in enumerate(train_loader):

                iter_count += 1

                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_p = batch_p.float().to(device)

                model_optim.zero_grad()
                pred, true = self._process_one_batch_ML(train_data, batch_x, batch_y, batch_p)
                loss = criterion(pred, true)
                train_iters += 1
                train_loss.append(loss.item())

                if (i + 1) % 10 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            del batch_x, batch_y, batch_p, pred, true, loss
            torch.cuda.empty_cache()
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.only_ml_model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if self.args.lradj == 'type3':
                self.args.lradj= "type1"
            adjust_learning_rate(model_optim, epoch + 1, self.args)

            best_model_path = path + '/' + 'checkpoint.pth'
        self.only_ml_model.load_state_dict(torch.load(best_model_path))
        return self.only_ml_model

    #During testing there was error for multi GPU training saved model to single GPU inference
    #I think error is due to model not running properly in loaded mode. Single GPU inference is best
    #This script loads model in single GPU and gets results
    def test(self, setting, config):
        test_data, test_loader = self._get_data(flag='test')
        # Use this block if u want to load best model and load results (comment out three lines)
        # Be careful with inverse scaler as datasets should be same on which this model was trained
        path = os.path.join(self.args.checkpoints, setting)
        best_model_path = path + '/' + 'checkpoint_Evidential.pth'
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        history_size = config["data"]['history_size']
        #unwrapping data parrallel.
        if isinstance(self.only_ml_model, torch.nn.DataParallel):
            self.only_ml_model = self.only_ml_model.module
        else:
            self.only_ml_model = self.only_ml_model
        state_dict = torch.load(best_model_path, map_location='cuda:0')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                name = k[7:]  # remove 'module.' prefix
            else:
                name = k
            new_state_dict[name] = v

        # Load the cleaned state dict
        self.only_ml_model.load_state_dict(new_state_dict)

        # Move the model to single GPU device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.only_ml_model.to(device)

        # Set to evaluation mode
        self.only_ml_model.eval()
        preds = []
        trues = []
        gt_poses = []
        variances = []
        start_time = time.time()
        total_batches = len(test_loader)
        percent_step = max(1, total_batches // 20)
        for i, (batch_x, batch_y, batch_p) in enumerate(test_loader):
            if (i + 1) % percent_step == 0 or i == 0:
                print(f"Progress: {100 * (i + 1) / total_batches:.1f}% ({i + 1}/{total_batches} batches)")
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_p = batch_p.float().to(device)

            pred, true = self._process_one_batch_ML(
                test_data, batch_x, batch_y, batch_p)


            # This part of the code was modified to reproject the data back to slip angles (inversion of the conversion process)
            if self.config["model"]["evidential"]:
                mu, lambda_, alpha, beta = pred  # tensors
                eps = 1e-6
                # Optional safety (important)
                alpha = torch.clamp(alpha, min=1.01)
                lambda_ = torch.clamp(lambda_, min=1e-6)

                # Variances
                aleatoric_var = beta / (alpha - 1 + eps)
                epistemic_var = beta / (lambda_ * (alpha - 1 + eps))
                total_var = aleatoric_var + epistemic_var
                pred = mu
            true_rescaled = test_data.scaler_target.inverse_transform((true[:, -1, :])).unsqueeze(-1) # modified
            if torch.isnan(pred).any():
                print("error")
            pred_rescaled = test_data.scaler_target.inverse_transform((pred[:, -1, :])).unsqueeze(-1)
            std = torch.tensor(test_data.scaler_target.std,device=total_var.device,dtype=total_var.dtype)
            variance_rescaled = (std**2 * (total_var[:, -1, :])).unsqueeze(-1)
            pose = batch_p[:, -1, :].unsqueeze(-1)
            preds.append(pred_rescaled.detach().cpu().numpy())
            trues.append(true_rescaled.detach().cpu().numpy())
            variances.append(variance_rescaled.detach().cpu().numpy())
            gt_poses.append(pose.detach().cpu().numpy())
        end_time = time.time()
        run_time = (end_time - start_time) / len(test_loader)
        print(run_time)
        preds = np.array(preds)
        trues = np.array(trues)
        gt_poses = np.array(gt_poses)
        variances = np.array(variances)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1]).squeeze(-1)
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1]).squeeze(-1)
        variances = variances.reshape(-1, variances.shape[-2], variances.shape[-1]).squeeze(-1)
        gt_poses = gt_poses.reshape(-1, gt_poses.shape[-2], gt_poses.shape[-1]).squeeze(-1)
        print('test shape:', preds.shape, trues.shape, gt_poses.shape)

        # Result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, max_ae = metric(preds[:, :], trues)
        print('mse:{}, mae:{},max_ae:{}'.format(mse, mae, max_ae))

        np.save(folder_path + f'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + f'pred.npy', preds)
        np.save(folder_path + f'true.npy', trues)
        np.save(folder_path + f'variance.npy', variances)
        np.save(folder_path + f'gt_poses.npy', gt_poses)
        return


    def _process_one_batch_ML(self, dataset, batch_x, batch_y, batch_p):
        window_length = self.config["data"]["window_size"]
        history_size = self.config["data"]["history_size"]
        print("model flops")
        flops, params = profile(self.only_ml_model, inputs=(batch_x[0:1, :, :],))
        print(f"FLOPs: {flops / 1e9:.4f} GFLOPs")
        print(f"Params: {params / 1e6:.2f} M")

        if self.config["model"]["evidential"]:
            mu_t,lambda_t,alpha_t,beta_t = self.only_ml_model(batch_x[:, :, :])
            pred = (mu_t,lambda_t,alpha_t,beta_t)
        else:
            pred = self.only_ml_model(batch_x[:,:,:])
        return pred, batch_y[:,-100:,:]


# The dataset object contains the statistics of the entire dataset and the scaler transform carried out in it
# We can use it to inverse the scaler transform and get actual values.
# This is implemented for test function.