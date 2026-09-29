import argparse
import os
import torch
from datasets.preprocessing import load_config

from exp.kalman_ml_basic import exp_kalman_ml_basic
from exp.only_ml import Exp_only_ml


#Things to note - check config, dataprocessing, ml model, kalman filter and then batch processor
print('Check model config file')
parser = argparse.ArgumentParser(description='Kalman ML for Vehicle dynamic state estimation')
config = load_config("./configs/base.yaml")
parser.add_argument('--model', type=str, default='ml-kalman',
                    help='model of experiment, options: [ml-kalman, only-ml, phy-ml, final-ml]')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--iter', type=int, default=1, help='no of training runs')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--batch_size', type=int, default=1000, help='batch size') #only ml 1000 #kf250
parser.add_argument('--patience', type=int, default=5, help='Patience')
parser.add_argument('--lradj', type=str, default='type3',help='adjust learning rate')
parser.add_argument('--finetune', type=bool, default=False,help='Run kalman filter if true')
#Change learing rate in config as well.
parser.add_argument('--train_epochs', type=int, default=25, help='Train_epochs')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate') #learning rate is .001 for GRU
args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace('', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

model_map = {
    "only-ml": Exp_only_ml,
}

try:
    Exp = model_map[args.model]
except KeyError:
    raise ValueError(f"Unknown model type: {args.model}. "
                     f"Available options are: {list(model_map.keys())}")


for ii in range(args.iter):
    # setting record of experiments
    setting = '{}'.format(args.model)

    exp = Exp(config,args)  # set experiments
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    #exp.train(setting)

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, config)
    torch.cuda.empty_cache()