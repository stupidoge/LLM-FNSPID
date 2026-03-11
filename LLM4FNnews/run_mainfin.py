"""
this is LLM4FN model, includes double attention network and contrative learning for finance dataest
"""

import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import pandas as pd
import gc

from models import Autoformer, DLinear, TimeLLM, LLM4FNnews, LLM4FN, TimeLLM_double_attention

from data_provider.data_factory import data_provider, data_provider_fin
import time
import random
import numpy as np
import os

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


from utils.tools import del_files, EarlyStopping, EarlyStoppingFin, adjust_learning_rate, vali, vali_contrast, vali_contrast_fin, load_content

parser = argparse.ArgumentParser(description='Time-LLM')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

companies = ['aal.csv', 'AAPL.csv', 'ABBV.csv', 'AMD.csv', 'amgn.csv', 'AMZN.csv', 'BABA.csv',
            'bhp.csv', 'bidu.csv', 'biib.csv', 'BRK-B.csv', 'C.csv', 'cat.csv', 'cmcsa.csv', 'cmg.csv',
            'cop.csv', 'COST.csv', 'crm.csv', 'CVX.csv', 'dal.csv', 'DIS.csv', 'ebay.csv', 'GE.csv',
            'gild.csv', 'gld.csv', 'GOOG.csv', 'gsk.csv', 'INTC.csv', 'KO.csv', 'mrk.csv', 'MSFT.csv',
            'mu.csv', 'nke.csv', 'nvda.csv', 'orcl.csv', 'pep.csv', 'pypl.csv', 'qcom.csv', 'QQQ.csv',
            'SBUX.csv', 'T.csv', 'tgt.csv', 'tm.csv', 'TSLA.csv', 'TSM.csv', 'uso.csv', 'v.csv', 'WFC.csv',
            'WMT.csv', 'xlf.csv']


def read_csv_case_insensitive(path, filename):
    # 分割文件名和扩展名
    name_part, extension = os.path.splitext(filename)
    # 准备大小写可能的文件名
    filenames_to_try = [name_part.lower() + extension, name_part.upper() + extension]

    # 尝试读取文件
    for fname in filenames_to_try:
        full_path = os.path.join(path, fname)
        if os.path.exists(full_path):
            return pd.read_csv(full_path)

    # 如果所有尝试都失败，抛出异常
    raise FileNotFoundError(f"Could not find the file using any case variations in {path}.")

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, DLinear]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='d',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768


# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='constant', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--mixed_precision', type=str, default='bf16',
                    help='Set the mixed precision mode, options: ["bf16", "fp16", None]')

args = parser.parse_args()



maes = []
mses = []
for data_path in companies:
    args.learning_rate = 0.02
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin,
                              mixed_precision=args.mixed_precision)
    args.data_path = data_path
    file_path_ts = os.path.join(args.root_path, args.data_path)
    # file_path_news = os.path.join('../FNSPID_dataset/news_summary', args.data_path)
    df_raw = pd.read_csv(file_path_ts).dropna()
    # df_new = pd.read_csv(file_path_news)
    df_new = read_csv_case_insensitive('../FNSPID_dataset/news_summary', args.data_path)

    df_new['date'] = pd.to_datetime(df_new['date'])  # 将日期列转换为日期时间格式

    num_samples = len(df_raw) * 0.10

    # if num_samples <= 400:
    #     args.seq_len = 50
    #     args.label_len = 20
    #     args.pred_len = 15
    # else:
    #     args.seq_len = 336
    #     args.label_len = 168
    #     args.pred_len = 96

    if num_samples <= 400:
        args.seq_len = 50
        args.label_len = 20
        args.pred_len = 3
        args.batch_size = 64
    else:
        args.seq_len = 50
        args.label_len = 20
        args.pred_len = 3
        args.batch_size = 256


    for ii in range(args.itr):
        # setting record of experiments
        mses_itr = []
        maes_itr = []
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
            args.task_name,
            args.model_id,

            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.des, ii)

        train_data, train_loader = data_provider_fin(args, 'train', df_raw)
        vali_data, vali_loader = data_provider_fin(args, 'val', df_raw)
        test_data, test_loader = data_provider_fin(args, 'test', df_raw)

        if args.model == 'Autoformer':
            model = Autoformer.Model(args).float()
        elif args.model == 'DLinear':
            model = DLinear.Model(args).float()
        else:
            # model = TimeLLM.Model(args).float()
            # model = LLM4FNnews.Model(args, df_news = df_new).float()
            model = TimeLLM_double_attention.Model(args).float()
            model_path = "pretrain_checkpoints/pretrain2_long_term_forecast_LLM4FN_50_3_TimeLLM_fin_ftM_sl50_ll20_pl3_dm32_nh8_el2_dl1_df128_fc3_ebtimeFF_'Exp'_0-'TimeLLM-LLM4FN'/checkpoint"
            # Load the pretrained model
            model.load_state_dict(torch.load(model_path, map_location=accelerator.device))

        path = os.path.join(args.checkpoints,
                            setting + '-' + args.model_comment)  # unique checkpoint saving path
        args.content = load_content(args)
        if not os.path.exists(path) and accelerator.is_local_main_process:
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStoppingFin(accelerator=accelerator, patience=args.patience)

        trained_parameters = []
        for p in model.parameters():
            if p.requires_grad is True:
                trained_parameters.append(p)

        model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

        if args.lradj == 'COS':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
        else:
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=args.pct_start,
                                                epochs=args.train_epochs,
                                                max_lr=args.learning_rate)

        criterion = nn.MSELoss()
        mae_metric = nn.L1Loss()

        train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
            train_loader, vali_loader, test_loader, model, model_optim, scheduler)

        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(args.train_epochs):
            iter_count = 0
            train_loss = []
            # print(torch.cuda.memory_allocated())
            # print(torch.cuda.max_memory_allocated())
            model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_seq)  in tqdm(enumerate(train_loader)):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(accelerator.device)
                batch_y = batch_y.float().to(accelerator.device)
                batch_x_mark = batch_x_mark.float().to(accelerator.device)
                batch_y_mark = batch_y_mark.float().to(accelerator.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(
                    accelerator.device)
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                    accelerator.device)

                # encoder - decoder
                if args.use_amp:
                    with torch.cuda.amp.autocast():
                        if args.output_attention:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if args.features == 'MS' else 0
                        outputs = outputs[:, -args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if args.output_attention:
                        # outputs, contrast_loss = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        outputs, contrast_loss = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_seq)
                    else:
                        # outputs, contrast_loss = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs, contrast_loss = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_seq)
                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -args.pred_len:, f_dim:]
                    loss_mse = criterion(outputs, batch_y)
                    # loss = criterion(outputs, batch_y)/criterion(outputs, batch_y).detach() + contrast_loss/contrast_loss.detach()
                    w = 0.3

                    loss = (1-w) * loss_mse + contrast_loss * w
                    train_loss.append(loss_mse.item())

                if (i + 1) % 100 == 0:
                    accelerator.print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    model_optim.step()
                if args.lradj == 'TST':
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                    scheduler.step()

            del outputs, batch_x, batch_y, dec_inp, batch_y_mark

            accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            # vali_loss, vali_mae_loss = vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric)
            # vali_loss, vali_mae_loss = vali_contrast(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric)
            vali_loss, vali_mae_loss = vali_contrast_fin(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric)
            # test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
            test_loss, test_mae_loss = vali_contrast_fin(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
            vali_loss, vali_mae_loss, test_loss, test_mae_loss = vali_loss.item(), vali_mae_loss.item(), test_loss.item(), test_mae_loss.item()

            accelerator.print(
                "Epoch: {0} |Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} MAE Loss: {4:.7f}".format(
                    epoch + 1, train_loss, vali_loss, test_loss, test_mae_loss))

            early_stopping(vali_loss, test_loss, test_mae_loss, model, path)
            if early_stopping.early_stop:
                accelerator.print("Early stopping")
                break

            if args.lradj != 'TST':
                if args.lradj == 'COS':
                    scheduler.step()
                    accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                else:
                    if epoch == 0:
                        args.learning_rate = model_optim.param_groups[0]['lr']
                        accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)

            else:
                accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
            # print(torch.cuda.memory_allocated())
            # print(torch.cuda.max_memory_allocated())

        print("data = {}, best test mse = {:.4f}, best test mae = {:.4f}".format(data_path, early_stopping.test_mse_min, early_stopping.test_mae_min))
        mses_itr.append(early_stopping.test_mse_min)
        maes_itr.append(early_stopping.test_mae_min)
        # model, train_loader, vali_loader, test_loader, model_optim, scheduler = None, None, None, None, None, None
        del model, train_loader, vali_loader, test_loader, model_optim, scheduler
        torch.cuda.empty_cache()
        gc.collect()
        # print(torch.cuda.memory_summary())

    print("mse_mean_itr = {:.4f}, mse_std_itr = {:.4f}".format(np.mean(mses_itr), np.std(mses_itr)))
    print("mae_mean_itr = {:.4f}, mae_std_itr = {:.4f}".format(np.mean(maes_itr), np.std(maes_itr)))
    mses.append(np.mean(mses_itr))
    maes.append(np.mean(maes_itr))


    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        # path = './checkpoints'  # unique checkpoint saving path
        del_files(path)  # delete checkpoint files
        accelerator.print('success delete checkpoints')



print("mse_mean_all = {:.4f}, mse_std_all = {:.4f}".format(np.mean(mses), np.std(mses)))
print("mae_mean_all = {:.4f}, mae_std_all = {:.4f}".format(np.mean(maes), np.std(maes)))