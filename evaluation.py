import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, cohen_kappa_score
from visualize import Draw_Confmat, Snr_Acc_Plot
import pandas as pd  # 新增导入
import os


def Run_Eval1(model,
             sig_test,
             lab_test,
             SNRs,
             test_idx,
             cfg,
             logger,
             best_type):
    model.eval()

    snrs = list(np.unique(SNRs))
    mods = list(cfg.classes.keys())
    # 使用 `.decode()` 转换字节串为普通字符串，仅对字节串类型执行解码
    mods = [mod.decode('utf-8') if isinstance(mod, bytes) else mod for mod in mods]

    Confmat_Set = np.zeros((len(snrs), len(mods), len(mods)), dtype=int)
    Accuracy_list = np.zeros(len(snrs), dtype=float)
    Accuracy_per_mod = {mod: np.zeros(len(snrs), dtype=float) for mod in mods}  # 存储每个调制类型在每个SNR下的准确率

    pre_lab_all = []
    label_all = []

    for snr_i, snr in tqdm(enumerate(snrs), total=len(snrs), desc="Processing SNRs"):
        test_SNRs = map(lambda x: SNRs[x], test_idx)
        test_SNRs = list(test_SNRs)
        test_SNRs = np.array(test_SNRs).squeeze()
        test_sig_i = sig_test[np.where(np.array(test_SNRs) == snr)]
        test_lab_i = lab_test[np.where(np.array(test_SNRs) == snr)]

        Sample = torch.chunk(test_sig_i, cfg.test_batch_size, dim=0)
        Label = torch.chunk(test_lab_i, cfg.test_batch_size, dim=0)

        pred_i = []
        label_i = []

        # 处理每个批次
        for sample, label in zip(Sample, Label):
            sample = sample.to(cfg.device)
            logit, _ = model(sample)
            pre_lab = torch.argmax(logit, 1).cpu()
            pred_i.append(pre_lab)
            label_i.append(label)

        pred_i = np.concatenate(pred_i)
        label_i = np.concatenate(label_i)

        pre_lab_all.append(pred_i)
        label_all.append(label_i)

        Confmat_Set[snr_i, :, :] = confusion_matrix(label_i, pred_i)
        Accuracy_list[snr_i] = accuracy_score(label_i, pred_i)

        # 计算每个调制类型的准确率
        for mod_i, mod in enumerate(mods):
            # 提取当前调制类型的真实标签和预测标签
            mod_mask = (label_i == mod_i)  # 获取该调制类型的标签
            mod_label_i = label_i[mod_mask]
            mod_pred_i = pred_i[mod_mask]

            # 计算该调制类型的准确率
            if len(mod_label_i) > 0:  # 确保该调制类型的样本数大于0
                mod_accuracy = accuracy_score(mod_label_i, mod_pred_i)
                Accuracy_per_mod[mod][snr_i] = mod_accuracy



    pre_lab_all = np.concatenate(pre_lab_all)
    label_all = np.concatenate(label_all)

    F1_score = f1_score(label_all, pre_lab_all, average='macro')
    kappa = cohen_kappa_score(label_all, pre_lab_all)
    acc = np.mean(Accuracy_list)

    logger.info(f'overall accuracy is: {acc}')
    logger.info(f'macro F1-score is: {F1_score}')
    logger.info(f'kappa coefficient is: {kappa}')

    ########################################################################

    csv_path = cfg.result_dir + '/' + best_type + '/accuracy_results.csv'  # 默认保存路径

    # 创建保存目录（如果不存在）
    os.makedirs(os.path.dirname(csv_path), exist_ok=True) if os.path.dirname(csv_path) else None

    # 创建DataFrame
    df = pd.DataFrame({
        'SNR': snrs,
        'Accuracy': Accuracy_list
    })

    # 保存到CSV
    df.to_csv(csv_path, index=False)
    logger.info(f'SNR和准确率数据已保存到: {csv_path}')

    #######################################################################

    # 保存每个调制类型的准确率到CSV
    for mod in mods:
        mod_accuracy_df = pd.DataFrame({
            'SNR': snrs,
            f'{mod}_Accuracy': Accuracy_per_mod[mod]
        })

        mod_accuracy_csv_result = cfg.result_dir + '/' + best_type
        mod_accuracy_csv_result_dir = '%s/mods_acc' % mod_accuracy_csv_result
        mod_accuracy_csv_path = os.path.join(mod_accuracy_csv_result_dir, f'{mod}_accuracy_results.csv')
        # 创建保存目录（如果不存在）
        os.makedirs(os.path.dirname(mod_accuracy_csv_path), exist_ok=True) if os.path.dirname(mod_accuracy_csv_path) else None

        mod_accuracy_df.to_csv(mod_accuracy_csv_path, index=False)
        logger.info(f'{mod}准确率数据已保存到: {mod_accuracy_csv_path}')

    #######################################################################


    if cfg.Draw_Confmat is True:
        Draw_Confmat1(Confmat_Set, snrs, cfg, best_type)
    if cfg.Draw_Acc_Curve is True:
        Snr_Acc_Plot1(Accuracy_list, Confmat_Set, snrs, cfg, best_type)



def Run_Eval(model,
             sig_test,
             lab_test,
             SNRs,
             test_idx,
             cfg,
             logger):
    model.eval()

    snrs = list(np.unique(SNRs))
    mods = list(cfg.classes.keys())
    # 使用 `.decode()` 转换字节串为普通字符串，仅对字节串类型执行解码
    mods = [mod.decode('utf-8') if isinstance(mod, bytes) else mod for mod in mods]

    Confmat_Set = np.zeros((len(snrs), len(mods), len(mods)), dtype=int)
    Accuracy_list = np.zeros(len(snrs), dtype=float)
    Accuracy_per_mod = {mod: np.zeros(len(snrs), dtype=float) for mod in mods}  # 存储每个调制类型在每个SNR下的准确率

    pre_lab_all = []
    label_all = []

    for snr_i, snr in tqdm(enumerate(snrs), total=len(snrs), desc="Processing SNRs"):
        test_SNRs = map(lambda x: SNRs[x], test_idx)
        test_SNRs = list(test_SNRs)
        test_SNRs = np.array(test_SNRs).squeeze()
        test_sig_i = sig_test[np.where(np.array(test_SNRs) == snr)]
        test_lab_i = lab_test[np.where(np.array(test_SNRs) == snr)]

        Sample = torch.chunk(test_sig_i, cfg.test_batch_size, dim=0)
        Label = torch.chunk(test_lab_i, cfg.test_batch_size, dim=0)

        pred_i = []
        label_i = []

        # 处理每个批次
        for sample, label in zip(Sample, Label):
            sample = sample.to(cfg.device)
            logit, _ = model(sample)
            pre_lab = torch.argmax(logit, 1).cpu()
            pred_i.append(pre_lab)
            label_i.append(label)

        pred_i = np.concatenate(pred_i)
        label_i = np.concatenate(label_i)

        pre_lab_all.append(pred_i)
        label_all.append(label_i)

        Confmat_Set[snr_i, :, :] = confusion_matrix(label_i, pred_i)
        Accuracy_list[snr_i] = accuracy_score(label_i, pred_i)

        # 计算每个调制类型的准确率
        for mod_i, mod in enumerate(mods):
            # 提取当前调制类型的真实标签和预测标签
            mod_mask = (label_i == mod_i)  # 获取该调制类型的标签
            mod_label_i = label_i[mod_mask]
            mod_pred_i = pred_i[mod_mask]

            # 计算该调制类型的准确率
            if len(mod_label_i) > 0:  # 确保该调制类型的样本数大于0
                mod_accuracy = accuracy_score(mod_label_i, mod_pred_i)
                Accuracy_per_mod[mod][snr_i] = mod_accuracy



    pre_lab_all = np.concatenate(pre_lab_all)
    label_all = np.concatenate(label_all)

    F1_score = f1_score(label_all, pre_lab_all, average='macro')
    kappa = cohen_kappa_score(label_all, pre_lab_all)
    acc = np.mean(Accuracy_list)

    logger.info(f'overall accuracy is: {acc}')
    logger.info(f'macro F1-score is: {F1_score}')
    logger.info(f'kappa coefficient is: {kappa}')

    ########################################################################

    csv_path = cfg.result_dir  + '/accuracy_results.csv'  # 默认保存路径

    # 创建保存目录（如果不存在）
    os.makedirs(os.path.dirname(csv_path), exist_ok=True) if os.path.dirname(csv_path) else None

    # 创建DataFrame
    df = pd.DataFrame({
        'SNR': snrs,
        'Accuracy': Accuracy_list
    })

    # 保存到CSV
    df.to_csv(csv_path, index=False)
    logger.info(f'SNR和准确率数据已保存到: {csv_path}')

    #######################################################################

    # 保存每个调制类型的准确率到CSV
    for mod in mods:
        mod_accuracy_df = pd.DataFrame({
            'SNR': snrs,
            f'{mod}_Accuracy': Accuracy_per_mod[mod]
        })


        mod_accuracy_csv_result_dir = '%s/mods_acc' % cfg.result_dir
        mod_accuracy_csv_path = os.path.join(mod_accuracy_csv_result_dir, f'{mod}_accuracy_results.csv')
        # 创建保存目录（如果不存在）
        os.makedirs(os.path.dirname(mod_accuracy_csv_path), exist_ok=True) if os.path.dirname(mod_accuracy_csv_path) else None

        mod_accuracy_df.to_csv(mod_accuracy_csv_path, index=False)
        logger.info(f'{mod}准确率数据已保存到: {mod_accuracy_csv_path}')

    #######################################################################


    if cfg.Draw_Confmat is True:
        Draw_Confmat(Confmat_Set, snrs, cfg)
    if cfg.Draw_Acc_Curve is True:
        Snr_Acc_Plot(Accuracy_list, Confmat_Set, snrs, cfg)





