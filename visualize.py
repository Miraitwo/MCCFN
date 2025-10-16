from functools import reduce
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os


def Draw_Confmat(Confmat_Set, snrs, cfg):
    # 检查 cfg.classes 中的键是否已经是字符串类型，如果不是则进行解码
    if not isinstance(list(cfg.classes.keys())[0], str):
        cfg.classes = {mod.decode('utf-8'): label for mod, label in cfg.classes.items()}

    for i, snr in enumerate(snrs):
        # 创建一个图形对象，并设置画布大小
        fig = plt.figure(figsize=(10, 8))  # 调整宽度和高度，根据需要调整

        # 创建DataFrame
        df_cm = pd.DataFrame(Confmat_Set[i],
                             index=cfg.classes,
                             columns=cfg.classes)

        # 绘制热力图
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")

        # 设置y轴和x轴标签的显示
        heatmap.yaxis.set_ticklabels(
            heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
        heatmap.xaxis.set_ticklabels(
            heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')

        # 设置坐标轴标签
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        # 自动调整布局，避免标签被截断
        plt.tight_layout()  # 自动调整布局，防止标签被截断

        # 保存图像
        conf_mat_dir = os.path.join(cfg.result_dir, 'conf_mat')
        os.makedirs(conf_mat_dir, exist_ok=True)
        fig.savefig(conf_mat_dir + '/' + f'ConfMat_{snr}dB.svg', format='svg', dpi=150)

        # 关闭图形
        plt.close()



def Snr_Acc_Plot(Accuracy_list, Confmat_Set, snrs, cfg):
    # 绘制总体准确率
    plt.plot(snrs, Accuracy_list)
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Overall Accuracy")
    plt.title(f"Overall Accuracy on {cfg.dataset} dataset")
    plt.yticks(np.linspace(0, 1, 11))
    plt.grid()

    acc_dir = os.path.join(cfg.result_dir, 'acc')
    os.makedirs(acc_dir, exist_ok=True)
    plt.savefig(acc_dir + '/' + 'acc.svg', format='svg', dpi=150)
    plt.close()

    # 计算每种调制类型的准确率
    Accuracy_Mods = np.zeros((len(snrs), Confmat_Set.shape[-1]))
    for i, snr in enumerate(snrs):
        Accuracy_Mods[i, :] = np.diagonal(Confmat_Set[i]) / Confmat_Set[i].sum(1)

    # 绘制每种调制信号的准确率
    plt.figure(figsize=(10, 8))  # 调整画布大小，增加空间
    mod_labels = [mod.decode('utf-8') if isinstance(mod, bytes) else mod for mod in cfg.classes.keys()]  # 解码字节串为字符串
    for j, mod_label in enumerate(mod_labels):
        plt.plot(snrs, Accuracy_Mods[:, j], label=mod_label)

    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Modulation Accuracy")
    plt.title(f"Accuracy by Modulation on {cfg.dataset} dataset")
    plt.grid()

    # 设置图例的位置和防止遮挡
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)  # 右侧显示图例，避免遮挡曲线

    # 保存图像
    plt.savefig(acc_dir + '/' + 'acc_mods.svg', format='svg', dpi=150, bbox_inches='tight')  # 使用 bbox_inches='tight' 确保图像不被裁切
    plt.close()


def save_training_process(train_process, cfg):
    fig1 = plt.figure(figsize=(8, 6))  # Increased figure size for better visibility
    plt.plot(train_process.epoch, train_process.lr_list)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Learning Rate", fontsize=16)  # Increased font size for better readability
    #plt.title("Learning Rate", fontsize=16)  # Increased font size for better readability
    plt.grid()
    plt.xticks(fontsize=14)  # Set font size for x-axis tick labels
    plt.yticks(fontsize=14)  # Set font size for y-axis tick labels
    plt.tight_layout()  # Adjust layout to fit everything properly
    fig1.savefig(cfg.result_dir + '/' + 'lr.svg', format='svg', dpi=150, bbox_inches='tight')  # Ensure the entire plot is saved
    plt.close()

    fig2 = plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process.epoch, train_process.train_loss, "ro-", label="Train loss")
    plt.plot(train_process.epoch, train_process.val_loss, "bs-", label="Val loss")
    plt.legend()
    plt.grid()
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Loss", fontsize=16)  # Increased font size for better readability
    plt.xticks(fontsize=14)  # Set font size for x-axis tick labels
    plt.yticks(fontsize=14)  # Set font size for y-axis tick labels
    plt.subplot(1, 2, 2)
    plt.plot(train_process.epoch, train_process.train_acc, "ro-", label="Train acc")
    plt.plot(train_process.epoch, train_process.val_acc, "bs-", label="Val acc")
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)  # Increased font size for better readability
    plt.legend()
    plt.grid()
    plt.xticks(fontsize=14)  # Set font size for x-axis tick labels
    plt.yticks(fontsize=14)  # Set font size for y-axis tick labels
    plt.tight_layout()  # Adjust layout to fit everything properly
    fig2.savefig(cfg.result_dir + '/' + 'loss_acc.svg', format='svg', dpi=150, bbox_inches='tight')  # Ensure the entire plot is saved
    plt.show()
    plt.close()


