import os.path
import time
import pandas as pd
import torch
from torch import optim, nn
from tqdm import tqdm
from utils import AverageMeter, EarlyStopping



class Trainer:
    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 cfg,
                 logger):
        super(Trainer, self).__init__()

        self.epochs_stats = None
        self.val_acc_list = None
        self.val_loss_list = None
        self.train_acc_list = None
        self.train_loss_list = None
        self.val_acc = None
        self.val_loss = None
        self.train_acc = None
        self.best_monitor = None
        self.lr_list = None
        self.train_loss = None
        self.t_s = None
        self.early_stopping = None
        self.criterion = None
        self.optimizer = None

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.logger = logger

        self.iter = 0

        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.previous_lr = None  # 用于保存上一次的学习率
        self.new_dropout_rate = 0.0  # 初始 dropout 比例


    def loop(self):
        self.before_train()
        for self.iter in range(0, self.cfg.epochs):
            self.before_train_step()
            self.run_train_step()
            self.after_train_step()
            self.before_val_step()
            self.run_val_step()
            self.after_val_step()
            if self.early_stopping.early_stop:
                self.logger.info('Early stopping')
                break

    @staticmethod
    def adjust_learning_rate(optimizer, gamma):
        """Sets the learning rate when we have to"""
        lr = optimizer.param_groups[0]['lr'] * gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def before_train(self):
        # 自动跟踪模型的参数和梯度
        # Automatically track model parameters and gradients
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.criterion = nn.CrossEntropyLoss().to(self.cfg.device)
        self.early_stopping = EarlyStopping(self.logger, patience=self.cfg.patience)

        self.lr_list = []
        self.best_monitor = 0.0
        self.train_loss_list = []
        self.train_acc_list = []
        self.val_loss_list = []
        self.val_acc_list = []



    def adjust_dropout(self):
        # 获取当前学习率
        # Get the current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']

        # 如果之前的学习率不存在，初始化为当前学习率
        # If the previous learning rate does not exist, initialize it to the current learning rate
        if self.previous_lr is None:
            self.previous_lr = current_lr

        # 只有当当前学习率小于上一次的学习率时才增加 dropout
        # Only increase dropout when the current learning rate is lower than the previous learning rate
        if current_lr < self.previous_lr:
            # 计算新的 dropout 比例
            self.new_dropout_rate = min(self.cfg.max_dropout_rate, max(self.new_dropout_rate + 0.1,self.cfg.initial_dropout_rate))
            self.set_dropout_rate(self.new_dropout_rate)
            self.logger.info(f"Learning rate decreased, increasing dropout rate to {self.new_dropout_rate:.4f}")

        # 更新之前的学习率
        # The learning rate before the update
        self.previous_lr = current_lr

    def set_dropout_rate(self, new_rate):
        # 设置模型中所有 Dropout 层的 p 值
        # Set the p value for all Dropout layers in the model
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = new_rate
                self.logger.info(f"Set dropout rate to {new_rate:.4f}")

    def before_train_step(self):
        # 在每个训练步骤开始前，调整 dropout rate
        # Adjust the dropout rate before each training step
        self.adjust_dropout()

        # 训练步骤的其他内容
        # Other content of the training steps
        self.model.train()
        self.t_s = time.time()
        self.train_loss = AverageMeter()
        self.train_acc = AverageMeter()
        self.logger.info(f"Starting training epoch {self.iter}:")

    def run_train_step(self):
        with tqdm(total=len(self.train_loader),
                  desc=f'Epoch{self.iter}/{self.cfg.epochs}',
                  postfix=dict,
                  mininterval=0.3) as pbar:
            for step, (sig_batch, lab_batch) in enumerate(self.train_loader):
                sig_batch = sig_batch.to(self.cfg.device)
                lab_batch = lab_batch.to(self.cfg.device)

                logit, regu_sum = self.model(sig_batch)

                loss = self.criterion(logit, lab_batch)
                loss += sum(regu_sum)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pre_lab = torch.argmax(logit, 1)
                acc = torch.sum(pre_lab == lab_batch.data).double().item() / lab_batch.size(0)

                self.train_loss.update(loss.item())
                self.train_acc.update(acc)

                pbar.set_postfix(**{'train_loss': self.train_loss.avg,
                                    'train_acc': self.train_acc.avg})
                pbar.update(1)

    def after_train_step(self):
        self.lr_list.append(self.optimizer.param_groups[0]['lr'])
        self.logger.info(
            '====> Epoch: {} Time: {:.2f} Train Loss: {} Train acc: {} lr: {:.5f}'.format(self.iter,
                                                                                          time.time() - self.t_s,
                                                                                          self.train_loss.avg,
                                                                                          self.train_acc.avg,
                                                                                          self.lr_list[-1]))
        self.train_loss_list.append(self.train_loss.avg)
        self.train_acc_list.append(self.train_acc.avg)

    def before_val_step(self):
        self.model.eval()
        self.t_s = time.time()
        self.val_loss = AverageMeter()
        self.val_acc = AverageMeter()
        self.logger.info(f"Starting validation epoch {self.iter}:")

    def run_val_step(self):
        with tqdm(total=len(self.val_loader),
                  desc=f'Epoch{self.iter}/{self.cfg.epochs}',
                  postfix=dict,
                  mininterval=0.3,
                  colour='blue') as pbar:
            for step, (sig_batch, lab_batch) in enumerate(self.val_loader):
                with torch.no_grad():
                    sig_batch = sig_batch.to(self.cfg.device)
                    lab_batch = lab_batch.to(self.cfg.device)

                    logit, regu_sum = self.model(sig_batch)

                    loss = self.criterion(logit, lab_batch)
                    loss += sum(regu_sum)

                    pre_lab = torch.argmax(logit, 1)
                    acc = torch.sum(pre_lab == lab_batch.data).double().item() / lab_batch.size(0)

                    self.val_loss.update(loss.item())
                    self.val_acc.update(acc)

                    pbar.set_postfix(**{'val_loss': self.val_loss.avg,
                                        'val_acc': self.val_acc.avg})
                    pbar.update(1)

    def after_val_step(self):
        # 记录当前 epoch 的验证损失和准确率
        # Record the validation loss and accuracy of the current epoch
        self.logger.info(
            '====> Epoch: {} Time: {:.2f} Val Loss: {} Val acc: {}'.format(
                self.iter, time.time() - self.t_s, self.val_loss.avg, self.val_acc.avg))

        # 验证集损失和准确率变化管理
        improvement = False

        # 保存最佳模型基于验证集的准确率
        # Save the best model based on validation set accuracy
        if self.cfg.monitor == 'acc':
            if self.val_acc.avg > self.best_val_acc:
                self.last_best_val_acc = self.best_val_acc
                self.best_val_acc = self.val_acc.avg
                save_model_name_acc = f"{self.cfg.dataset}_{self.cfg.model}.pkl"
                torch.save(self.model.state_dict(), os.path.join(self.cfg.model_dir, save_model_name_acc))
                self.logger.info(f"New best accuracy achieved: {self.best_val_acc:.4f}. Model saved (accuracy).")
                improvement = True

            if self.val_loss.avg < self.best_val_loss:
                self.last_best_val_loss = self.val_loss.avg
                self.best_val_loss = self.val_loss.avg
                self.logger.info(f"New lowest validation loss achieved: {self.best_val_loss:.4f}.")
                improvement = True

        # 如果有改进，重置早停计数器为 0
        # If there is improvement, reset the early stopping counter to 0
        if improvement:
            self.early_stopping.counter = 0
        else:
            # 如果没有改进，执行早停逻辑，并可能调整学习率
            # If there is no improvement, execute early stopping logic and potentially adjust the learning rate
            self.early_stopping(self.val_loss.avg, self.model)

            # 根据早停计数器调整学习率
            # Adjust learning rate based on early stopping counter
            if self.early_stopping.counter != 0 and self.early_stopping.counter % self.cfg.milestone_step == 0:
                self.adjust_learning_rate(self.optimizer, self.cfg.gamma)
                self.logger.info(f"Adjusted learning rate to {self.optimizer.param_groups[0]['lr']}")

        # # Adjust the learning rate when the difference between training set accuracy and validation set accuracy exceeds a threshold
        acc_diff = self.train_acc.avg - self.val_acc.avg  # 计算训练集和验证集的准确率差异 Calculate the accuracy difference between the training set and the validation set

        # 如果差异超过阈值，增加忍耐轮数计数器 If the difference exceeds the threshold, increase the patience round counter
        if acc_diff > self.cfg.acc_threshold:
            if not hasattr(self, 'acc_diff_counter'):
                self.acc_diff_counter = 0  # 初始化计数器
            if not hasattr(self, 'first_acc_diff_triggered'):  # 标记首次触发调整学习率
                self.first_acc_diff_triggered = False

            if not self.first_acc_diff_triggered:
                # 第一次差异超过阈值时立即调整学习率
                # Adjust the learning rate immediately when the first difference exceeds the threshold
                self.adjust_learning_rate(self.optimizer, self.cfg.gamma)
                self.logger.info(f"First time accuracy difference exceeded threshold, adjusted learning rate.")
                self.first_acc_diff_triggered = True  # 标记为已首次触发 Marked as triggered for the first time

            else:
                # 后续的差异仍大于阈值时，增加计数器
                # When the subsequent difference is still greater than the threshold, increment the counter
                self.acc_diff_counter += 1
                self.logger.info(
                    f"Accuracy difference is large: Train acc: {self.train_acc.avg:.4f}, Val acc: {self.val_acc.avg:.4f}, Counter: {self.acc_diff_counter}")

        else:
            # 如果差异小于阈值，重置计数器
            # If the difference is less than the threshold, reset the counter
            self.acc_diff_counter = 0
            self.first_acc_diff_triggered = False

        # 重置计数器条件：如果验证集准确率上升或者损失有变化
        # Reset counter condition: if validation accuracy increases or loss changes
        if self.val_acc.avg > self.last_best_val_acc or self.val_loss.avg < self.last_best_val_loss:
            # 如果有改善，重置计数器
            self.acc_diff_counter = 0
            self.first_acc_diff_triggered = False  # 重置首次触发标志

        # 如果已经经过了忍耐轮数，进行学习率调整
        # If the number of patience rounds has been exceeded, adjust the learning rate
        if self.acc_diff_counter >= self.cfg.patience_acc_diff:
            self.adjust_learning_rate(self.optimizer, self.cfg.gamma)
            self.logger.info(
                f"Accuracy difference exceeded threshold for {self.cfg.patience_acc_diff} epochs. Adjusting learning rate.")
            self.acc_diff_counter = 0  # 重置计数器

        # 记录当前验证集的损失和准确率
        # Record the loss and accuracy of the current validation set
        self.val_loss_list.append(self.val_loss.avg)
        self.val_acc_list.append(self.val_acc.avg)



        # 记录所有轮次的统计数据
        # Record Statistics for All Rounds
        self.epochs_stats = pd.DataFrame(
            data={"epoch": range(self.iter + 1),
                  "lr_list": self.lr_list,
                  "train_loss": self.train_loss_list,
                  "val_loss": self.val_loss_list,
                  "train_acc": self.train_acc_list,
                  "val_acc": self.val_acc_list}
        )

