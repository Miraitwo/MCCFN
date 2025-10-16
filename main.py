import os.path
import argparse
from utils import *
from data_loader import Load_Dataset, Dataset_Split, Create_Data_Loader
from visualize import *
from evaluation import Run_Eval
from training import Trainer


def parse_args(models_name,dataset_name):
    parser = argparse.ArgumentParser(description="训练调制识别模型.")
    parser.add_argument("--model", type=str, default=models_name, help="要训练的模型")
    parser.add_argument("--epochs", type=int, default=100, help="训练的轮数")
    parser.add_argument("--batch_size", type=int, default=128, help="每个批次的样本数量（设置较低以减少CUDA内存使用）")


    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument('--dataset', type=str, default=dataset_name)  # 2016.10a, 2016.10b, rml22, 2016.04c
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--device', type=str,
                        default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--Draw_Confmat', type=bool, default=True)
    parser.add_argument('--Draw_Acc_Curve', type=bool, default=True)

    return parser.parse_args()

def main(models_name, dataset_name):
    args = parse_args(models_name, dataset_name)
    fix_seed(args.seed)
    cfg = Config(args.dataset, args.model, train=(args.mode == 'train'))
    cfg = merge_args2cfg(cfg, vars(args))
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))
    log_exp_settings(logger, cfg)

    logger.info("loading %s", cfg.model)
    model = load_model(cfg)

    Signals, Labels, SNRs, snrs, mods = Load_Dataset(cfg.dataset, logger)
    train_set, test_set, val_set, test_idx = Dataset_Split(
        Signals=Signals,
        Labels=Labels,
        SNRs=SNRs,
        snrs=snrs,
        mods=mods,
        logger=logger,
        val_size=cfg.val_size,
        test_size=cfg.test_size
    )
    Signals_test, Labels_test = test_set

    if args.mode == 'train':
        train_loader, val_loader = Create_Data_Loader(train_set, val_set, cfg, logger)
        trainer = Trainer(model,
                          train_loader,
                          val_loader,
                          cfg,
                          logger)
        trainer.loop()

        save_training_process(trainer.epochs_stats, cfg)

        save_model_name = cfg.dataset + '_' + cfg.model + '.pkl'
        model.load_state_dict(torch.load(os.path.join(cfg.model_dir, save_model_name)))
        Run_Eval(model,
                 Signals_test,
                 Labels_test,
                 SNRs,
                 test_idx,
                 cfg,
                 logger)

    elif args.mode == 'eval':
        model.load_state_dict(torch.load(os.path.join(cfg.model_dir, save_model_name)))
        Run_Eval(model,
                 Signals_test,
                 Labels_test,
                 SNRs,
                 test_idx,
                 cfg,
                 logger)

if __name__ == '__main__':




    models_name_list = ['MCCFN']
    dataser_name_list = ['2016.10a']
    for dataset_name in dataser_name_list:
        for model_name in models_name_list:
            main(model_name, dataset_name)


