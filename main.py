import argparse
import os

import pytorch_lightning as pl
import soundfile as sf
import torch
import wandb
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from config import CONFIG
from dataset import TrainDataset, TestLoader, BlindTestLoader, NonBlindTestLoader
from models.frn import PLCModel, OnnxWrapper
from utils.utils import mkdir_p

parser = argparse.ArgumentParser()

parser.add_argument('--dirpath', default=None,
                    help='directory to log values')
parser.add_argument('--ckpt', default=None,
                    help='path to .ckpt file to continue training with')
parser.add_argument('--hprm', default=None,
                    help='path to hparams.yaml file to initialize PLCModel')
parser.add_argument('--lr', default=None, type=float,
                    help='if not specified will use value from config')
parser.add_argument('--version', default=None,
                    help='version to resume')
parser.add_argument('--mode', default='train',
                    help='training or testing mode')
parser.add_argument('--compute_metrics', default=None,
                    help='If true and mode test, will use test step with BlindTestLoader')

args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG.gpus)
assert args.mode in ['train', 'eval', 'test', 'onnx', 'nbtest'], "--mode should be 'train', 'eval', 'test' or 'onnx'"


def resume(train_dataset, val_dataset, version, config):
    print("Version", version)
    model_path = os.path.join(CONFIG.LOG.log_dir, 'version_{}/checkpoints/'.format(str(version)))
    # config_path = os.path.join(CONFIG.LOG.log_dir, 'version_{}/'.format(str(version)) + 'hparams.yaml')
    model_name = [x for x in os.listdir(model_path) if x.endswith(".ckpt")][0]
    ckpt_path = model_path + model_name
    checkpoint = PLCModel.load_from_checkpoint(ckpt_path,
                                               strict=True,
                                               # hparams_file=config_path,
                                               config=config)

    return checkpoint, ckpt_path


def train():
    train_dataset = TrainDataset('train')
    val_dataset = TrainDataset('val')
    checkpoint_callback = ModelCheckpoint(dirpath=CONFIG.LOG.log_dir, every_n_epochs=1, save_top_k=3,
                                          monitor=CONFIG.WANDB.monitor, mode='min', verbose=True,
                                          filename='frn-{epoch:02d}-{val_loss:.4f}', save_weights_only=False,
                                          save_last=True, save_on_train_epoch_end=True)
    gpus = [int(i) for i in CONFIG.gpus.split(',')]

    logger = WandbLogger(project=CONFIG.WANDB.project, log_model=False,
                         resume=CONFIG.WANDB.resume_wandb_run, id=CONFIG.WANDB.wandb_run_id)
    if CONFIG.WANDB.sweep:
        wandb.init(project="first_frn_sweep")
        config = wandb.config
    else:
        config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
        wandb.config = config

    if args.version is not None:
        model, ckpt_path = resume(train_dataset, val_dataset, args.version, config)
    elif args.ckpt is not None:
        ckpt_path = args.ckpt
        model = PLCModel.load_from_checkpoint(args.ckpt,
                                              strict=True,
                                              # hparams_file=args.hprm,
                                              config=config)
    else:
        ckpt_path = None
        model = PLCModel(config=config)
    logger.watch(model, log_graph=False)
    trainer = pl.Trainer(logger=logger,
                         log_every_n_steps=2,
                         gradient_clip_val=CONFIG.TRAIN.clipping_val,
                         devices=gpus,
                         max_epochs=CONFIG.TRAIN.epochs,
                         accelerator="gpu" if len(gpus) >= 1 else None,
                         callbacks=[checkpoint_callback],
                         limit_val_batches=CONFIG.TRAIN.limit_val_batches,
                         check_val_every_n_epoch=CONFIG.TRAIN.check_val_every_n_epoch,
                         num_sanity_val_steps=CONFIG.TRAIN.limit_sainty_steps)
    print(model.hparams)
    print(
        'Dataset: {}, Train files: {}, Val files {}'.format(CONFIG.DATA.dataset, len(train_dataset), len(val_dataset)))
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=CONFIG.TRAIN.batch_size,
                              num_workers=CONFIG.TRAIN.workers, persistent_workers=True)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=CONFIG.TRAIN.batch_size,
                            num_workers=CONFIG.TRAIN.workers, persistent_workers=True)
    torch.set_float32_matmul_precision("high")
    if ckpt_path is None:
        trainer.fit(model, train_loader, val_loader)
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)


def to_onnx(model, onnx_path):
    model.eval()

    model = OnnxWrapper(model)

    torch.onnx.export(model,
                      model.sample,
                      onnx_path,
                      export_params=True,
                      opset_version=12,
                      input_names=model.input_names,
                      output_names=model.output_names,
                      do_constant_folding=True,
                      verbose=False)


if __name__ == '__main__':
    if args.mode == 'train':
        torch.set_float32_matmul_precision("high")
        if CONFIG.WANDB.sweep:
            sweep_config = yaml.load(open("sweep_config.yaml", "r"), Loader=yaml.FullLoader)
            sweep_id = wandb.sweep(sweep_config, project="first_frn_sweep")
            if CONFIG.WANDB.sweep_id is None:
                wandb.agent(sweep_id=sweep_id, function=train)
            else:
                wandb.agent(sweep_id=CONFIG.WANDB.sweep_id, function=train)
        else:
            train()
    else:
        config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
        ckpt_path = CONFIG.NBTEST.ckpt_path
        model = PLCModel.load_from_checkpoint(ckpt_path, strict=True, config=config)
        model.eval()
        model.freeze()
        if args.mode == 'eval':
            model.cuda(device=0)
            trainer = pl.Trainer(accelerator='gpu', devices=1, enable_checkpointing=False, logger=False)
            testset = TestLoader()
            test_loader = DataLoader(testset, batch_size=1, num_workers=4)
            trainer.test(model, test_loader)
            print('Version', args.version)
            masking = CONFIG.DATA.EVAL.masking
            prob = CONFIG.DATA.EVAL.transition_probs[0]
            loss_percent = (1 - prob[0]) / (2 - prob[0] - prob[1]) * 100
            print('Evaluate with real trace' if masking == 'real' else
                  'Evaluate with generated trace with {:.2f}% packet loss'.format(loss_percent))
        elif args.mode == 'test':
            model.cuda(device=0)
            testset = BlindTestLoader(test_dir=CONFIG.TEST.in_dir)
            test_loader = DataLoader(testset, batch_size=1, num_workers=4)
            trainer = pl.Trainer(accelerator='gpu', devices=1, enable_checkpointing=False, logger=False)
            if args.compute_metrics:
                trainer.test(model, test_loader)
            else:
                preds = trainer.predict(model, test_loader, return_predictions=True)
                mkdir_p(CONFIG.TEST.out_dir)
                mkdir_p(CONFIG.TEST.out_dir)
                mkdir_p(CONFIG.TEST.out_dir_orig)
                for idx, path in enumerate(test_loader.dataset.data_list):
                    out_path = os.path.join(CONFIG.TEST.out_dir, os.path.basename(path))
                    sf.write(out_path, preds[idx][0].squeeze(0), samplerate=CONFIG.DATA.sr, subtype='PCM_16')
                    out_orig_path = os.path.join(CONFIG.TEST.out_dir_orig, os.path.basename(path))
                    sf.write(out_orig_path, preds[idx][1].squeeze(0), samplerate=CONFIG.DATA.sr, subtype='PCM_16')
        elif args.mode == 'nbtest':
            with torch.no_grad():
                mkdir_p(CONFIG.NBTEST.out_dir16)
                mkdir_p(CONFIG.NBTEST.out_dir48)
                mkdir_p(CONFIG.NBTEST.out_dir_orig)
                model.cuda()
                model.eval()
                testset = NonBlindTestLoader(size=CONFIG.NBTEST.to_synthesize)
                test_loader = DataLoader(testset, batch_size=1, num_workers=1)
                trainer = pl.Trainer(accelerator='gpu', devices=1)
                result = trainer.test(model, test_loader, ckpt_path=ckpt_path)
                print(result)
        else:
            onnx_path = 'lightning_logs/version_{}/checkpoints/frn.onnx'.format(str(args.version))
            to_onnx(model, onnx_path)
            print('ONNX model saved to', onnx_path)
