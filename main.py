import argparse
import os

import pytorch_lightning as pl
import soundfile as sf
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.model_summary import summarize
from torch.utils.data import DataLoader

from config import CONFIG
from dataset import TrainDataset, TestLoader, BlindTestLoader, NonBlindTestLoader
from torch.utils.data import DataLoader
from models.frn import PLCModel, OnnxWrapper
from pytorch_lightning.loggers import WandbLogger
from utils.utils import mkdir_p

parser = argparse.ArgumentParser()

parser.add_argument('--dirpath', default=None,
                    help='directory to log values')
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


def resume(train_dataset, val_dataset, version):
    print("Version", version)
    model_path = os.path.join(CONFIG.LOG.log_dir, 'version_{}/checkpoints/'.format(str(version)))
    config_path = os.path.join(CONFIG.LOG.log_dir, 'version_{}/'.format(str(version)) + 'hparams.yaml')
    model_name = [x for x in os.listdir(model_path) if x.endswith(".ckpt")][0]
    ckpt_path = model_path + model_name
    checkpoint = PLCModel.load_from_checkpoint(ckpt_path,
                                               strict=True,
                                               hparams_file=config_path,
                                               train_dataset=train_dataset,
                                               val_dataset=val_dataset,
                                               window_size=CONFIG.DATA.window_size)

    return checkpoint


def train():
    train_dataset = TrainDataset('train')
    val_dataset = TrainDataset('val')
    args.dirpath = args.dirpath if args.dirpath is not None else CONFIG.LOG.log_dir
    checkpoint_callback = ModelCheckpoint(dirpath=args.dirpath,
                                          monitor=CONFIG.WANDB.monitor, mode='min', verbose=True,
                                          filename='frn-{epoch:02d}-{val_loss:.4f}', save_weights_only=False)
    gpus = CONFIG.gpus.split(',')
    logger = WandbLogger(project=CONFIG.WANDB.project, log_model=False) # TO DO REFACTOR CONFIG
    if args.version is not None:
        model = resume(train_dataset, val_dataset, args.version)
    else:
        model = PLCModel(train_dataset,
                         val_dataset,
                         window_size=CONFIG.DATA.window_size,
                         enc_layers=CONFIG.MODEL.enc_layers,
                         enc_in_dim=CONFIG.MODEL.enc_in_dim,
                         enc_dim=CONFIG.MODEL.enc_dim,
                         pred_dim=CONFIG.MODEL.pred_dim,
                         pred_layers=CONFIG.MODEL.pred_layers,
                         lr=args.lr)
    logger.watch(model, log_graph=False)
    trainer = pl.Trainer(logger=logger,
                         gradient_clip_val=CONFIG.TRAIN.clipping_val,
                         gpus=len(gpus),
                         max_epochs=CONFIG.TRAIN.epochs,
                         accelerator="gpu" if len(gpus) > 1 else None,
                         callbacks=[checkpoint_callback],
                         limit_val_batches=CONFIG.TRAIN.limit_val_batches,
                         check_val_every_n_epoch=CONFIG.TRAIN.check_val_every_n_epoch)
    print(model.hparams)
    print(
        'Dataset: {}, Train files: {}, Val files {}'.format(CONFIG.DATA.dataset, len(train_dataset), len(val_dataset)))
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=CONFIG.TRAIN.batch_size,
                              num_workers=CONFIG.TRAIN.workers, persistent_workers=True)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=CONFIG.TRAIN.batch_size,
                            num_workers=CONFIG.TRAIN.workers, persistent_workers=True)
    trainer.fit(model, train_loader, val_loader)


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
    torch.set_float32_matmul_precision("high")

    if args.mode == 'train':
        train()
    else:
        model = resume(None, None, args.version)
        print(model.hparams)
        print(summarize(model))

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
                mkdir_p(CONFIG.TEST.out_dir_orig)
                for idx, path in enumerate(test_loader.dataset.data_list):
                    out_path = os.path.join(CONFIG.TEST.out_dir, os.path.basename(path))
                    sf.write(out_path, preds[idx][0].squeeze(0), samplerate=CONFIG.DATA.sr, subtype='PCM_16')
                    out_orig_path = os.path.join(CONFIG.TEST.out_dir_orig, os.path.basename(path))
                    sf.write(out_orig_path, preds[idx][1].squeeze(0), samplerate=CONFIG.DATA.sr, subtype='PCM_16')
        elif args.mode == 'nbtest':
            with torch.no_grad():
                mkdir_p(CONFIG.NBTEST.out_dir)
                mkdir_p(CONFIG.NBTEST.out_dir_orig)
                model.cuda(device=0)
                model.eval()
                testset = NonBlindTestLoader()
                test_loader = DataLoader(testset, batch_size=1, num_workers=4)
                trainer = pl.Trainer(accelerator='gpu', devices=1, enable_checkpointing=False, logger=False)
                data_lists = [i for i in test_loader.dataset.data_list]
                result = trainer.predict(model, test_loader, return_predictions=True)
                import pdb
                pdb.set_trace()
                idx = 0
                for j, in range(len(data_lists)):
                    if j > 3:
                        break
                    if j % 2 != 0:
                        idx += 1
                    pred = result[idx]
                    inp_wav = result[idx + 1]
                    # save files
                    out_path = os.path.join(CONFIG.NBTEST.out_dir, os.path.basename(data_lists[j]))
                    sf.write(out_path, pred.squeeze(0).cpu().numpy(), samplerate=CONFIG.DATA.sr, subtype='PCM_16')
                    out_orig_path = os.path.join(CONFIG.NBTEST.out_dir_orig, os.path.basename(data_lists[j]))
                    sf.write(out_orig_path, inp_wav.squeeze(0).cpu().numpy(), samplerate=CONFIG.DATA.sr, subtype='PCM_16')
        else:
            onnx_path = 'lightning_logs/version_{}/checkpoints/frn.onnx'.format(str(args.version))
            to_onnx(model, onnx_path)
            print('ONNX model saved to', onnx_path)
