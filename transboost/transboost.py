import math
import pathlib
from argparse import ArgumentParser

import torch
from torch.nn import functional as F
from torchvision.models import resnet as pytorch_resnet

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from torchmetrics import Accuracy

from utils import DatasetWithIndex, TransductiveDatasetTest
from utils import SyncFunction

from data import all_data_modules


class PseudoLabelSelector(pl.LightningModule):
    def __init__(self, hparams, model):
        super(PseudoLabelSelector, self).__init__()
        self.save_hyperparameters(hparams)
        self.model = model
        self.accuracy = Accuracy()

        self.idx = []
        self.confidence = []
        self.pseudo_labels = []

    def test_dataloader(self):
        dataset_with_idx = DatasetWithIndex(data.test_dataloader().dataset)
        return torch.utils.data.DataLoader(
            dataset_with_idx,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers
        )

    def test_step(self, batch, batch_id):
        idx, images, labels = batch
        preds = self.model(images)
        probs = F.softmax(preds, dim=1)
        confidence = probs.max(dim=1)[0]
        pseudo_labels = probs.argmax(dim=1)
        self.idx.append(idx)
        self.confidence.append(confidence)
        self.pseudo_labels.append(pseudo_labels)

    def test_epoch_end(self, outputs):
        confidence = torch.zeros(self.hparams.num_test_samples, dtype=torch.float, device='cpu')
        pseudo_labels = torch.zeros(self.hparams.num_test_samples, dtype=torch.long, device='cpu')

        if self.hparams.gpus > 1:
            self.idx = torch.cat(self.all_gather(self.idx), dim=1)
            self.confidence = torch.cat(self.all_gather(self.confidence), dim=1).cpu()
            self.pseudo_labels = torch.cat(self.all_gather(self.pseudo_labels), dim=1).cpu()
            for i, c, p in zip(self.idx, self.confidence, self.pseudo_labels):
                confidence[i] = c
                pseudo_labels[i] = p
        else:
            self.idx = torch.cat(self.idx)
            confidence[self.idx] = torch.cat(
                self.confidence).cpu()
            pseudo_labels[self.idx] = torch.cat(self.pseudo_labels).cpu()

        del self.idx
        self.confidence = confidence
        self.pseudo_labels = pseudo_labels


class PretrainAccuracyChecker(pl.LightningModule):
    def __init__(self, hparams, model):
        super(PretrainAccuracyChecker, self).__init__()
        self.save_hyperparameters(hparams)
        self.model = model
        self.accuracy = Accuracy()

    def test_dataloader(self):
        return data.test_dataloader()

    def test_step(self, batch, batch_id, dataloader_idx=0):
        images, labels = batch
        preds = self.model(images)
        probs = F.softmax(preds, dim=1)
        acc = self.accuracy(probs, labels)
        self.log('acc/pretrain', acc)


class TransBoost(pl.LightningModule):
    def __init__(self, hparams):
        super(TransBoost, self).__init__()
        self.save_hyperparameters(hparams)
        self.accuracy = Accuracy()
        self.csv_logger = None

        assert self.hparams.model in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        resnet_class = getattr(pytorch_resnet, self.hparams.model)
        self.model = resnet_class(num_classes=self.hparams.num_classes, pretrained=True)

        self.pseudo_labels = None
        self.confidence = None

    def train_dataloader(self):
        test_dataset_with_pseudo = TransductiveDatasetTest(data.test_dataset(data.train_transforms),
                                                           self.pseudo_labels, self.confidence)
        dl_test = torch.utils.data.DataLoader(
            test_dataset_with_pseudo,
            batch_size=self.hparams.batch_size,
            shuffle=False if self.hparams.dev else True,
            num_workers=self.hparams.num_workers
        )
        return [data.train_dataloader(), dl_test]

    def val_dataloader(self):
        return data.test_dataloader()

    def test_dataloader(self):
        return data.test_dataloader()

    def configure_optimizers(self):
        params = {
            'params': self.model.parameters(),
            'lr': self.hparams.learning_rate,
            'weight_decay': self.hparams.weight_decay
        }

        if self.hparams.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                **params,
                momentum=0.9,
                nesterov=True
            )
        elif self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                **params
            )
        else:
            raise f'Unrecognized optimizer - {self.hparams.optimizer}'

        dataloaders = self.train_dataloader()
        steps_per_epoch = max(len(dataloaders[0]), len(dataloaders[1])) // self.hparams.gpus
        total_steps = self.hparams.max_epochs * steps_per_epoch

        warmup_epochs = 0
        s = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            linear_warmup_decay(steps_per_epoch * warmup_epochs, total_steps, cosine=self.hparams.cosine)
        )

        scheduler = {
            "scheduler": s,
            "interval": "step",
            "name": "learning_rate"
        }
        return [optimizer], [scheduler]

    def transductive_loss(self, softmaxes, labels, confidence):
        random_indices = torch.randperm(softmaxes.shape[0])
        delta = (labels != labels[random_indices]).float()
        pairwise_l2_norm = torch.linalg.vector_norm(softmaxes - softmaxes[random_indices], ord=2, dim=1)
        pairwise_similarities = math.sqrt(2) - pairwise_l2_norm
        pairwise_confidence = confidence * confidence[random_indices]
        loss = pairwise_confidence * delta * pairwise_similarities
        return loss.mean()

    def training_step(self, batch, batch_id):
        train_batch, test_batch = batch
        images_train, labels_train = train_batch
        images_test, pseudo_labels_test, test_confidence = test_batch

        total_images = torch.cat((images_train, images_test))
        total_preds = self.model(total_images)
        preds_train = total_preds[:images_train.shape[0]]
        preds_test = total_preds[images_train.shape[0]:]

        loss_train = F.cross_entropy(preds_train, labels_train)
        acc_train = self.accuracy(preds_train.argmax(dim=1), labels_train)
        self.log('loss/train', loss_train)
        self.log('acc/train', acc_train)

        probs_test = F.softmax(preds_test, dim=1)
        if self.hparams.gpus > 1:
            probs_test = SyncFunction.apply(probs_test)
            pseudo_labels_test = SyncFunction.apply(pseudo_labels_test)
            test_confidence = SyncFunction.apply(test_confidence)
        loss_transductive = self.transductive_loss(probs_test, pseudo_labels_test, test_confidence)
        self.log('loss/transductive', loss_transductive)

        total_loss = loss_train + self.hparams.lamda * loss_transductive
        return total_loss

    def validation_step(self, batch, batch_id, dataloader_idx=0):
        images, labels = batch
        preds = self.model(images)
        loss = F.cross_entropy(preds, labels)
        acc = self.accuracy(preds.argmax(dim=1), labels)
        self.log('loss/val', loss)
        self.log('acc/val', acc)

    def validation_epoch_end(self, outputs):
        if self.local_rank == 0 and type(self.logger) is WandbLogger and not self.trainer.sanity_checking:
            if self.csv_logger is None:
                self.csv_logger = CSVLogger(save_dir=self.logger.experiment.project,
                                            name=self.logger.experiment.id,
                                            version='logs')
                self.csv_logger.log_hyperparams(self.hparams)
            self.csv_logger.log_metrics(self.trainer.logged_metrics, step=self.global_step)
            self.csv_logger.save()

    def test_step(self, batch, batch_id, dataloader_idx=0):
        images, labels = batch
        preds = self.model(images)
        probs = F.softmax(preds, dim=1)
        loss = F.cross_entropy(preds, labels)
        acc = self.accuracy(probs, labels)
        self.log('loss/test', loss)
        self.log('acc/test', acc)


def run(args):
    model = TransBoost(args)

    log_name = f'{args.data}_{args.model}_seed{args.seed}'
    log_name += f'_lamda{args.lamda}_batchsize{args.batch_size}_epochs{args.max_epochs}'
    logger = WandbLogger(name=log_name, project='TransBoost') if args.wandb else False

    checkpoint_callback = pl.callbacks.ModelCheckpoint(filename='best', monitor='acc/val', mode='max', save_last=True)
    learning_rate_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')
    gpus_stats_callback = pl.callbacks.GPUStatsMonitor()
    callbacks = [checkpoint_callback]
    if args.wandb:
        callbacks.append(learning_rate_callback)
        if args.gpu_monitor:
            callbacks.append(gpus_stats_callback)

    main_trainer = pl.Trainer(gpus=args.gpus,
                              accelerator='ddp' if args.gpus > 1 else None,
                              logger=logger,
                              deterministic=True,
                              max_epochs=args.max_epochs,
                              fast_dev_run=args.dev,
                              callbacks=callbacks,
                              resume_from_checkpoint=args.resume)

    path = pathlib.Path(f'cache/{args.data}_{args.model}/')
    path.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        checker = PretrainAccuracyChecker(args, model.model)
        main_trainer.test(checker)
        selector = PseudoLabelSelector(args, model.model)
        main_trainer.test(selector)
        torch.save(selector.pseudo_labels, path / 'pseudo_labels.pt')
        torch.save(selector.confidence, path / 'confidence.pt')
        model.pseudo_labels = selector.pseudo_labels
        model.confidence = selector.confidence
    else:
        try:
            model.pseudo_labels = torch.load(path / 'pseudo_labels.pt')
            model.confidence = torch.load(path / 'confidence.pt')
        except:
            raise 'Cant resume the procedure'

    if not args.test_only:
        main_trainer.fit(model)
    main_trainer.test(model)


if __name__ == '__main__':
    parser = ArgumentParser()

    # setup arguments
    parser.add_argument('--dev', type=int, default=0, help='debugging mode')
    parser.add_argument('--gpus', type=int, required=True, help='number of gpus')
    parser.add_argument('--resume', type=str, default=None, help='path')
    parser.add_argument('--data-dir', type=str, required=True, help='data dir')
    parser.add_argument('--num-workers', type=int, default=4, help='number of cpus per gpu')
    parser.add_argument('--wandb', default=False, action='store_true', help='logging in wandb')
    parser.add_argument('--gpu-monitor', default=False, action='store_true',
                        help='monitors gpus. Note: slowing the training process')
    parser.add_argument('--data', type=str, default='imagenet', help='dataset name')
    parser.add_argument('--model', type=str, required=True, help='model name')

    # train arguments
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--max-epochs', type=int, default=120, help='number of fine-tuning epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batchsize for each gpu, for each train/test. i.e.: actual batchsize = 128 x num_gpus x 2')
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--cosine', default=False, action='store_true', help='apply cosine annealing lr scheduler')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--lamda', type=float, default=2, help='TransBoost loss hyperparameter')

    # test arguments
    parser.add_argument('--test-only', default=False, action='store_true', help='run testing only')

    args = parser.parse_args()

    pl.seed_everything(args.seed)

    data_params = {
        'data_dir': args.data_dir,
        'num_workers': args.num_workers,
        'batch_size': args.batch_size,
        'shuffle': True,
        'pin_memory': False,
        'drop_last': False,
        'seed': args.seed,
        'val_split': 0,
    }
    data = all_data_modules[args.data](**data_params)

    data.prepare_data()
    data.setup()

    args.num_classes = data.num_classes
    args.num_train_samples = data.num_samples
    args.num_test_samples = len(data.test_dataset(data.val_transforms))

    run(args)
