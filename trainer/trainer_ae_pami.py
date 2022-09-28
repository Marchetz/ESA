import os
import matplotlib.pyplot as plt
import datetime
import io
from PIL import Image
from torchvision.transforms import ToTensor
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models.model_ae import model_ae
import dataset_invariance
from torch.autograd import Variable
import tqdm
import pdb


class Trainer:
    def __init__(self, config):
        """
        The Trainer class handles the training procedure for training the autoencoder.
        :param config: configuration parameters (see train_ae.py)
        """

        # test folder creating
        self.name_test = str(datetime.datetime.now())[:19]
        self.folder_tensorboard = 'runs/runs_ae_pami/'
        self.folder_test = 'training/training_ae_pami/' + self.name_test + '_' + config.info
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)
        self.folder_test = self.folder_test + '/'
        self.file = open(self.folder_test + "details.txt", "w")

        print('Creating dataset...')
        tracks = json.load(open(config.dataset_file))
        self.dim_clip = 180
        self.data_train = dataset_invariance.TrackDataset(tracks,
                                                          len_past=config.past_len,
                                                          len_future=config.future_len,
                                                          train=True,
                                                          dim_clip=self.dim_clip)
        self.train_loader = DataLoader(self.data_train,
                                       batch_size=config.batch_size,
                                       num_workers=1,
                                       shuffle=True
                                       )
        self.data_test = dataset_invariance.TrackDataset(tracks,
                                                         len_past=config.past_len,
                                                         len_future=config.future_len,
                                                         train=False,
                                                         dim_clip=self.dim_clip)
        self.test_loader = DataLoader(self.data_test,
                                      batch_size=config.batch_size,
                                      num_workers=1,
                                      shuffle=False
                                      )
        print('Dataset created')

        self.settings = {
            "batch_size": config.batch_size,
            "use_cuda": config.cuda,
            "dim_feature_tracklet": config.past_len * 2,
            "dim_feature_future": config.future_len * 2,
            "dim_embedding_key": config.dim_embedding_key,
            "past_len": config.past_len,
            "future_len": config.future_len,
        }
        self.max_epochs = config.max_epochs

        # model
        self.mem_n2n = model_ae(self.settings)
        #self.mem_n2n = torch.load('training/training_ae/2020-05-09 22:54:47_dimEmbed_60/model_ae_2020-05-09 22:54:47')

        # loss
        self.criterionLoss = nn.MSELoss()
        self.criterionLoss_map = nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.mem_n2n.parameters(), lr=config.learning_rate)
        self.iterations = 0
        if config.cuda:
            self.criterionLoss = self.criterionLoss.cuda()
            self.criterionLoss_map = self.criterionLoss_map.cuda()
            self.mem_n2n = self.mem_n2n.cuda()
        self.start_epoch = 0
        self.config = config

        # Write details to file
        self.write_details()
        self.file.close()

        # Tensorboard summary: configuration
        self.writer = SummaryWriter(self.folder_tensorboard + self.name_test + '_' + config.info)
        self.writer.add_text('Training Configuration', 'model name: {}'.format(self.mem_n2n.name_model), 0)
        self.writer.add_text('Training Configuration', 'dataset train: {}'.format(len(self.data_train)), 0)
        self.writer.add_text('Training Configuration', 'dataset test: {}'.format(len(self.data_test)), 0)
        self.writer.add_text('Training Configuration', 'batch_size: {}'.format(self.config.batch_size), 0)
        self.writer.add_text('Training Configuration', 'learning rate init: {}'.format(self.config.learning_rate), 0)
        self.writer.add_text('Training Configuration', 'dim_embedding_key: {}'.format(self.config.dim_embedding_key), 0)

    def write_details(self):
        """
        Serialize configuration parameters to file.
        """

        self.file.write('points of past track: {}'.format(self.config.past_len) + '\n')
        self.file.write('points of future track: {}'.format(self.config.future_len) + '\n')
        self.file.write('train size: {}'.format(len(self.data_train)) + '\n')
        self.file.write('test size: {}'.format(len(self.data_test)) + '\n')
        self.file.write('batch size: {}'.format(self.config.batch_size) + '\n')
        self.file.write('learning rate: {}'.format(self.config.learning_rate) + '\n')
        self.file.write('embedding dim: {}'.format(self.config.dim_embedding_key) + '\n')

    def draw_track(self, past, future, pred=None, scene=None, index_tracklet=0, num_epoch=0):
        """
        Plot past and future trajectory and save it to tensorboard.
        :param past: the observed trajectory
        :param future: ground truth future trajectory
        :param pred: predicted future trajectory
        :param index_tracklet: index of the trajectory in the dataset (default 0)
        :param num_epoch: current epoch (default 0)
        :param save_fig: True or False if you want to save the plot as figure to Tensorboard
        :param train: True or False, indicates whether the sample is in the training or testing set
        :return: None
        """

        plt.imshow(scene)
        past = past.cpu().numpy()
        future = future.cpu().numpy()
        plt.plot(past[:, 0] * 2 + 60, past[:, 1] * 2, c='blue', marker='o', markersize=3)
        plt.plot(future[:, 0] * 2 + 60, future[:, 1] * 2, c='green', marker='o', markersize=3)

        if pred is not None:
            pred = pred.cpu().numpy()
            plt.plot(pred[:, 0] * 2 + 60, pred[:, 1] * 2, color='red', linewidth=1, marker='o', markersize=1)
        plt.axis('equal')

        # Save figure in Tensorboard
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)
        self.writer.add_image('Image_test/track' + str(index_tracklet), image.squeeze(0), num_epoch)

        plt.close()

    def fit(self):
        """
        Autoencoder training procedure. The function loops over the data in the training set max_epochs times.
        :return: None
        """
        config = self.config
        # Training loop
        for epoch in range(self.start_epoch, config.max_epochs):

            print(' ----- Epoch: {}'.format(epoch))
            loss = self._train_single_epoch()
            print('Loss: {}'.format(loss))

            if (epoch + 1) % 10 == 0:
                print('test on train dataset')
                #dict_metrics_train = self.evaluate(self.train_loader, epoch + 1)

                print('test on TEST dataset')
                dict_metrics_test = self.evaluate(self.test_loader, epoch + 1)

                # Tensorboard summary: learning rate
                for param_group in self.opt.param_groups:
                    self.writer.add_scalar('learning_rate', param_group["lr"], epoch)

                # Tensorboard summary: train
                # self.writer.add_scalar('accuracy_train/eucl_mean', dict_metrics_train['eucl_mean'], epoch)
                # self.writer.add_scalar('accuracy_train/Horizon10s', dict_metrics_train['horizon10s'], epoch)
                # self.writer.add_scalar('accuracy_train/Horizon20s', dict_metrics_train['horizon20s'], epoch)
                # self.writer.add_scalar('accuracy_train/Horizon30s', dict_metrics_train['horizon30s'], epoch)
                # self.writer.add_scalar('accuracy_train/Horizon40s', dict_metrics_train['horizon40s'], epoch)
                # self.writer.add_scalar('accuracy_test/accuracy_map', dict_metrics_train['accuracy_map'], epoch)

                # Tensorboard summary: test
                self.writer.add_scalar('accuracy_test/eucl_mean', dict_metrics_test['eucl_mean'], epoch)
                self.writer.add_scalar('accuracy_test/Horizon10s', dict_metrics_test['horizon10s'], epoch)
                self.writer.add_scalar('accuracy_test/Horizon20s', dict_metrics_test['horizon20s'], epoch)
                self.writer.add_scalar('accuracy_test/Horizon30s', dict_metrics_test['horizon30s'], epoch)
                self.writer.add_scalar('accuracy_test/Horizon40s', dict_metrics_test['horizon40s'], epoch)
                self.writer.add_scalar('accuracy_test/accuracy_map', dict_metrics_test['accuracy_map'], epoch)

                # Save model checkpoint
                torch.save(self.mem_n2n, self.folder_test + 'model_ae_epoch_' + str(epoch) + '_' + self.name_test)

                # Tensorboard summary: model weights
                for name, param in self.mem_n2n.named_parameters():
                    self.writer.add_histogram(name, param.data, epoch)

        # Save final trained model
        torch.save(self.mem_n2n, self.folder_test + 'model_ae_' + self.name_test)

    def evaluate(self, loader, epoch=0):
        """
        Evaluate the model.
        :param loader: pytorch dataloader to loop over the data
        :param epoch: current epoch (default 0)
        :return: a dictionary with performance metrics
        """

        iou_total = eucl_mean = horizon10s = horizon20s = horizon30s = horizon40s = 0
        dict_metrics = {}

        # Loop over samples
        self.mem_n2n.eval()
        with torch.no_grad():
            for step, (index, past, future, presents, angle_presents, videos, vehicles, number_vec, scene, scene_irm, scene_one_hot) \
                    in enumerate(tqdm.tqdm(loader)):
                scene_input = scene_one_hot
                scene_gt = scene_one_hot.max(1)[1].type(torch.cuda.LongTensor)
                if self.config.cuda:
                    past = past.cuda()
                    future = future.cuda()
                    scene_input = scene_input.cuda()
                    scene_gt = scene_gt.cuda()
                pred, pred_map = self.mem_n2n(past, future, scene_input)
                pred = pred.data
                pred_map = pred_map.data

                pred_map = pred_map.permute(0, 2, 3, 1)
                _, pred_map = torch.max(pred_map, 3)
                SMOOTH = 1e-6
                intersection = (pred_map & scene_gt).float().sum((1, 2))
                union = (pred_map | scene_gt).float().sum((1, 2))
                iou = (intersection + SMOOTH) / (union + SMOOTH)
                iou_total += iou.sum()

                distances = torch.norm(pred - future, dim=2)
                eucl_mean += torch.sum(torch.mean(distances, 1))
                horizon10s += torch.sum(distances[:, 9])
                horizon20s += torch.sum(distances[:, 19])
                horizon30s += torch.sum(distances[:, 29])
                horizon40s += torch.sum(distances[:, 39])

                # Draw sample: the first of the batch
                if loader == self.test_loader:
                    self.draw_track(past[0],
                                    future[0],
                                    pred[0],
                                    scene=pred_map[0].cpu().numpy(),
                                    index_tracklet=index[0],
                                    num_epoch=epoch)

            dict_metrics['eucl_mean'] = eucl_mean / len(loader.dataset)
            dict_metrics['horizon10s'] = horizon10s / len(loader.dataset)
            dict_metrics['horizon20s'] = horizon20s / len(loader.dataset)
            dict_metrics['horizon30s'] = horizon30s / len(loader.dataset)
            dict_metrics['horizon40s'] = horizon40s / len(loader.dataset)
            dict_metrics['accuracy_map'] = iou_total.item() / len(loader.dataset)

        return dict_metrics

    def _train_single_epoch(self):
        """
        Training loop over the dataset for an epoch
        :return: loss
        """
        config = self.config
        self.mem_n2n.train()
        for step, (index, past, future, presents, angle_presents, videos, vehicles, number_vec, scene, scene_irm, scene_one_hot) \
                in enumerate(tqdm.tqdm(self.train_loader)):
            self.iterations += 1
            scene_input = scene_one_hot
            scene_gt = scene_one_hot.max(1)[1].type(torch.cuda.LongTensor)
            if config.cuda:
                past = past.cuda()
                future = future.cuda()
                scene_input = scene_input.cuda()
            self.opt.zero_grad()

            # Get prediction and compute loss
            pred, map_pred = self.mem_n2n(past, future, scene_input)
            loss_pred = self.criterionLoss(pred, future)
            loss_map = self.criterionLoss_map(map_pred, scene_gt)
            loss = loss_pred + loss_map
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.mem_n2n.parameters(), 1.0, norm_type=2)
            self.opt.step()

            # Tensorboard summary: loss
            self.writer.add_scalar('loss/loss_total', loss, self.iterations)
            self.writer.add_scalar('loss/loss_pred', loss_pred, self.iterations)
            self.writer.add_scalar('loss/loss_map', loss_map, self.iterations)

        return loss.item()
