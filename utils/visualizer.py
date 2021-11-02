"""
created by songkey@pku.edu.cn
date @ 2021-10-15
"""
import visdom
import numpy as np
import cv2

class Visualizer():
    def __init__(self, args):
        self.data = dict(
            train_all_history = [],
            test_list_hist = [],
            train_list_hist = [],
            info_len = 20,
            sil = 40,
            wind = 200,
            linex = 0,
        )

        self.vis = None
        if args.use_visdom:
            try:
                if args.visdom_id == '':
                    self.vis = visdom.Visdom(server=args.visdom_ip, port=args.visdom_port, env=args.visdom_name)
                else:
                    self.vis = visdom.Visdom(server=args.visdom_ip, port=args.visdom_port, env=args.visdom_name,
                                             username=args.visdom_id, password=args.visdom_pw)
            except:
                print('Visdom connect error!')
                self.vis = None

    def load_data(self, checkpoints):
        if 'visdom_data' in checkpoints:
            self.data = checkpoints['visdom_data']

    def display_test_hist(self, txt):
        self.data['test_list_hist'] += txt.split('\n')

        if self.vis is None: pass
        try:
            self.vis.text('<br>'.join(self.data['test_list_hist']), win='test log',
                          opts=dict(title='testing log'))
        except: self.vis = None

    def display_test_clean(self):
        self.data['test_screen'] = []

    def display_test(self, txt):
        self.data['test_screen'] += txt.split('\n')

        if self.vis is None: pass
        try:
            self.vis.text('<br>'.join(self.data['test_screen']), win='testing screen',
                          opts=dict(title='testing screen'))
        except: self.vis = None

    def display_train_hist(self, info):
        self.data['train_list_hist'].append(info)
        self.data['train_all_history'].append(info)
        if len(self.data['train_list_hist']) > self.data['info_len']:
            self.data['train_list_hist'] = self.data['train_list_hist'][1:]

        if self.vis is None: pass
        try:
            self.vis.text('<br>'.join(self.data['train_list_hist']), win='disp train_list',
                          opts=dict(title='training log'))
        except: self.vis = None

    def show_image(self, image, win_id, text):
        self.data[win_id] = image
        if self.vis is None: pass
        try:
            self.vis.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1), win=win_id,
                                 opts=dict(title=text, caption=''))
        except: self.vis = None

    def plot_test_loss(self, epoch, loss_mean, loss_names):
        if not 'plot_test_loss_mean' in self.data:
            self.data['plot_test_loss_mean'] = loss_mean.reshape(1, -1)
            self.data['plot_test_loss_epoch'] = [epoch]
            self.data['plot_test_loss_names'] = loss_names
        else:
            self.data['plot_test_loss_mean'] = np.vstack((self.data['plot_test_loss_mean'], loss_mean.reshape(1, -1)))
            self.data['plot_test_loss_epoch'].append(epoch)

        if self.vis is None: pass
        try:
            self.vis.line(
                X=self.data['plot_test_loss_epoch'],
                Y=self.data['plot_test_loss_mean'],
                opts={
                    'title': 'test loss over epoch {}'.format(epoch),
                    'xlabel': 'epoch',
                    'legend': self.data['plot_test_loss_names'],
                    'ylabel': 'loss'},
                win='test_loss_dict')
        except: self.vis = None

    def plot_train_loss(self, epoch, iteration, it_start, loss_vec, loss_names):
        if not 'plot_train_loss_vec' in self.data:
            self.data['plot_train_loss_vec'] = loss_vec.reshape(1, -1)
            self.data['plot_train_loss_ct'] = [it_start]
            self.data['plot_train_loss_ep'] = [epoch]
            self.data['plot_train_loss_it'] = [iteration]
            self.data['plot_train_loss_names'] = loss_names
        else:
            self.data['plot_train_loss_vec'] = np.vstack((self.data['plot_train_loss_vec'], loss_vec.reshape(1, -1)))
            self.data['plot_train_loss_ct'].append(it_start)
            self.data['plot_train_loss_ep'].append(epoch)
            self.data['plot_train_loss_it'].append(iteration)

        if self.vis is None: pass

        data_num = len(self.data['plot_train_loss_ct'])

        # screen
        if data_num <= self.data['wind']:
            X = self.data['plot_train_loss_ct']
            Y = self.data['plot_train_loss_vec']
        else:
            X = self.data['plot_train_loss_ct'][-self.data['wind']:]
            Y = self.data['plot_train_loss_vec'][-self.data['wind']:, :]

        try:
            self.vis.line(
                X=X,
                Y=Y,
                opts={
                    'title': 'train loss screen ep{} it{}'.format(epoch, iteration),
                    'xlabel': 'ct',
                    'legend': self.data['plot_train_loss_names'],
                    'ylabel': 'loss'},
                win='train_loss_dict')
        except: self.vis = None

        # history
        if data_num >= self.data['sil']:
            mean_vec = np.mean(self.data['plot_train_loss_vec'][-self.data['sil']:], axis=0).reshape(1, -1)
            if not 'plot_train_loss_vec_mean' in self.data:
                self.data['plot_train_loss_vec_mean'] = mean_vec
            else:
                self.data['plot_train_loss_vec_mean'] = np.vstack((self.data['plot_train_loss_vec_mean'], mean_vec))

            try:
                self.vis.line(
                    X=self.data['plot_train_loss_ct'][-self.data['plot_train_loss_vec_mean'].shape[0]:],
                    Y=self.data['plot_train_loss_vec_mean'],
                    opts={
                        'title': 'train loss history ep{} it{}'.format(epoch, iteration),
                        'xlabel': 'ct',
                        'legend': self.data['plot_train_loss_names'],
                        'ylabel': 'loss'},
                    win='train_loss_dict_history')
            except: self.vis = None

