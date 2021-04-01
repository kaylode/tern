import os
import torch.nn as nn
import torch
from tqdm import tqdm
from .checkpoint import Checkpoint
import numpy as np
from loggers.loggers import Logger
import time
import pandas as pd
from torch.cuda import amp
# from utils.gradcam import GradCam, show_cam_on_image
from augmentations import Denormalize
from utils.utils import draw_retrieval_results

class Trainer():
    def __init__(self,
                config,
                model, 
                trainloader, 
                valloader,
                **kwargs):

        self.cfg = config
        self.model = model
        self.optimizer = model.optimizer
        self.criterion = model.criterion
        self.trainloader = trainloader
        self.valloader = valloader
        self.metrics = model.metrics #list of metrics
        self.set_attribute(kwargs)
        
        

    def fit(self, start_epoch = 0, start_iter = 0, num_epochs = 10 ,print_per_iter = None):
        self.num_epochs = num_epochs
        self.num_iters = (num_epochs+1) * len(self.trainloader)
        if self.checkpoint is None:
            self.checkpoint = Checkpoint(save_per_epoch = int(num_epochs/10)+1)

        if print_per_iter is not None:
            self.print_per_iter = print_per_iter
        else:
            self.print_per_iter = int(len(self.trainloader)/10)
        
        self.epoch = start_epoch

        # For one-cycle lr only
        if self.scheduler is not None and self.step_per_epoch:
            self.scheduler.last_epoch = start_epoch - 1

        self.start_iter = start_iter % len(self.trainloader)

        print(f'===========================START TRAINING=================================')
        for epoch in range(self.epoch, self.num_epochs):
            try:
                self.epoch = epoch
                self.training_epoch()

                self.iters=0
                if self.evaluate_per_epoch != 0:
                    if epoch % self.evaluate_per_epoch == 0 and epoch+1 >= self.evaluate_per_epoch:
                        self.evaluate_epoch()
                        
                if self.scheduler is not None and self.step_per_epoch:
                    self.scheduler.step()
                    lrl = [x['lr'] for x in self.optimizer.param_groups]
                    lr = sum(lrl) / len(lrl)
                    log_dict = {'Learning rate/Epoch': lr}
                    self.logging(log_dict)
                

            except KeyboardInterrupt:   
                self.checkpoint.save(self.model, save_mode = 'last', epoch = self.epoch, iters = self.iters, best_value=self.best_value)
                print("Stop training, checkpoint saved...")
                break

        print("Training Completed!")

    def training_epoch(self):
        self.model.train()

        running_loss = {}
        running_time = 0

        self.optimizer.zero_grad()
        for i, batch in enumerate(self.trainloader):
            
            start_time = time.time()
            
            with amp.autocast(enabled=self.use_amp):
                loss, loss_dict = self.model.training_step(batch)
                if self.use_accumulate:
                    loss /= self.accumulate_steps

            self.model.scaler(loss, self.optimizer)
            
            if self.use_accumulate:
                if (i+1) % self.accumulate_steps == 0 or i == len(self.trainloader)-1:
                    self.model.scaler.step(self.optimizer, clip_grad=self.clip_grad, parameters=self.model.parameters())
                    self.optimizer.zero_grad()

                    if self.scheduler is not None and not self.step_per_epoch:
                        self.scheduler.step((self.num_epochs + i) / len(self.trainloader))
                        lrl = [x['lr'] for x in self.optimizer.param_groups]
                        lr = sum(lrl) / len(lrl)
                        log_dict = {'Learning rate/Iterations': lr}
                        self.logging(log_dict)
            else:
                self.model.scaler.step(self.optimizer, clip_grad=self.clip_grad, parameters=self.model.parameters())
                self.optimizer.zero_grad()
                if self.scheduler is not None and not self.step_per_epoch:
                    # self.scheduler.step()
                    self.scheduler.step((self.num_epochs + i) / len(self.trainloader))
                    lrl = [x['lr'] for x in self.optimizer.param_groups]
                    lr = sum(lrl) / len(lrl)
                    log_dict = {'Learning rate/Iterations': lr}
                    self.logging(log_dict)
                
            torch.cuda.synchronize()

            end_time = time.time()

            for (key,value) in loss_dict.items():
                if key in running_loss.keys():
                    running_loss[key] += value
                else:
                    running_loss[key] = value

            running_time += end_time-start_time
            self.iters = self.start_iter + len(self.trainloader)*self.epoch + i + 1
            if self.iters % self.print_per_iter == 0:
                
                for key in running_loss.keys():
                    running_loss[key] /= self.print_per_iter
                    running_loss[key] = np.round(running_loss[key], 5)
                loss_string = '{}'.format(running_loss)[1:-1].replace("'",'').replace(",",' ||')
                print("[{}|{}] [{}|{}] || {} || Time: {:10.4f}s".format(self.epoch, self.num_epochs, self.iters, self.num_iters,loss_string, running_time))
                self.logging({"Training Loss/Batch" : running_loss['T']/ self.print_per_iter,})
                running_loss = {}
                running_time = 0

            if (self.iters % self.checkpoint.save_per_iter == 0 or self.iters == self.num_iters - 1):
                print(f'Save model at [{self.epoch}|{self.iters}] to last.pth')
                self.checkpoint.save(
                    self.model, 
                    save_mode = 'last', 
                    epoch = self.epoch, 
                    iters = self.iters, 
                    best_value=self.best_value)

    def evaluate_epoch(self):
        self.model.eval()
        epoch_loss = {}

        metric_dict = {}
        print('=============================EVALUATION===================================')
        start_time = time.time()
        with torch.no_grad():
            for batch in tqdm(self.valloader):      
                loss, loss_dict = self.model.evaluate_step(batch)
                
                for (key,value) in loss_dict.items():
                    if key in epoch_loss.keys():
                        epoch_loss[key] += value
                    else:
                        epoch_loss[key] = value

        end_time = time.time()
        running_time = end_time - start_time
        metric_dict = self.model.get_metric_values()
        self.model.reset_metrics()

        for key in epoch_loss.keys():
            epoch_loss[key] /= len(self.valloader)
            epoch_loss[key] = np.round(epoch_loss[key], 5)
        loss_string = '{}'.format(epoch_loss)[1:-1].replace("'",'').replace(",",' ||')
        print()
        print("[{}|{}] || {} || Time: {:10.4f} s".format(self.epoch, self.num_epochs, loss_string, running_time))

        for metric, score in metric_dict.items():
            print(metric +': ' + str(score), end = ' | ')
        print()
        print('==========================================================================')

        log_dict = {"Validation Loss/Epoch" : epoch_loss['T'] / len(self.valloader),}
        log_dict.update(metric_dict)
        self.logging(log_dict)

        # Save model gives best mf1 score
        if metric_dict['F1'] > self.best_value:
            self.best_value = metric_dict['F1']
            self.checkpoint.save(self.model, save_mode = 'best', epoch = self.epoch, iters = self.iters, best_value=self.best_value)

        if self.visualize_when_val:
            self.visualize_batch()

    def visualize_batch(self):
        # Vizualize Grad Class Activation Mapping
        if not os.path.exists('./samples'):
            os.mkdir('./samples')

        # Retrieval dict {post_id: {'post_ids': [], 'scores': []}
        retrieval_results = np.load('./results/query_results.npy', allow_pickle=True)
        
        batch = next(iter(self.valloader))
        post_ids = batch['post_ids'][:3]

        query_pool = os.path.join('data', self.cfg.project_name, self.cfg.val_imgs)
        gallery_pool = os.path.join('data', self.cfg.project_name, self.cfg.train_imgs)
        df = pd.read_csv('./data/shopee-matching/annotations/train_clean.csv')
        for idx, post_id in enumerate(post_ids):
            query_post = df[df.posting_id == post_id]
            query_image = os.path.join(query_pool, query_post.image.values[0])
            query_title = query_post.cleaned_title.values[0]
            queries = [[query_image, query_title]]

            gallery_post_ids = retrieval_results.item()[post_id]
            
            top_k_relevant_post_scores = gallery_post_ids['scores'][:5]
            top_k_relevant_post_ids = gallery_post_ids['post_ids'][:5]

            top_k_relevant_posts = df[df.posting_id.isin(top_k_relevant_post_ids)]
            top_k_relevant_posts = [
                (os.path.join(gallery_pool,image_name), title) for (image_name, title) in zip(
                    top_k_relevant_posts['image'], 
                    top_k_relevant_posts['cleaned_title'])
            ]
      
            image_outname = os.path.join('samples', f'{self.epoch}_{self.iters}_{idx}.jpg')
            fig = draw_retrieval_results(queries, top_k_relevant_posts, image_outname)
            self.logger.write_image('samples', fig)

    def logging(self, logs):
        tags = [l for l in logs.keys()]
        values = [l for l in logs.values()]
        self.logger.write(tags= tags, values= values)

    def set_accumulate_step(self):
        self.use_accumulate = False
        if self.cfg.total_accumulate_steps > 0:
            self.use_accumulate = True
            self.accumulate_steps = max(round(self.cfg.total_accumulate_steps / self.cfg.batch_size), 1) 

    def set_amp(self):
        self.use_amp = False
        if self.cfg.mixed_precision:
            self.use_amp = True

    def __str__(self):
        s0 =  "##########   MODEL INFO   ##########"
        s1 = "Model name: " + self.model.model_name
        s2 = f"Number of trainable parameters:  {self.model.trainable_parameters():,}"
       
        s5 = "Training iterations per epoch: " + str(len(self.trainloader))
        s6 = "Validating iterations per epoch: " + str(len(self.valloader))
        return "\n".join([s0,s1,s2,s5,s6])

    def set_attribute(self, kwargs):
        self.checkpoint = None
        self.scheduler = None
        self.clip_grad = 10.0
        self.logger = None
        self.visualize_when_val = True
        self.step_per_epoch = False
        self.evaluate_per_epoch = 1
        self.best_value = 0.0
        self.set_accumulate_step()
        self.set_amp()
        for i,j in kwargs.items():
            setattr(self, i, j)

        if self.logger is None:
            self.logger = Logger()