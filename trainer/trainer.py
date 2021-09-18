import os
import time
from matplotlib.pyplot import show
import numpy as np
from tqdm import tqdm
import pandas as pd

import torch
from torch.cuda import amp

from .checkpoint import Checkpoint
from loggers.loggers import Logger


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
        self.iters = self.start_iter + len(self.trainloader)*self.epoch + 1

        print(f'===========================START TRAINING=================================')
        for epoch in range(self.epoch, self.num_epochs):
            try:
                self.epoch = epoch
                self.training_epoch()

                if self.evaluate_per_epoch != 0:
                    if epoch % self.evaluate_per_epoch == 0 and epoch+1 >= self.evaluate_per_epoch:
                        self.evaluate_epoch()

                if self.scheduler is not None and self.step_per_epoch:
                    self.scheduler.step()
                    lrl = [x['lr'] for x in self.optimizer.param_groups]
                    lr = sum(lrl) / len(lrl)
                    log_dict = {'Training/Learning rate': lr}
                    self.logging(log_dict, step=self.epoch)
                

            except KeyboardInterrupt:   
                self.checkpoint.save(
                    self.model, 
                    save_mode = 'last', 
                    epoch = self.epoch, 
                    iters = self.iters, 
                    best_value=self.best_value,
                    class_names=None,
                    config=self.cfg)
                print("Stop training, checkpoint saved...")
                break

        print("Training Completed!")

    def training_epoch(self):
        self.model.train()

        running_loss = {}
        running_time = 0

        self.optimizer.zero_grad()
        self.trainloader.create_batches()
        for i, raw_batch in enumerate(self.trainloader.batches):
            
            start_time = time.time()
            batch = self.trainloader.collate_fn(raw_batch)
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
                        self.scheduler.step()
                        lrl = [x['lr'] for x in self.optimizer.param_groups]
                        lr = sum(lrl) / len(lrl)
                        log_dict = {'Training/Learning rate': lr}
                        self.logging(log_dict, step=self.iters)
            else:
                self.model.scaler.step(self.optimizer, clip_grad=self.clip_grad, parameters=self.model.parameters())
                self.optimizer.zero_grad()
                if self.scheduler is not None and not self.step_per_epoch:
                    self.scheduler.step()
                    lrl = [x['lr'] for x in self.optimizer.param_groups]
                    lr = sum(lrl) / len(lrl)
                    log_dict = {'Training/Learning rate': lr}
                    self.logging(log_dict, step=self.iters)
                

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
                
                log_dict = {f"Training/{k} Loss": v/self.print_per_iter for k,v in running_loss.items()}
                self.logging(log_dict, step=self.iters)
                running_loss = {}
                running_time = 0

            if (self.iters % self.checkpoint.save_per_iter == 0 or self.num_iters == self.num_iters - 1):
                print(f'Save model at [{self.iters}|{self.num_iters}] to last.pth')
                self.checkpoint.save(
                    self.model, 
                    save_mode = 'last', 
                    epoch = self.epoch, 
                    iters = self.iters, 
                    best_value=self.best_value,
                    class_names=None,
                    config=self.cfg)
                
    def evaluate_epoch(self):
        self.model.eval()
        epoch_loss = {}

        metric_dict = {}
        print('=============================EVALUATION===================================')
        start_time = time.time()
        with torch.no_grad():
            for batch in tqdm(self.valloader):
                _, loss_dict = self.model.evaluate_step(batch)
                
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

        log_dict = {f"Validation/{k} Loss": v/len(self.valloader) for k,v in epoch_loss.items()}

        metric_log_dict = {f"Validation/{k}":v for k,v in metric_dict.items()}
        log_dict.update(metric_log_dict)
        self.logging(log_dict, step=self.epoch)

        # Save model gives best average MAP score
        if metric_dict['MAP'] > self.best_value:
            self.best_value = metric_dict['MAP']
            self.checkpoint.save(
                self.model, 
                save_mode = 'best', 
                epoch = self.epoch, 
                iters = self.iters, 
                best_value=self.best_value,
                class_names=None,
                config=self.cfg)

        if self.visualize_when_val:
            self.visualize_batch()
        
    def visualize_batch(self):
        from utils.utils import draw_retrieval_results
        import random

        # Retrieval dict {post_id: {'post_ids': [], 'scores': []}
        retrieval_results = np.load('./results/query_results.npy', allow_pickle=True)
        
        query_ids = list(retrieval_results.item().keys())
        query_ids = random.choices(query_ids, k=32)

        for idx, query_id in enumerate(query_ids):
            query_id = int(query_id)
            gallery_ids = retrieval_results.item()[query_id]
            
            top_k_relevant_image_scores = gallery_ids['scores'][:5]
            top_k_relevant_image_ids = gallery_ids['image_ids'][:5]
            ground_truth_id = gallery_ids['target_ids']

            top_k_relevant_image_paths = self.valloader.dataset.load_image_by_id(top_k_relevant_image_ids)
            ground_truth_image_path = self.valloader.dataset.load_image_by_id(ground_truth_id)
            query = self.valloader.dataset.load_annotations_by_id(query_id)[0]

            top_k_relevant_results = [i for i in zip(top_k_relevant_image_paths, top_k_relevant_image_scores)]
            fig = draw_retrieval_results(query, top_k_relevant_results, gt_path = ground_truth_image_path, figsize=(25,6))
            self.logger.write_image(query, fig, step=self.epoch)
        
       
    def logging(self, logs, step):
        tags = [l for l in logs.keys()]
        values = [l for l in logs.values()]
        self.logger.write(tags= tags, values= values, step=step)

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
        self.step_per_epoch = False
        self.evaluate_per_epoch = 1
        self.visualize_when_val = True
        self.best_value = 0.0
        self.set_accumulate_step()
        self.set_amp()

        for i,j in kwargs.items():
            setattr(self, i, j)

        if self.logger is None:
            self.logger = Logger()