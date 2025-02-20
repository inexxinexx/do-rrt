# Use two nets `actor` and `actor_b` for Direction.FORWARD and Direction.BACKWARD, respectively

from .networks import Actor
import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import os
from .dataset_loader import get_traj_DataLoader, GoalOrientation
import json
import numpy as np

class Actor_agent():
    def __init__(self, mode='train', **kwargs):
        self.__dict__.update(kwargs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if mode == 'train':
            self.loader = get_traj_DataLoader(self.folder_path, self.batch_size, self.result_path) # this line produces `*_range.json`, so
                                                                                                   # it must be before `self.load_*_range()`
            
        self.load_pos_range()
        self.load_r_range()
        
        self.actor = Actor(self.state_dim, len(GoalOrientation), self.pos_min, self.pos_max).to(self.device)
        self.actor_b = Actor(self.state_dim, len(GoalOrientation), self.pos_min, self.pos_max).to(self.device)
        
        if torch.cuda.device_count() > 1:                                   # use multiple GPUs
            print('Using', torch.cuda.device_count(), 'GPUs.')
            self.actor = nn.DataParallel(self.actor)
            self.actor_b = nn.DataParallel(self.actor_b)
            
        if mode == 'train':
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=float(self.lr_a))
            self.actor_b_optimizer = optim.Adam(self.actor_b.parameters(), lr=float(self.lr_a))
                        
        self.last_saved_epoch = self.num_epochs // self.save_every_epochs * self.save_every_epochs - 1
        
        # Activate the `train` or `eval` mode
        if mode == 'train':
            self.actor.train()
            self.actor_b.train()
        else:
            self.actor.eval()
            self.actor_b.eval()
            
    def load_pos_range(self):
        with open(os.path.join(self.result_path, 'pos_range.json'), 'r') as f:
            pos_range = json.load(f)
            
        self.pos_min = torch.tensor(pos_range['min'], dtype=torch.float32).to(self.device)
        self.pos_max = torch.tensor(pos_range['max'], dtype=torch.float32).to(self.device)
        
    def load_r_range(self):
        with open(os.path.join(self.result_path, 'r_range.json'), 'r') as f:
            r_range = json.load(f)
            
        self.r_min = torch.tensor(r_range['min'], dtype=torch.float32).to(self.device)
        self.r_max = torch.tensor(r_range['max'], dtype=torch.float32).to(self.device)
    
    def save(self, epoch):
        print(f'\nSaving model(s) at epoch {epoch}...')
        if torch.cuda.device_count() > 1:                                       # models are instances of nn.DataParallel
            models_dict = {
                'actor': self.actor.module.state_dict(),
                'actor_b': self.actor_b.module.state_dict()
            }
        else:
            models_dict = {
                'actor': self.actor.state_dict(),
                'actor_b': self.actor_b.state_dict()
            }
        torch.save(models_dict, os.path.join(self.model_para_path, f'm2_epoch{epoch}.pth'))
        
    def load(self, epoch=None):
        if epoch is None:
            epoch = self.last_saved_epoch
            
        path = os.path.join(self.model_para_path, f'm2_epoch{epoch}.pth')
        if os.path.exists(path):
            print(f'\nLoading model(s) at epoch {epoch}...')
            
            if torch.cuda.is_available():
                checkpoint = torch.load(path)
            else:
                # Load onto CPU
                checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
                
            if torch.cuda.device_count() > 1:                                       # models are instances of nn.DataParallel
                self.actor.module.load_state_dict(checkpoint['actor'])
                self.actor_b.module.load_state_dict(checkpoint['actor_b'])
            else:
                self.actor.load_state_dict(checkpoint['actor'])
                self.actor_b.load_state_dict(checkpoint['actor_b'])
        else:
            print(f'Error: Cannot find model(s) at epoch {epoch}!\nExiting the program...\n')
            exit(1)
    
    def _train_net(self, batch):
        s_s, ns_s, g_s = batch
                
        s_s = s_s.to(self.device)
        ns_s = ns_s.to(self.device)
        g_s = g_s.to(self.device)
        
        # optimize self.actor
        pi = self.actor(s_s, g_s)
        pi_loss = F.mse_loss(pi, ns_s)
        
        self.actor_optimizer.zero_grad()
        pi_loss.backward()
        self.actor_optimizer.step()
        
        # optimize self.actor_b
        pi_b = self.actor_b(ns_s, g_s)
        pi_b_loss = F.mse_loss(pi_b, s_s)
        
        self.actor_b_optimizer.zero_grad()
        pi_b_loss.backward()
        self.actor_b_optimizer.step()
        
        return pi_loss.item(), pi_b_loss.item()
        
    def train(self):
        training_time = 0.0
        epoch_loss_recorder = {
            'pi_loss': np.zeros(self.num_epochs),
            'pi_b_loss': np.zeros(self.num_epochs)
        }
        
        for epoch in range(self.num_epochs):
            sum_pi_loss = 0.0
            sum_pi_b_loss = 0.0
            num_batches = len(self.loader)
            cur_time = time.time()
            
            for batch in self.loader:
                pi_loss, pi_b_loss = self._train_net(batch)
                
                sum_pi_loss += pi_loss
                sum_pi_b_loss += pi_b_loss
                            
            # Training info recording
            elapsed_time = time.time() - cur_time
            print('elapsed_time: {} = {}m {}s'.format(elapsed_time, elapsed_time // 60, elapsed_time % 60))
            
            ave_pi_loss = sum_pi_loss / num_batches
            ave_pi_b_loss = sum_pi_b_loss / num_batches
            print('ave_pi_loss: {:.12f}, ave_pi_b_loss: {:.12f}'.format(ave_pi_loss, ave_pi_b_loss))
            
            training_time += elapsed_time
            
            epoch_loss_recorder['pi_loss'][epoch] = ave_pi_loss
            epoch_loss_recorder['pi_b_loss'][epoch] = ave_pi_b_loss
            
            if (epoch + 1) % self.save_every_epochs == 0:
                self.save(epoch)
            
        print('training_time: {} = {}m {}s'.format(training_time, training_time // 60, training_time % 60))
        
        np.save(os.path.join(self.result_path, 'loss_epochs{}_{:.0f}.npy'.format(self.num_epochs, time.time())), epoch_loss_recorder)
        
    def predict(self, s, g):
        s = s.to(self.device)
        g = g.to(self.device)
        
        ns = self.actor(s, g)
        
        r_min = self.r_min
        r_max = self.r_max
        
        d = torch.norm(ns - s)
        if d < r_min or d > r_max:
            r = r_min if d < r_min else r_max
            ns = s + r / d * (ns - s)
        return ns
    
    def predict_b(self, ns, g):
        ns = ns.to(self.device)
        g = g.to(self.device)
        
        s = self.actor_b(ns, g)                                                     # the only difference from `predict()`
        
        r_min = self.r_min
        r_max = self.r_max
        
        d = torch.norm(s - ns)
        if d < r_min or d > r_max:
            r = r_min if d < r_min else r_max
            s = ns + r / d * (s - ns)
        return s