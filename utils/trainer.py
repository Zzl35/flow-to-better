import os
import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from copy import deepcopy
from utils.helpers import EMA


class Trainer:
    def __init__(self, preference_model, diffusion_model, actor, dataset, logger, timer, evaluator, device='cuda:0') -> None:
        self.preference_model = preference_model
        self.diffusion_model = diffusion_model
        self.actor = actor
        self.dataset = dataset
        self.logger = logger
        self.timer = timer
        self.evaluator = evaluator
        self.device = device
    
    def train_preference(self, optim, scheduler, max_iters, num_steps_per_iter, batch_size):
        self.preference_model.train()
        sigmoid = nn.Sigmoid()
        loss = nn.BCELoss()
                
        self.timer.reset()
        for iter in range(max_iters):
            for step in range(num_steps_per_iter):
                (obss_1, obss_2), (actions_1, actions_2), (timesteps_1, timesteps_2), (masks_1, masks_2), prefs = self.dataset.sample_prefs(batch_size)

                scores_1 = self.preference_model(obss_1, actions_1, timesteps_1, masks_1)
                scores_2 = self.preference_model(obss_2, actions_2, timesteps_2, masks_2)

                scores = sigmoid(torch.cat([scores_1, scores_2], -1))
                mean_scores = torch.mean(scores).item()
                pref_loss = loss(scores, prefs)
                
                optim.zero_grad()
                pref_loss.backward()
                nn.utils.clip_grad_norm_(self.preference_model.parameters(), .25)
                optim.step()

                self.logger.logkv_mean('preference_loss', pref_loss.item())
                self.logger.logkv_mean('preference_scores', mean_scores)

                pred = torch.argmax(torch.cat([scores_1, scores_2],dim=-1),dim=-1)
                label = torch.argmax(prefs, dim=-1)
                correct = torch.sum(pred==label).item() / len(pred)
                self.logger.logkv_mean('train_accuracy', correct)

                scheduler.step()
                
            # log
            elapsed_time, total_time = self.timer.reset()
            elapsed_fps = num_steps_per_iter / elapsed_time
            self.logger.logkv('preference model fps', elapsed_fps)
            self.logger.logkv('preference model total time', total_time)
            
            self.logger.set_timestep(iter)
            self.logger.dumpkvs(exclude=["diffusion_training_progress"])

            if iter % 10 == 0:
                torch.save(self.preference_model.state_dict(), os.path.join(self.logger.checkpoint_dir, f'preference-{iter}.pth'))

            if (iter + 1) % 5 ==0: 
                save_path = os.path.join(self.logger._video_dir, 'iter=%d.png'%(iter+1))
                self.render_preference_model(save_path, discount=1.)
        
        # save
        torch.save(self.preference_model.state_dict(), os.path.join(self.logger.model_dir, 'preference.pth'))

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.diffusion_model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.diffusion_model)

    def train_diffusion(
            self,
            optim,
            ema_decay,
            epoch_start_ema,
            update_ema_every,
            max_iters,
            num_steps_per_iter,
            batch_size,
            render_epoch=20,
        ):
        self.ema = EMA(ema_decay)
        self.ema_model = deepcopy(self.diffusion_model)
        self.update_ema_every = update_ema_every
        self.epoch_start_ema = epoch_start_ema
        self.step = 0

        self.diffusion_model.train()
        
        self.timer.reset()
        for iter in range(max_iters):
            # train step    
            for step in range(num_steps_per_iter):
                condition_traj, target_traj, init_obs, _ = self.dataset.sample_batch(batch_size)

                loss, infos = self.diffusion_model.loss(target_traj, condition_traj, init_obs)
                
                optim.zero_grad()
                loss.backward()
                optim.step()

                self.step += 1

                for k, v in infos.items():
                    self.logger.logkv_mean(k, v.item())
                    
            elapsed_time, total_time = self.timer.reset()
            elapsed_fps = num_steps_per_iter / elapsed_time
            self.logger.logkv('diffusion fps', elapsed_fps)
            self.logger.logkv('diffusion total time', total_time)

            if (iter + 1) % render_epoch == 0:
                torch.save(self.diffusion_model.state_dict(), os.path.join(self.logger.model_dir, f'diffusion-%d.pth'%(iter+1)))

            # log
            self.logger.set_timestep(iter)
            self.logger.dumpkvs(exclude=["diffusion_training_progress"])
            
    def train_actor(
        self, 
        dataset,
        optim, 
        max_iters, 
        num_steps_per_iter,
        batch_size
    ):
        
        self.step = 0
        self.timer.reset()
        self.actor.train()

        for iter in range(max_iters):
            for step in range(num_steps_per_iter):
                observations, actions = dataset.sample_batch(batch_size)
                actor_loss = self.actor.loss(observations, actions)
                
                optim.zero_grad()
                actor_loss.backward()
                optim.step()

                self.step += 1
                self.logger.logkv('actor loss', actor_loss.item())
            
            # log
            elapsed_time, total_time = self.timer.reset()
            elapsed_fps = num_steps_per_iter / elapsed_time
            self.logger.logkv('actor fps', elapsed_fps)
            self.logger.logkv('actor total time', total_time)
            
            if (iter + 1) % 10 == 0:
                for k, v in self.evaluator.evaluate(self.actor).items():
                    self.logger.logkv(k, v)
                self.logger.set_timestep(iter)
                self.logger.dumpkvs(exclude=["actor_training_progress"])
            
        torch.save(self.actor.state_dict(), os.path.join(self.logger.model_dir, f'actor.pth'))

    @torch.no_grad()
    def render_preference_model(self, save_path, discount=1., batch_size=32):
        batch_size = batch_size
        start = 0
        scores = np.zeros_like(self.dataset.returns)
        while start < len(self.dataset.trajs):
            end = min(start + batch_size, len(self.dataset.trajs))
            observations, actions, returns, timesteps, masks = self.dataset.get_batch(np.arange(start, end))
            scores[start: end] = self.preference_model._predict_traj_return(observations, actions, timesteps, masks, discount=discount)
            start += batch_size

        returns = self.dataset.returns
        sort_idxs = returns.argsort()

        normed_returns = (returns[sort_idxs] - returns.min()) / (returns.max() - returns.min())
        normed_prefs = (scores[sort_idxs] - scores.min()) / (scores.max() - scores.min())
        
        plt.plot(np.arange(len(returns)), normed_returns)
        plt.plot(np.arange(len(returns)), normed_prefs)
        # plt.show()
        plt.savefig(save_path)
        plt.close()

