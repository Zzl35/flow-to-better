import torch
import numpy as np

    
class Evaluator(object):
    def __init__(self, env, normalizer) -> None:
        self.env = env
        self.normalizer = normalizer
        
    def evaluate(self, actor, eval_num=50, device=torch.device('cuda')):
        actor.eval()
        eval_length, eval_reward = 0, 0.
        success = 0
        for _ in range(eval_num):
            state = self.env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    state = self.normalizer(state, 'observations')
                    s = torch.FloatTensor(state.reshape(1, -1)).to(device)
                action = actor.act(s)
                state, reward, done, info = self.env.step(action)
                eval_reward += reward
                eval_length += 1
                if "success" in info.keys():
                    if info['success'] > 0:
                        success += 1
                        done = True
                        
        returns = eval_reward / eval_num
        length = eval_length / eval_num
        succ_rate = success / eval_num
        if "success" in info.keys():
            score = returns
        else:
            try:
                score = self.env.get_normalized_score(returns) * 100
            except:
                if self.env.observation_space.shape[0] <= 12:
                    score = returns / 3294.
                else:
                    score = returns / 5143.
        metrics = {'return': returns, 'length': length, 'score': score, 'success': succ_rate}
        actor.train()
        
        return metrics