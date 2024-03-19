import numpy as np


def is_valid(trajs, tlens, threshold=30):
    masks = np.zeros(len(trajs))
    for i, traj in enumerate(trajs):
        if np.abs(traj[:tlens[i]]).max() < threshold:
            masks[i] = 1
    return masks


def get_tlen(trajs, normalizer, obs_dim):
    tlens = []
    max_len = trajs.shape[1]
    for traj in trajs:
        observations = normalizer.unnormalize(traj[..., :obs_dim], "observations")
        actions = normalizer.unnormalize(traj[..., obs_dim:], "actions")
        for i in range(max_len):
            if np.sum(np.abs(observations[i])) + np.sum(np.abs(actions[i])) < 1:
                tlens.append(i)
                break
            if i == max_len - 1:
                tlens.append(max_len)
    return np.array(tlens)


def flow_to_better(pref_model, diffusion_model, dataset, select_num, discount=1., threshold=1.05):
    trajs, init_obs = dataset.get_top_traj(select_num)
    tlens = get_tlen(trajs.detach().cpu().numpy(), dataset.normalizer, diffusion_model.observation_dim)
    score = pref_model.predict_traj_return(trajs.detach().cpu().numpy(), tlens, min_r=dataset.min_return, max_r=dataset.max_return, discount=discount)
    flow_step = dataset.improve_step - dataset.get_block_id(score.min())
    print("flow_step", flow_step)
    
    cond = trajs
    cond_tlens = get_tlen(trajs.detach().cpu().numpy(), dataset.normalizer, diffusion_model.observation_dim)
    cond_score = pref_model.predict_traj_return(trajs, cond_tlens, min_r=dataset.min_return, max_r=dataset.max_return, discount=discount)
    print("min_score: %.4f    max_score:%.4f    mean_score:%.4f"%(np.min(cond_score), np.max(cond_score), np.mean(cond_score)))
    
    generate_trajs = [cond.detach().cpu().numpy()]
    for i in range(flow_step):
        print("flow step:", i)
        generate_traj = diffusion_model.flow_one_step(cond, init_obs)
        
        tlens = get_tlen(generate_traj.detach().cpu().numpy(), dataset.normalizer, diffusion_model.observation_dim)
        score = pref_model.predict_traj_return(generate_traj, tlens, min_r=dataset.min_return, max_r=dataset.max_return, discount=discount)
        valid = is_valid(generate_traj.detach().cpu().numpy(), tlens)
        
        indices = []
        ratio = score / cond_score
        for j in range(len(score)):
            if valid[j] and ratio[j] > threshold:
                cond[j] = generate_traj[j]
                cond_score[j] = score[j]  
                indices.append(j)
        
        print("improve ratio: %.4f"%(len(indices) / len(cond)))
        print("min_score: %.4f    max_score:%.4f    mean_score:%.4f"%(np.min(cond_score), np.max(cond_score), np.mean(cond_score)))
        
        generate_trajs.append(generate_traj[indices].detach().cpu().numpy())
        if (len(indices) / len(cond)) < 0.05:
            break
    
    generate_trajs = np.concatenate(generate_trajs)
    tlens = get_tlen(generate_trajs, dataset.normalizer, diffusion_model.observation_dim)
    valid = is_valid(generate_trajs, tlens)
    score = pref_model.predict_traj_return(generate_trajs, tlens, dataset.min_return, dataset.max_return, discount=discount) * valid
    indices = np.argsort(score)[-select_num:]
    generate_trajs = generate_trajs[indices]
    tlens = tlens[indices]
    
    return generate_trajs, tlens
