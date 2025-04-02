# python train.py --num_envs 3 --num_steps 600 --reset_envs

import os
import tyro
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from libero.libero.utils.video_utils import VideoWriter

from utils.openvla import OpenVLAConfig, make_libero_envs, get_openvla_model, get_openvla_action, get_dummy_action
from utils.libero import MODALITY_CONFIG

from policy.agent import LiberoAgent

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    save_videos: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    video_folder: str = './videos'
    """the path to save videos"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to wandb"""

    
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 3
    """the number of parallel environments for trajectory collection"""
    num_eval_envs: int = 5
    """the number of parallel environments for evaluation"""
    eval_frequency: int = 1
    """the number of iterations to run evaluation"""
    num_steps: int = 600
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    actor_checkpoint_path: str = None
    """the checkpoint to use for the actor network"""
    pretrain_value_iters: int = 0
    """the number of pretraining iterations for the value network"""
    reset_envs: bool = False
    """whether to reset the 'done' environments during trajectory collection"""
    num_steps_wait: int = 10
    """Number of steps to wait for objects to stabilize in sim"""

    libero_task_suite: str = 'libero_object'
    """the LIBERO task suite"""
    libero_task_id: int = 0
    """the LIBERO task id"""
    camera_dim: int = 256
    """the camera dimension for the environment (height and width)"""
    openvla_checkpoint_path: str = "openvla/openvla-7b-finetuned-libero-object"
    """the checkpoint path for the OpenVLA model"""
    teacher_coef: float = 1.0

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""



def perform_evaluation(eval_policy, global_step, args):
    envs, initial_states, _, _ = make_libero_envs(
        num_envs=args.num_eval_envs, 
        task_suite_name=args.libero_task_suite, 
        task_id=args.libero_task_id, 
        horizon=args.num_steps+args.num_steps_wait,
        camera_dim=args.camera_dim
    )

    envs.reset()
    next_obs = envs.set_init_state(initial_states)

    rewards = torch.zeros((args.num_steps, args.num_eval_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_eval_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_eval_envs)).to(device)

    video_writer = VideoWriter(args.video_folder, args.save_videos)
    next_done = torch.zeros(args.num_eval_envs).to(device)

    for step in range(0, args.num_steps):
        print(f"eval step: {step}")
        dones[step] = next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, _, _, _, value = eval_policy.get_action_and_value(next_obs)
            values[step] = value.flatten()

        next_obs, reward, next_done, _ = envs.step(action.cpu().numpy())

        video_writer.append_vector_obs(
            next_obs, next_done, camera_name="agentview_image"
        )
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_done = torch.Tensor(next_done).to(device)

        if next_done.sum() == args.num_envs:
            break
    
    envs.close()
    
    video_writer.save()
    if args.track:
        wandb.save(f"{args.video_folder}/video.mp4")
    
    with torch.no_grad():
        next_value = eval_policy.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values
    
    success_rate = next_done.sum() / args.num_envs
    cum_rewards = rewards.sum(dim=0)
    max_cum_reward = cum_rewards.max()
    avg_cum_reward = cum_rewards.mean()
    avg_cum_returns = returns.sum(dim=0).mean()

    writer.add_scalar("charts/validation_success_rate", success_rate, global_step)
    writer.add_scalar("charts/validation_max_reward", max_cum_reward, global_step)
    writer.add_scalar("charts/validation_avg_reward", avg_cum_reward, global_step)
    writer.add_scalar("charts/validation_avg_return", avg_cum_returns, global_step)


    print(f"success_rate: {success_rate}")
    print(f"max_reward: {max_cum_reward}")
    print(f"avg_reward: {avg_cum_reward}")
    print(f"avg_return: {avg_cum_returns}")



if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.libero_task_suite}_{args.libero_task_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # device setup
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    MODALITY_CONFIG['device'] = str(device)

    # env setup
    single_action_space = torch.zeros(7)
    envs, initial_states, task, task_emb = make_libero_envs(
        num_envs=args.num_envs, 
        task_suite_name=args.libero_task_suite, 
        task_id=args.libero_task_id, 
        horizon=args.num_steps+args.num_steps_wait,
        camera_dim=args.camera_dim
    )

    # student policy setup
    student_policy = LiberoAgent(
        task_emb=task_emb, 
        checkpoint_path=args.actor_checkpoint_path,
    ).to(device)
    optimizer = optim.Adam(student_policy.parameters(), lr=args.learning_rate, eps=1e-5)

    # teacher policy setup
    openvla_config = OpenVLAConfig()
    openvla_config.task_suite_name = args.libero_task_suite
    openvla_config.pretrained_checkpoint = args.openvla_checkpoint_path

    openvla_processor, openvla_model = get_openvla_model(openvla_config)
    openvla_model = openvla_model.eval().to(device)  # set to eval mode

    # ALGO Logic: Storage setup
    obs = np.empty((args.num_steps, args.num_envs), dtype=object)
    # obs = torch.zeros((args.num_steps, args.num_envs) + single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + single_action_space.shape).to(device)
    action_means = torch.zeros((args.num_steps, args.num_envs) + single_action_space.shape).to(device)
    teacher_actions = torch.zeros((args.num_steps, args.num_envs) + single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Start training
    global_step = 0
    start_time = time.time()
    print(f"Running for {args.num_iterations} iterations")

    for iteration in range(1, args.num_iterations + 1):
        print(f"iteration: {iteration}")
        iteration_start = time.time()
        
        # Prepare environment
        envs.reset()
        next_obs = envs.set_init_state(initial_states)
        next_done = torch.zeros(args.num_envs).to(device)

        # Wait for environment to stabilize
        for i in range(0, args.num_steps_wait):
            dummy_action = get_dummy_action(openvla_config)
            dummy_action = np.tile(dummy_action, (args.num_envs, 1)).astype(np.float32)
            next_obs, _, _, _ = envs.step(dummy_action)
        
        # Freeze actor if pre-training value function
        if iteration <= args.pretrain_value_iters:
            print("Freezing actor parameters. Only training value network for this iteration.")
            for param in student_policy.actor.parameters():
                param.requires_grad = False
        else:
            for param in student_policy.actor.parameters():
                param.requires_grad = True

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow


        for step in range(0, args.num_steps):
            print(f"train step: {step}")

            done_env_ids = np.nonzero(next_done.cpu().numpy())[0]
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, action_mean, logprob, _, value = student_policy.get_action_and_value(next_obs)
                values[step] = value.flatten()
                actions[step] = action
                action_means[step] = action_mean
                logprobs[step] = logprob
                
                for i in range(0, args.num_envs):
                    teacher_action = get_openvla_action(
                        cfg=openvla_config,
                        model=openvla_model,
                        obs=next_obs[i],
                        task_description=task.language,
                        processor=openvla_processor
                    )
                    teacher_actions[step][i] = torch.tensor(teacher_action).to(device)
                
                # action = teacher_actions[step]
            
            next_obs, reward, next_done, infos = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_done = torch.Tensor(next_done).to(device)

            if len(done_env_ids) > 0 and args.reset_envs:
                print(f"Resetting done envs: {done_env_ids}")
                next_obs[done_env_ids] = envs.reset(done_env_ids)
                next_done[done_env_ids] = torch.zeros(len(done_env_ids)).to(device)

        if iteration % args.eval_frequency == 0:
            perform_evaluation(student_policy, global_step, args)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = student_policy.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        # b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_obs = obs.reshape((-1,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + single_action_space.shape)
        b_action_means = action_means.reshape((-1,) + single_action_space.shape)
        b_teach_actions = teacher_actions.reshape((-1,) + single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, _, newlogprob, entropy, newvalue = student_policy.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                #############################################################################
                # Adding MSE(student_action_mean, teacher_action) to compute Refined Policy Distillation loss
                # Paper: https://arxiv.org/pdf/2503.05833
                #############################################################################
                teacher_loss = F.mse_loss(b_action_means[mb_inds], b_teach_actions[mb_inds]).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - (args.ent_coef * entropy_loss) + (v_loss * args.vf_coef) + (teacher_loss * args.teacher_coef)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(student_policy.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/teacher_loss", teacher_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        print(f"iteration {iteration} took {time.time() - iteration_start:.2f} seconds")

    envs.close()
    writer.close()
