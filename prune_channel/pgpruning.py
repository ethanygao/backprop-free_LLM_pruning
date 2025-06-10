import math
import time

import torch
import torch.nn as nn
import transformers
import numpy as np
import tqdm
import gc

from torch.profiler import profile, record_function, ProfilerActivity

from modelutils import WrappedLlaMaLayer

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    """
    This class implements a learning rate scheduler that first linearly increases the learning rate from 0 to the base learning rate over a number of warmup steps, 
    then decreases it according to the cosine annealing schedule for the remaining steps until total_steps, 
    and finally keeps the learning rate at eta_min for any steps beyond total_steps.
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): Number of steps over which to increase the learning rate.
        total_steps (int): Total number of steps for the learning rate schedule.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        last_epoch (int, optional): The index of the last epoch. Default: -1.
    """
    def __init__(self, optimizer, warmup_steps, total_steps, eta_min=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        This method returns the learning rate for the current step.
        It first linearly increases the learning rate from 0 to the base learning rate over warmup_steps, 
        then decreases it according to the cosine annealing schedule for the remaining steps until total_steps, 
        and finally keeps the learning rate at eta_min for any steps beyond total_steps.
        """
        if self.last_epoch < self.warmup_steps:
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
        elif self.last_epoch <= self.total_steps:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]
        else:
            return [self.eta_min for _ in self.base_lrs]
        
def get_scores_params(model):
    # parameters = list(model.named_parameters())

    # score_params = [v for n, v in parameters if ("score" in n) and v.requires_grad]
    # weight_params = [v for n, v in parameters if ("score" not in n) and v.requires_grad]

    # return score_params, weight_params
    attn_score_params = []
    mlp_score_params = []
    for wrapped_layer in model.model.layers:
        if not isinstance(wrapped_layer, WrappedLlaMaLayer):
            continue
        # score_params.extend([wrapped_layer.attn_scores, wrapped_layer.mlp_scores])
        attn_score_params.append(wrapped_layer.attn_scores)
        mlp_score_params.append(wrapped_layer.mlp_scores)
    return attn_score_params, mlp_score_params


def get_optimizer(paras_to_opt, opt_type, lr, weight_decay):
    # parameters = list(model.named_parameters())
    # weight_params = [v for n, v in parameters if ("score" not in n) and v.requires_grad]
    # score_params = [v for n, v in parameters if ("score" in n) and v.requires_grad]
    
    if opt_type == 'adam':
        optimizer = torch.optim.Adam(
            paras_to_opt, 
            lr=lr,
            betas=(0.5, 0.999),
            weight_decay=weight_decay
        )
    elif opt_type == 'sgd':
        optimizer = torch.optim.SGD(
            paras_to_opt,
            lr=lr,
            # manually update weights should remove this, otherwise the results will not be equal.
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise NotImplementedError

    return optimizer

def get_scheduler(optimizer, scheduler_type, total_iters, warmup=0, start_iter=0, end_iter=0):
    # if warmup > 0.:
    #     assert scheduler_type == 'poly'

    if end_iter == 0:
        end_iter = total_iters

    if end_iter < total_iters:
        assert scheduler_type == 'step'

    if scheduler_type == 'poly':
        if warmup > 0.:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                        lambda step: min(1., float(step) / warmup) * (1 - float(step) / total_iters) ** 0.9,
                        last_epoch=-1)
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                        lambda step: (1 - float(step) / total_iters) ** 0.9,
                        last_epoch=-1)
    elif scheduler_type == 'cosine':
        # the second param is T_max, which means half of the period
        # so the lr is progressively decreasing in total_iters
        ## add warm up
        if warmup > 0:
            scheduler = WarmupCosineAnnealingLR(optimizer, warmup, total_iters)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters)  
    
    # < start_iter, large lr, start_iter - end_iter, middle lr, > end_iter, small lr
    elif scheduler_type == 'step':
        milestones = (np.array([start_iter, end_iter]) * total_iters).astype('int')
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
    elif scheduler_type == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
    elif scheduler_type == 'const':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1., last_epoch=-1)
    else:
        raise NotImplementedError

    return scheduler

class ChannelPGPruner:
    def __init__(self, subnet, tokenizer, total_iters, logger, train_data_name, prune_rate_start=1.0, prune_rate_target=0.1, 
                 prune_start_iter=None, prune_end_iter=None, criterion=nn.MSELoss(), device=torch.device("cuda:0"),
                 K=2, attn_score_lr=1e-3, mlp_score_lr=1e-3, ma_window_size=1, mode="train", **kwargs):
        
        self.subnet = subnet
        self.tokenizer = tokenizer
        
        self.criterion = criterion
        self.logger = logger
        self.writer = kwargs.get('summary_writer', None)

        self.prune_rate_start = prune_rate_start
        self.prune_rate_target = prune_rate_target
        # This constraint is added to ensure that the prune rate start and target are the same, dumb behavior but i have no idea. This code is shit now :(
        assert self.prune_rate_start == self.prune_rate_target, "Prune rate start and target should be the same."

        self.K = K

        # self.dataloader = dataloader
        self.total_iters = total_iters
        self.prune_start_iter = 0 if prune_start_iter is None else prune_start_iter
        self.prune_end_iter = total_iters if prune_end_iter is None else prune_end_iter

        self.step = 0

        warmup_length = kwargs.get('warmup_length', 0)

        lr_policy = kwargs.get('lr_policy', 'cosine')
        attn_score_params, mlp_score_params = get_scores_params(self.subnet)
        # score_params = attn_score_params + mlp_score_params
        # self.score_opt = get_optimizer(score_params, opt_type='adam', lr=score_lr, weight_decay=1e-4)
        # self.score_scheduler = get_scheduler(self.score_opt, scheduler_type='cosine', total_iters=total_iters, warmup=warmup_length)
        self.attn_score_opt = get_optimizer(attn_score_params, opt_type='adam', lr=attn_score_lr, weight_decay=1e-4)
        self.attn_score_scheduler = get_scheduler(self.attn_score_opt, scheduler_type=lr_policy, total_iters=total_iters, warmup=warmup_length)
        self.mlp_score_opt = get_optimizer(mlp_score_params, opt_type='adam', lr=mlp_score_lr, weight_decay=1e-4)
        self.mlp_score_scheduler = get_scheduler(self.mlp_score_opt, scheduler_type=lr_policy, total_iters=total_iters, warmup=warmup_length)
        
        self.set_mode(mode)
        # To improve the stability of the training
        self.baseline_delta = torch.tensor(0.0)     # 0.0
        self.window_size = ma_window_size   # moving average window size
        self.lamda = torch.tensor(1. / self.window_size)   # 1. / self.window_size
        
        self.penalty_lamda_init = kwargs.get('penalty_lamda_init', 1e-2)
        self.penalty_lamda_final = kwargs.get('penalty_lamda_final', 1e-2)
        self.logger.info("penalty_lamda_init: {}, penalty_lamda_final: {}".format(self.penalty_lamda_init, self.penalty_lamda_final))

        # debug flag
        self.DEBUG = kwargs.get('DEBUG', False)

        # global/local hard mask
        self.global_hard_mask = kwargs.get('global_hard_mask', True)

        # whether cancel moving average baseline
        self.cancel_ma_baseline = kwargs.get('cancel_ma_baseline', False)
        
        ## set train dataset (task) name to determine how to calculate the loss
        self.train_data_name = train_data_name

    def get_loss(self, input_batch):
        # TODO: this is a temporary solution. I have no better idea
        if self.train_data_name in ['wikitext2', 'c4']:
            return self.subnet(**input_batch).loss
        else:
            raise NotImplementedError
        # return self.subnet(**input_batch).loss

    def set_mode(self, mode):
        self.mode = mode
        if 'train' in mode:
            self.subnet.train()
            self.set_model_mask_type('soft_mask')
        elif 'test' in mode:
            self.subnet.eval()
        else:
            raise NotImplementedError
        
    def set_model_mask_type(self, mask_type):
        if mask_type == 'hard_mask':
            prune_rate = self.get_current_prune_rate()
            if self.global_hard_mask:
                attn_thresh, mlp_thresh = self.get_hard_prune_thresh(prune_rate)
                for wrapped_layer in self.subnet.model.layers:
                    if not isinstance(wrapped_layer, WrappedLlaMaLayer):
                        continue
                    wrapped_layer.set_mask(mask_type, attn_thresh, mlp_thresh)
            else:
                for wrapped_layer in self.subnet.model.layers:  # in a layer-wise manner
                    if not isinstance(wrapped_layer, WrappedLlaMaLayer):
                        continue
                    attn_scores = wrapped_layer.attn_scores.flatten().cpu()
                    mlp_scores = wrapped_layer.mlp_scores.flatten().cpu()
                    # get attn & mlp threshold
                    attn_thresh = torch.sort(attn_scores.cuda())[0][int(attn_scores.numel() * (1 - prune_rate))].cpu().item()
                    mlp_thresh = torch.sort(mlp_scores.cuda())[0][int(mlp_scores.numel() * (1 - prune_rate))].cpu().item()
                    # set mask
                    wrapped_layer.set_mask(mask_type, attn_thresh, mlp_thresh)
        else:
            for wrapped_layer in self.subnet.model.layers:
                if not isinstance(wrapped_layer, WrappedLlaMaLayer):
                    continue
                wrapped_layer.set_mask(mask_type)

    def set_score_no_grad(self):
        # self.subnet.scores.requires_grad = False
        # self.subnet.scores.grad = None
        for wrapped_layer in self.subnet.model.layers:
            if not isinstance(wrapped_layer, WrappedLlaMaLayer):
                continue
            wrapped_layer.attn_scores.requires_grad = False
            wrapped_layer.mlp_scores.requires_grad = False
            wrapped_layer.attn_scores.grad = None
            wrapped_layer.mlp_scores.grad = None
            
    def set_index(self, j):
        for wrapped_layer in self.subnet.model.layers:
            if not isinstance(wrapped_layer, WrappedLlaMaLayer):
                continue
            wrapped_layer.set_index(j)
    
    def clear_model_mask(self):
        for wrapped_layer in self.subnet.model.layers:
            if not isinstance(wrapped_layer, WrappedLlaMaLayer):
                continue
            wrapped_layer.clear()

    def get_current_prune_rate(self):
        if self.prune_rate_start == self.prune_rate_target:
            prune_rate = self.prune_rate_target
        else:
            if self.step < self.prune_start_iter:
                prune_rate = self.prune_rate_start
            elif self.step < self.prune_end_iter:
                prune_rate = self.prune_rate_target + (self.prune_rate_start - self.prune_rate_target) \
                         * (1 - (self.step - self.prune_start_iter) / (self.prune_end_iter - self.prune_start_iter)) ** 3
            else:
                prune_rate = self.prune_rate_target

        return prune_rate
    
    def get_hard_prune_thresh(self, prune_rate):
        """
        given the prune rate, get the threshold of the scores
        This is only used for global threshold (global hard mask)
        """
        all_attn_scores = []
        all_mlp_scores = []
        for wrapped_layer in self.subnet.model.layers:
            if not isinstance(wrapped_layer, WrappedLlaMaLayer):
                continue
            all_attn_scores.append(wrapped_layer.attn_scores.flatten().cpu())
            all_mlp_scores.append(wrapped_layer.mlp_scores.flatten().cpu())

        combined_attn_scores = torch.cat(all_attn_scores)
        combined_mlp_scores = torch.cat(all_mlp_scores)

        attn_thresh = torch.sort(combined_attn_scores.cuda())[0][int(combined_attn_scores.numel() * (1 - prune_rate))].cpu()
        mlp_thresh = torch.sort(combined_mlp_scores.cuda())[0][int(combined_mlp_scores.numel() * (1 - prune_rate))].cpu()

        attn_thresh, mlp_thresh = attn_thresh.item(), mlp_thresh.item()
        return attn_thresh, mlp_thresh
    
    def solve_v_total(self, prune_rate):
        """
        solve the total v, which is the sum of the scores of all layers
        invoked in `constrainScoreByWhole`
        """
        all_attn_scores = []
        all_mlp_scores = []
        for wrapped_layer in self.subnet.model.layers:
            if not isinstance(wrapped_layer, WrappedLlaMaLayer):
                continue
            all_attn_scores.append(wrapped_layer.attn_scores)
            all_mlp_scores.append(wrapped_layer.mlp_scores)

        all_attn_scores = torch.cat(all_attn_scores)
        all_mlp_scores = torch.cat(all_mlp_scores)

        def solve_v(scores, prune_rate):
            k = scores.numel() * prune_rate
            a, b = 0, scores.max()

            def f(v):
                s = (scores - v).clamp(0, 1).sum().cpu()
                return s - k

            if f(0) < 0:
                return 0

            itr = 0
            while True:
                itr += 1
                v = (a + b) / 2
                obj = f(v)
                if abs(obj) < 1e-3 or itr > 20:
                    break
                if obj < 0:
                    b = v
                else:
                    a = v
            return max(0, v)

        return solve_v(all_attn_scores, prune_rate), solve_v(all_mlp_scores, prune_rate)
    
    @torch.no_grad()
    def constrainScoreByWhole(self, prune_rate):
        v_total_attn, v_total_mlp = self.solve_v_total(prune_rate)

        for wrapped_layer in self.subnet.model.layers:
            if not isinstance(wrapped_layer, WrappedLlaMaLayer):
                continue
            wrapped_layer.attn_scores.sub_(v_total_attn).clamp_(0, 1)
            wrapped_layer.mlp_scores.sub_(v_total_mlp).clamp_(0, 1)

    @torch.no_grad()
    def constrainScoreByLayer(self, prune_rate):
        def solve_v(scores, prune_rate):
            k = scores.numel() * prune_rate
            a, b = 0, scores.max()

            def f(v):
                s = (scores - v).clamp(0, 1).sum().cpu()
                return s - k

            if f(0) < 0:
                return 0

            itr = 0
            while True:
                itr += 1
                v = (a + b) / 2
                obj = f(v)
                if abs(obj) < 1e-3 or itr > 20:
                    break
                if obj < 0:
                    b = v
                else:
                    a = v
            return max(0, v)

        for wrapped_layer in self.subnet.model.layers:
            if not isinstance(wrapped_layer, WrappedLlaMaLayer):
                continue
            attn_scores = wrapped_layer.attn_scores
            mlp_scores = wrapped_layer.mlp_scores
            attn_scores.sub_(solve_v(attn_scores, prune_rate)).clamp_(0, 1)
            mlp_scores.sub_(solve_v(mlp_scores, prune_rate)).clamp_(0, 1)

    def log_score_state(self):
        cur_prune_rate = self.get_current_prune_rate()

        all_attn_scores = []
        all_mlp_scores = []
        layer_idx = 0
        pruned_layer_idxs = []
        for wrapped_layer in self.subnet.model.layers:
            if not isinstance(wrapped_layer, WrappedLlaMaLayer):
                # self.logger.info(f"Layer {layer_idx} is not WrappedLlaMaLayer, skip logging")
                layer_idx += 1
                continue
            all_attn_scores.append(wrapped_layer.attn_scores)
            all_mlp_scores.append(wrapped_layer.mlp_scores)
            pruned_layer_idxs.append(layer_idx)
            layer_idx += 1
        all_attn_scores = torch.cat(all_attn_scores).cpu().numpy().flatten()
        all_mlp_scores = torch.cat(all_mlp_scores).cpu().numpy().flatten()

        self.logger.info(f"Pruned Layer indices: {pruned_layer_idxs}")
        self.logger.info("Attentio Head")
        self.logger.info('Mean: {:.3f}, \t Median: {:.3f}, \t Min: {:.3f}, \t Max: {:.3f}, \t 0 percentage: {:.3f}, \t 1 percentage: {:.3f}'.format(
            np.mean(all_attn_scores), np.median(all_attn_scores), np.min(all_attn_scores), np.max(all_attn_scores), 
            np.mean(all_attn_scores == 0), np.mean(all_attn_scores == 1)))
        self.logger.info("MLP")
        self.logger.info('Mean: {:.3f}, \t Median: {:.3f}, \t Min: {:.3f}, \t Max: {:.3f}, \t 0 percentage: {:.3f}, \t 1 percentage: {:.3f}'.format(
            np.mean(all_mlp_scores), np.median(all_mlp_scores), np.min(all_mlp_scores), np.max(all_mlp_scores), 
            np.mean(all_mlp_scores == 0), np.mean(all_mlp_scores == 1)))
        
    @torch.no_grad()      # Have to add this due to memory constraint. this should be removed when updating model weights with backprop.
    # def prune_pg(self, inp, target, causal_attention_mask):
    def prune_pg(self, input_batch):
        self.attn_score_opt.zero_grad()
        self.mlp_score_opt.zero_grad()
        
        use_cache = self.subnet.model.config.use_cache
        self.subnet.model.config.use_cache = False
        
        loss_list = []
        is_valid_sample = torch.zeros(self.K).bool()

        for j in range(self.K):
            self.set_index(j)
            """
            output = self.subnet(inp, causal_attention_mask)[0][:, :-1, :]
            # lm_out = self.subnet(inp, causal_attention_mask)
            # lm_logits = lm_out.logits
            # output = lm_logits[:, :-1, :].contiguous()
            output = output.reshape(-1, output.size(-1))

            target = target.view(-1)
            loss = self.criterion(output, target)
            """
            # loss = self.subnet(**input_batch).loss
            loss = self.get_loss(input_batch)
            # loss_list.append(loss)
            if torch.isnan(loss).any():
                self.logger.info(f"Step {self.step}, Sample {j} has NaN loss")
                is_valid_sample[j] = False
            else:
                loss_list.append(loss)
                is_valid_sample[j] = True
            torch.cuda.empty_cache()

        # if all samples have NaN loss, skip updating scores
        # TODO: bad code style. :( should be refactored.
        if not is_valid_sample.any():
            self.logger.info(f"Step {self.step}, All samples have NaN loss, skip updating scores")
            return

        loss_all = torch.mean(torch.stack(loss_list))

        if self.step == 0:
            self.baseline_delta = loss_all

        self.inplace_update_scores_by_pg_grad(loss_all, loss_list, is_valid_sample)

        torch.cuda.empty_cache()

        prune_rate = self.get_current_prune_rate()
        if self.global_hard_mask:
            self.constrainScoreByWhole(prune_rate)
        else:
            self.constrainScoreByLayer(prune_rate)
        torch.cuda.empty_cache()
        
        self.step += 1
        self.subnet.model.config.use_cache = use_cache

    def inplace_update_scores_by_pg_grad(self, loss_all, loss_list, is_valid_sample):
        self.baseline_delta = self.baseline_delta * (1. - self.lamda) + loss_all * self.lamda if not self.cancel_ma_baseline else 0.0
        penalty_lamda = 0 if self.penalty_lamda_init < 0.0 else (self.penalty_lamda_final - self.penalty_lamda_init) * self.step / self.total_iters + self.penalty_lamda_init     # adjustable

        if self.writer is not None:
            self.writer.add_scalar('loss/loss_all', loss_all, self.step)
            self.writer.add_scalar('loss/baseline_delta', self.baseline_delta, self.step)
            self.writer.add_scalar('loss/penalty_lamda', penalty_lamda, self.step)
            self.writer.add_scalar('learning_rate/attn_score_lr', self.attn_score_scheduler.get_last_lr()[0], self.step)
            self.writer.add_scalar('learning_rate/mlp_score_lr', self.mlp_score_scheduler.get_last_lr()[0], self.step)

        # Performing the policy gradient update
        for wrapped_layer in self.subnet.model.layers:
            if not isinstance(wrapped_layer, WrappedLlaMaLayer):
                continue
            for score_type in ['attn', 'mlp']:
                scores = getattr(wrapped_layer, f'{score_type}_scores')
                stored_masks = getattr(wrapped_layer, f'stored_{score_type}_mask')[is_valid_sample].float()
                last_lr = self.attn_score_scheduler.get_last_lr()[0] if score_type == 'attn' else self.mlp_score_scheduler.get_last_lr()[0]

                stored_masks = (stored_masks - scores)/torch.sqrt((scores+1e-7) * (1-scores+1e-7))
                # moving average baseline
                loss = torch.stack(loss_list).reshape(stored_masks.shape[0], *[1]*(stored_masks.ndim-1)) - self.baseline_delta

                # policy gradient
                grad = torch.sum(loss.to(stored_masks.device) * stored_masks, dim=0) / (self.K-1)

                # penalty gradient
                ## regualirzation function: x(1-x)
                # grad += penalty_lamda * (1-2*scores)
                ## regualirzation function: cross entropy
                grad += penalty_lamda * (torch.log(1-scores+1e-7) - torch.log(scores+1e-7))   

                # update scores
                # scores.sub_(self.score_scheduler.get_last_lr()[0] * grad)
                scores.sub_(last_lr * grad)


        # self.score_scheduler.step()
        self.attn_score_scheduler.step()
        self.mlp_score_scheduler.step()
        # if DEBUG and self.step % 10 == 0:
        if self.DEBUG:
            print("DEBUG: step={}, baseline_delta = {}, loss_list: {}, loss_mean: {}, reward: {}".format(self.step, self.baseline_delta, torch.stack(loss_list), loss_all, torch.stack(loss_list) - self.baseline_delta))
            # print("LayerScores: {}".format(self.subnet.scores))
            # print("scores grad: {}".format(grad_))

    def get_prune_info(self):
        prune_rate = self.get_current_prune_rate()
        global_attn_thresh, global_mlp_thresh = self.get_hard_prune_thresh(prune_rate)
        prune_info = {
            'end_step': self.step,
            'prune_rate': prune_rate,
        }
        layer_idx = 0
        for wrapped_layer in self.subnet.model.layers:
            if not isinstance(wrapped_layer, WrappedLlaMaLayer):
                self.logger.info(f"Layer {layer_idx} is not WrappedLlaMaLayer, skip pruning. Corresponding layer info is None.")
                layer_info = None
            else:
                if self.global_hard_mask:
                    attn_thresh, mlp_thresh = global_attn_thresh, global_mlp_thresh
                else:
                    attn_scores = wrapped_layer.attn_scores.flatten().cpu()
                    mlp_scores = wrapped_layer.mlp_scores.flatten().cpu()
                    attn_thresh = torch.sort(attn_scores.cuda())[0][int(attn_scores.numel() * (1 - prune_rate))].cpu().item()
                    mlp_thresh = torch.sort(mlp_scores.cuda())[0][int(mlp_scores.numel() * (1 - prune_rate))].cpu().item()
                hard_attn_mask, hard_mlp_mask = wrapped_layer.get_mask('hard_mask', attn_thresh, mlp_thresh) 
                layer_info = {
                    'attn_scores': wrapped_layer.attn_scores.cpu().detach(),
                    'mlp_scores': wrapped_layer.mlp_scores.cpu().detach(),
                    'init_attn_mask': wrapped_layer.init_attn_mask.cpu().detach(),
                    'init_mlp_mask': wrapped_layer.init_mlp_mask.cpu().detach(),
                    'hard_attn_mask': hard_attn_mask.cpu().detach(),
                    'hard_mlp_mask': hard_mlp_mask.cpu().detach(),
                }
            prune_info[f'layer_{layer_idx}'] = layer_info
            layer_idx += 1
        return prune_info
    
    def structure_prune(self):
        """
        model's layer should be WrappedLlaMaLayer.
        Prune llama model by hard mask by default.
        """
        prune_rate = self.get_current_prune_rate()
        global_attn_thresh, global_mlp_thresh = self.get_hard_prune_thresh(prune_rate)
        
        layer_num = len(self.subnet.model.layers)
        for i in range(layer_num):
            wrapped_layer = self.subnet.model.layers[i]
            num_key_value_groups = wrapped_layer.layer.self_attn.num_key_value_groups if isinstance(wrapped_layer, WrappedLlaMaLayer) else wrapped_layer.self_attn.num_key_value_groups
            head_dim = wrapped_layer.layer.self_attn.head_dim if isinstance(wrapped_layer, WrappedLlaMaLayer) else wrapped_layer.self_attn.head_dim
            # assert isinstance(wrapped_layer, WrappedLlaMaLayer), "model should be an instance of LlamaForCausalLM"
            if not isinstance(wrapped_layer, WrappedLlaMaLayer):
                self.logger.info(f"Layer {i} is not WrappedLlaMaLayer, skip pruning")
                continue
            
            if self.global_hard_mask:
                attn_thresh, mlp_thresh = global_attn_thresh, global_mlp_thresh
            else:
                attn_scores = wrapped_layer.attn_scores.flatten().cpu()
                mlp_scores = wrapped_layer.mlp_scores.flatten().cpu()
                attn_thresh = torch.sort(attn_scores.cuda())[0][int(attn_scores.numel() * (1 - prune_rate))].cpu().item()
                mlp_thresh = torch.sort(mlp_scores.cuda())[0][int(mlp_scores.numel() * (1 - prune_rate))].cpu().item()
            
            ##### only attn head pruning implmented #####
            attn_mask, mlp_mask = wrapped_layer.get_mask('hard_mask', attn_thresh, mlp_thresh)
            retain_kv_heads = torch.count_nonzero(attn_mask)
            retain_q_heads = torch.count_nonzero(attn_mask) * num_key_value_groups
            q_attn_mask = attn_mask.repeat_interleave(num_key_value_groups).repeat_interleave(head_dim)
            kv_attn_mask = attn_mask.repeat_interleave(head_dim)
            
            layer = wrapped_layer.layer
            """ Attention Weight Pruning """
            # Prune the query, key and value projection weights
            # We reduce the size of the weights based on the attention mask (output dim/channel/head is pruned)
            layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[torch.where(q_attn_mask)[0]]
            layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[torch.where(kv_attn_mask)[0]]
            layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data[torch.where(kv_attn_mask)[0]]

            # Update output dimensions of q, k, v projections based on remaining heads
            layer.self_attn.q_proj.out_features = q_attn_mask.sum().item()
            layer.self_attn.k_proj.out_features = kv_attn_mask.sum().item()
            layer.self_attn.v_proj.out_features = kv_attn_mask.sum().item()

            output_weight = layer.self_attn.o_proj.weight.data

            # Prune the output projection weight (its input channel)
            output_weight = layer.self_attn.o_proj.weight.data[:, torch.where(q_attn_mask)[0]]
            # Update layer configurations for the new output shape after pruning
            layer.self_attn.num_heads = retain_q_heads
            layer.self_attn.hidden_size = retain_q_heads * head_dim
            layer.self_attn.num_key_value_heads = retain_kv_heads
            layer.self_attn.num_key_value_groups = layer.self_attn.num_heads // layer.self_attn.num_key_value_heads

            # Update the input dimension of the output projection based on the attn mask
            layer.self_attn.o_proj.in_features = q_attn_mask.sum().item()

            # Assign the pruned weights
            layer.self_attn.o_proj.weight.data = output_weight
            
            """ MLP Weight Pruning """
            # Prune the up and gate projection weights
            layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[torch.where(mlp_mask)[0]]
            layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[torch.where(mlp_mask)[0]]

            # Update output dimensions of up and gate projections based on the mlp mask
            layer.mlp.up_proj.out_features = mlp_mask.sum().item()
            layer.mlp.gate_proj.out_features = mlp_mask.sum().item()
            
            output_weight = layer.mlp.down_proj.weight.data
            layer.mlp.intermediate_size = mlp_mask.sum().item()

            # Prune the down projection weight
            output_weight = layer.mlp.down_proj.weight.data[:, torch.where(mlp_mask)[0]]

            # Update the input dimension of the down projection based on the mlp mask
            layer.mlp.down_proj.in_features = mlp_mask.sum().item()

            # Assign the pruned weights
            layer.mlp.down_proj.weight.data = output_weight
            
            """ Replace the layer with the pruned layer """
            self.subnet.model.layers[i] = layer

            # Explicitly empty the CUDA cache to clean up some memory
            # del wrapped_layer.attn_scores, wrapped_layer.mlp_scores
            # del wrapped_layer
            gc.collect()
        
        # Explicitly empty the CUDA cache to clean up some memory
        torch.cuda.empty_cache()
        return self.subnet
    
    def check_sparsity(self, mask_type):
        """
        This function is used to check the pruned but not compressed model's sparsity

        Args: 
            mask_type: assign which kind of mask is used to prune the model
        
        Returns:
            Ratio of the count of unpruned weights to total parameters in the model.
        """
        # set mask for model pruning
        self.set_model_mask_type(mask_type)

        wrapped_layers = self.subnet.model.layers
        intermediate_size = self.subnet.config.intermediate_size
        hidden_size = self.subnet.config.hidden_size

        count = 0
        total_params = 0
        # for wrapped_layer in self.subnet.model.layers:
        for i in range(len(wrapped_layers)):
            wrapped_layer = wrapped_layers[i]
            num_key_value_groups = wrapped_layer.layer.self_attn.num_key_value_groups if isinstance(wrapped_layer, WrappedLlaMaLayer) else wrapped_layer.self_attn.num_key_value_groups
            head_dim = wrapped_layer.layer.self_attn.head_dim if isinstance(wrapped_layer, WrappedLlaMaLayer) else wrapped_layer.self_attn.head_dim

            sub_count = 0
            sub_params = 0
            if isinstance(wrapped_layer, WrappedLlaMaLayer):
                attn_mask, mlp_mask = wrapped_layer.mask_func()
                prn_head_num = attn_mask.sum().cpu().item(); prn_intermediate_size = mlp_mask.sum().cpu().item()
                org_head_num = attn_mask.shape[0]; org_intermediate_size = intermediate_size
            else:   # This is an original LlamaDecoderLayer
                attn_mask = torch.ones(wrapped_layer.self_attn.num_key_value_heads, dtype=torch.bool)    # hidden_size can also be used
                mlp_mask = torch.ones(wrapped_layer.mlp.intermediate_size, dtype=torch.bool) # intermediate_size can also be used
                prn_head_num = org_head_num = wrapped_layer.self_attn.num_key_value_heads
                prn_intermediate_size = org_intermediate_size = wrapped_layer.mlp.intermediate_size
            # Attention Module
            q_attn_mask = attn_mask.repeat_interleave(num_key_value_groups).repeat_interleave(head_dim)
            kv_attn_mask = attn_mask.repeat_interleave(head_dim)
            q_out_dim = o_in_dim = q_attn_mask.sum().cpu().item()
            kv_out_dim = kv_attn_mask.sum().cpu().item()
            qkv_numel = hidden_size * q_out_dim + hidden_size * kv_out_dim * 2
            o_numel = o_in_dim * hidden_size
            org_qkv_numel = hidden_size * hidden_size + 2 * hidden_size * hidden_size / num_key_value_groups
            org_o_numel = hidden_size * hidden_size
            # MLP Module
            up_out_dim = gate_out_dim = down_in_dim = mlp_mask.sum().cpu().item()
            up_numel = hidden_size * up_out_dim
            gate_numel = hidden_size * gate_out_dim
            down_numel = hidden_size * down_in_dim
            org_up_numel = hidden_size * intermediate_size
            org_gate_numel = hidden_size * intermediate_size
            org_down_numel = hidden_size * intermediate_size

            # sum
            sub_count += qkv_numel + o_numel + up_numel + gate_numel + down_numel
            count += qkv_numel + o_numel + up_numel + gate_numel + down_numel
            sub_params += org_qkv_numel + org_o_numel + org_up_numel + org_gate_numel + org_down_numel
            total_params += org_qkv_numel + org_o_numel + org_up_numel + org_gate_numel + org_down_numel

            self.logger.info(f"layer {i} prune rate in \'{mask_type}\' {float(sub_count)/sub_params:.6f}, Head: {prn_head_num}/{org_head_num}, Intermediate: {prn_intermediate_size}/{org_intermediate_size}")
        
        total_prune_rate = float(count) / total_params
        self.logger.info(f"Total prune rate in \'{mask_type}\' {total_prune_rate}")

        # clear mask
        self.clear_model_mask()

        return total_prune_rate
        