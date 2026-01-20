import abc
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F
from utils import get_data_loader


class ContinualLearner(nn.Module, metaclass=abc.ABCMeta):
    '''Abstract module to add continual learning capabilities to a classifier (e.g., param regularization, replay).'''

    def __init__(self):
        super().__init__()

        # List with the methods to create generators that return the parameters on which to apply param regularization
        self.param_list = [self.named_parameters]

        # Optimizer (and whether it needs to be reset)
        self.optimizer = None
        self.optim_type = "adam"
        self.optim_list = []

        # Scenario, singlehead & negative samples
        self.scenario = 'task'
        self.classes_per_context = 2
        self.singlehead = False
        self.neg_samples = 'all'

        # LwF / Replay
        self.replay_mode = "none"
        self.replay_targets = "hard"
        self.KD_temp = 2.
        self.use_replay = "normal"
        self.eps_agem = 0.
        self.lwf_weighting = False

        # Parameter-regularization
        self.weight_penalty = False
        self.reg_strength = 0
        self.importance_weighting = 'fisher'  # 'fisher' (EWC) or 'si'
        self.fisher_n = None
        self.fisher_labels = "all"
        self.fisher_batch = 1
        self.context_count = 0
        self.data_size = None
        self.epsilon = 0.1
        self.offline = False
        self.gamma = 1.
        self.alpha = 1e-10

        # XdG (kept for compatibility with existing classifier methods)
        self.mask_dict = None
        self.excit_buffer_list = []

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    #----------------- XdG-specifc functions -----------------#

    def apply_XdGmask(self, context):
        '''Apply context-specific mask, by setting activity of pre-selected subset of nodes to zero.

        [context]   <int>, starting from 1'''

        assert self.mask_dict is not None
        torchType = next(self.parameters()).detach()

        for i, excit_buffer in enumerate(self.excit_buffer_list):
            gating_mask = np.repeat(1., len(excit_buffer))
            gating_mask[self.mask_dict[context][i]] = 0.
            excit_buffer.set_(torchType.new(gating_mask))

    def reset_XdGmask(self):
        '''Remove context-specific mask, by setting all "excit-buffers" to 1.'''
        torchType = next(self.parameters()).detach()
        for excit_buffer in self.excit_buffer_list:
            gating_mask = np.repeat(1., len(excit_buffer))
            excit_buffer.set_(torchType.new(gating_mask))

    #------------- "Synaptic Intelligence"-specifc functions -------------#

    def register_starting_param_values(self):
        '''Register the starting parameter values into the model as a buffer.'''
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    self.register_buffer('{}_SI_prev_context'.format(n), p.detach().clone())

    def prepare_importance_estimates_dicts(self):
        '''Prepare <dicts> to store running importance estimates and param-values before update.'''
        W = {}
        p_old = {}
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    W[n] = p.data.clone().zero_()
                    p_old[n] = p.data.clone()
        return W, p_old

    def update_importance_estimates(self, W, p_old):
        '''Update the running parameter importance estimates in W.'''
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        W[n].add_(-p.grad * (p.detach() - p_old[n]))
                    p_old[n] = p.detach().clone()

    def update_omega(self, W, epsilon):
        '''After completing training on a context, update the per-parameter regularization strength.'''
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    p_prev = getattr(self, '{}_SI_prev_context'.format(n))
                    p_current = p.detach().clone()
                    p_change = p_current - p_prev
                    omega_add = W[n] / (p_change ** 2 + epsilon)
                    try:
                        omega = getattr(self, '{}_SI_omega'.format(n))
                    except AttributeError:
                        omega = p.detach().clone().zero_()
                    omega_new = omega + omega_add
                    self.register_buffer('{}_SI_prev_context'.format(n), p_current)
                    self.register_buffer('{}_SI_omega'.format(n), omega_new)

    def surrogate_loss(self):
        '''Calculate SI's surrogate loss.'''
        try:
            losses = []
            for gen_params in self.param_list:
                for n, p in gen_params():
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        prev_values = getattr(self, '{}_SI_prev_context'.format(n))
                        omega = getattr(self, '{}_SI_omega'.format(n))
                        losses.append((omega * (p - prev_values) ** 2).sum())
            return sum(losses)
        except AttributeError:
            return torch.tensor(0., device=self._device())

    #----------------- EWC-specifc functions -----------------#

    def initialize_fisher(self):
        '''Initialize diagonal fisher matrix with the prior precision.'''
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    self.register_buffer('{}_EWC_prev_context'.format(n), p.detach().clone() * 0)
                    self.register_buffer('{}_EWC_estimated_fisher'.format(n), torch.ones(p.shape) / self.data_size)

    def estimate_fisher(self, dataset, allowed_classes=None):
        '''After completing training on a context, estimate diagonal of Fisher Information matrix.'''

        est_fisher_info = {}
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    est_fisher_info[n] = p.detach().clone().zero_()

        mode = self.training
        self.eval()

        data_loader = get_data_loader(dataset, batch_size=1 if self.fisher_batch is None else self.fisher_batch,
                                      cuda=self._is_on_cuda())

        for index, (x, y) in enumerate(data_loader):
            if self.fisher_n is not None:
                if index > self.fisher_n:
                    break
            x = x.to(self._device())
            output = self(x) if allowed_classes is None else self(x)[:, allowed_classes]
            if self.fisher_labels == 'all':
                with torch.no_grad():
                    label_weights = F.softmax(output, dim=1)
                for label_index in range(output.shape[1]):
                    label = torch.LongTensor([label_index]).to(self._device())
                    negloglikelihood = F.cross_entropy(output, label)
                    self.zero_grad()
                    negloglikelihood.backward(retain_graph=True if (label_index + 1) < output.shape[1] else False)
                    for gen_params in self.param_list:
                        for n, p in gen_params():
                            if p.requires_grad:
                                n = n.replace('.', '__')
                                if p.grad is not None:
                                    est_fisher_info[n] += label_weights[:, label_index].mean() * p.grad.data.clone().pow(2)
            else:
                if self.fisher_labels == 'sample':
                    with torch.no_grad():
                        label = Categorical(F.softmax(output, dim=1)).sample()
                elif self.fisher_labels == 'pred':
                    label = output.max(1)[1]
                elif self.fisher_labels == 'true':
                    label = y.to(self._device())
                else:
                    raise ValueError("Invalid fisher_labels option: {}".format(self.fisher_labels))
                negloglikelihood = F.cross_entropy(output, label)
                self.zero_grad()
                negloglikelihood.backward()
                for gen_params in self.param_list:
                    for n, p in gen_params():
                        if p.requires_grad:
                            n = n.replace('.', '__')
                            if p.grad is not None:
                                est_fisher_info[n] += p.grad.data.clone().pow(2)

        for n in est_fisher_info:
            est_fisher_info[n] /= (index + 1)

        self.train(mode=mode)

        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    self.register_buffer('{}_EWC_prev_context'.format(n), p.detach().clone())
                    if self.offline:
                        self.register_buffer('{}_EWC_estimated_fisher'.format(n), est_fisher_info[n])
                    else:
                        try:
                            fisher_old = getattr(self, '{}_EWC_estimated_fisher'.format(n))
                        except AttributeError:
                            fisher_old = p.detach().clone().zero_()
                        self.register_buffer('{}_EWC_estimated_fisher'.format(n),
                                             (self.gamma * fisher_old + est_fisher_info[n]))

    def ewc_loss(self):
        '''Calculate EWC's quadratic penalty.'''
        try:
            losses = []
            for gen_params in self.param_list:
                for n, p in gen_params():
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        prev_values = getattr(self, '{}_EWC_prev_context'.format(n))
                        fisher = getattr(self, '{}_EWC_estimated_fisher'.format(n))
                        losses.append((fisher * (p - prev_values) ** 2).sum())
            return sum(losses)
        except AttributeError:
            return torch.tensor(0., device=self._device())
