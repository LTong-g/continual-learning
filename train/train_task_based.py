import torch
from torch import optim
from torch.utils.data import ConcatDataset
import numpy as np
import tqdm
import copy
from utils import get_data_loader
from models.cl.continual_learner import ContinualLearner


def train_cl(model, train_datasets, iters=2000, batch_size=32, baseline='none',
             loss_cbs=list(), eval_cbs=list(), context_cbs=list(), **kwargs):
    '''Train a model (with a "train_a_batch" method) on multiple contexts.

    [model]               <nn.Module> main model to optimize across all contexts
    [train_datasets]      <list> with for each context the training <DataSet>
    [iters]               <int>, # of optimization-steps (i.e., # of mini-batches) per context
    [batch_size]          <int>, # of samples per mini-batch
    [baseline]            <str>, 'joint': model trained once on data from all contexts
                                 'cummulative': model trained incrementally, always using data all contexts so far
    [*_cbs]               <list> of call-back functions to evaluate training-progress
    '''

    # Set model in training-mode
    model.train()

    # Use cuda?
    cuda = model._is_on_cuda()
    device = model._device()

    previous_model = None

    # Register starting parameter values (needed for SI)
    if isinstance(model, ContinualLearner) and model.importance_weighting == 'si':
        model.register_starting_param_values()

    # Are there different active classes per context?
    per_context = (model.scenario == "task" or (model.scenario == "class" and model.neg_samples == "current"))
    per_context_singlehead = per_context and (model.scenario == "task" and model.singlehead)

    # Loop over all contexts.
    for context, train_dataset in enumerate(train_datasets, 1):

        if baseline == 'joint':
            if context < len(train_datasets):
                continue
            else:
                baseline = "cummulative"

        if baseline == "cummulative" and (not per_context):
            train_dataset = ConcatDataset(train_datasets[:context])

        if isinstance(model, ContinualLearner) and model.importance_weighting == 'si':
            W, p_old = model.prepare_importance_estimates_dicts()

        # Find [active_classes]
        if model.scenario == "task":
            if not model.singlehead:
                active_classes = [list(
                    range(model.classes_per_context * i, model.classes_per_context * (i + 1))
                ) for i in range(context)]
            else:
                active_classes = None
        elif model.scenario == "domain":
            active_classes = None
        elif model.scenario == "class":
            if model.neg_samples == "all-so-far":
                active_classes = list(range(model.classes_per_context * context))
            elif model.neg_samples == "all":
                active_classes = None
            elif model.neg_samples == "current":
                active_classes = [list(
                    range(model.classes_per_context * i, model.classes_per_context * (i + 1))
                ) for i in range(context)]

        # Reset state of optimizer(s) for every context (if requested)
        if model.optim_type == "adam_reset":
            model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))

        iters_left = 1
        progress = tqdm.tqdm(range(1, iters + 1))

        for batch_index in range(1, iters + 1):
            iters_left -= 1
            if iters_left == 0:
                data_loader = iter(get_data_loader(train_dataset, batch_size, cuda=cuda, drop_last=True))
                iters_left = len(data_loader)

            x, y = next(data_loader)
            y = y - model.classes_per_context * (context - 1) if per_context and not per_context_singlehead else y
            x, y = x.to(device), y.to(device)

            x_ = y_ = scores_ = None
            if model.replay_mode == "current" and previous_model is not None:
                x_ = x
                with torch.no_grad():
                    scores_ = previous_model.classify(x_, no_prototypes=True)
                if model.scenario == "class" and model.neg_samples == "all-so-far":
                    scores_ = scores_[:, :(model.classes_per_context * (context - 1))]
                if model.replay_targets == "hard":
                    _, y_ = torch.max(scores_, dim=1)

            loss_dict = model.train_a_batch(
                x, y, x_=x_, y_=y_, scores_=scores_, rnt=0.5, active_classes=active_classes, context=context
            )

            if isinstance(model, ContinualLearner) and model.importance_weighting == 'si':
                model.update_importance_estimates(W, p_old)

            for loss_cb in loss_cbs:
                if loss_cb is not None:
                    loss_cb(progress, batch_index, loss_dict, context=context)
            for eval_cb in eval_cbs:
                if eval_cb is not None:
                    eval_cb(model, batch_index, context=context)

        progress.close()

        if isinstance(model, ContinualLearner) and model.importance_weighting == 'si' and model.weight_penalty:
            model.update_omega(W, model.epsilon)

        if isinstance(model, ContinualLearner) and model.importance_weighting == 'fisher' and model.weight_penalty:
            model.estimate_fisher(train_dataset, allowed_classes=active_classes if per_context else None)

        for context_cb in context_cbs:
            if context_cb is not None:
                context_cb(model, iters, context=context)

        if model.replay_mode == "current":
            previous_model = copy.deepcopy(model).eval()
