import torch
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import ConcatDataset
import numpy as np
import tqdm
import copy
from utils import get_data_loader,checkattr
from data.manipulate import SubDataset, MemorySetDataset
from models.cl.continual_learner import ContinualLearner


def train_cl(model, train_datasets, iters=2000, batch_size=32, baseline='none',
             loss_cbs=list(), eval_cbs=list(), sample_cbs=list(), context_cbs=list(),
             generator=None, gen_iters=0, gen_loss_cbs=list(), **kwargs):
    '''Train a model (with a "train_a_batch" method) on multiple contexts.

    [model]               <nn.Module> main model to optimize across all contexts
    [train_datasets]      <list> with for each context the training <DataSet>
    [iters]               <int>, # of optimization-steps (i.e., # of mini-batches) per context
    [batch_size]          <int>, # of samples per mini-batch
    [baseline]            <str>, 'joint': model trained once on data from all contexts
                                 'cummulative': model trained incrementally, always using data all contexts so far
    [generator]           None or <nn.Module>, if separate generative model is trained (for [gen_iters] per context)
    [*_cbs]               <list> of call-back functions to evaluate training-progress
    '''

    # Set model in training-mode
    model.train()

    # Use cuda?
    cuda = model._is_on_cuda()
    device = model._device()

    # Initiate possible sources for replay (no replay for 1st context)
    ReplayStoredData = ReplayGeneratedData = ReplayCurrentData = False
    previous_model = None

    # Register starting parameter values (needed for SI)
    if isinstance(model, ContinualLearner) and model.importance_weighting=='si':
        model.register_starting_param_values()

    # Loop over all contexts.
    for context, train_dataset in enumerate(train_datasets, 1):

        # If using the "joint" baseline, skip to last context, as model is only be trained once on data of all contexts
        if baseline=='joint':
            if context<len(train_datasets):
                continue
            else:
                baseline = "cummulative"

        # If using the "cummulative" (or "joint") baseline, create a large training dataset of all contexts so far
        if baseline=="cummulative":
            train_dataset = ConcatDataset(train_datasets[:context])

        # Add memory buffer (if available) to current dataset (if requested)
        if checkattr(model, 'add_buffer') and context>1:
            memory_dataset = MemorySetDataset(model.memory_sets)
            training_dataset = ConcatDataset([train_dataset, memory_dataset])
        else:
            training_dataset = train_dataset

        # Prepare <dicts> to store running importance estimates and param-values before update (needed for SI)
        if isinstance(model, ContinualLearner) and model.importance_weighting=='si':
            W, p_old = model.prepare_importance_estimates_dicts()

        # Find [active_classes]
        # --> one <list> with active classes of all contexts so far
        active_classes = list(range(model.classes_per_context * context))

        # Reset state of optimizer(s) for every context (if requested)
        if model.optim_type=="adam_reset":
            model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))
        if (generator is not None) and generator.optim_type=="adam_reset":
            generator.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))

        # Initialize # iters left on current data-loader(s)
        iters_left = iters_left_previous = 1

        # Define tqdm progress bar(s)
        progress = tqdm.tqdm(range(1, iters+1))
        if generator is not None:
            progress_gen = tqdm.tqdm(range(1, gen_iters+1))

        # Loop over all iterations
        iters_to_use = iters if (generator is None) else max(iters, gen_iters)
        for batch_index in range(1, iters_to_use+1):

            # Update # iters left on current data-loader(s) and, if needed, create new one(s)
            iters_left -= 1
            if iters_left==0:
                data_loader = iter(get_data_loader(training_dataset, batch_size, cuda=cuda, drop_last=True))
                # NOTE:  [train_dataset]  is training-set of current context
                #      [training_dataset] is training-set of current context with stored samples added (if requested)
                iters_left = len(data_loader)
            if ReplayStoredData:
                iters_left_previous -= 1
                if iters_left_previous==0:
                    batch_size_to_use = min(batch_size, len(ConcatDataset(previous_datasets)))
                    data_loader_previous = iter(get_data_loader(ConcatDataset(previous_datasets),
                                                                batch_size_to_use, cuda=cuda, drop_last=True))
                    iters_left_previous = len(data_loader_previous)


            # -----------------Collect data------------------#

            #####-----CURRENT BATCH-----#####
            x, y = next(data_loader)                             #--> sample training data of current context
            # --> adjust the y-targets to the 'active range'
            x, y = x.to(device), y.to(device)                    #--> transfer them to correct device
            # If --bce & --bce-distill, calculate scores for past classes of current batch with previous model
            binary_distillation = hasattr(model, "binaryCE") and model.binaryCE and model.binaryCE_distill
            if binary_distillation and (previous_model is not None):
                with torch.no_grad():
                    scores = previous_model.classify(
                        x, no_prototypes=True
                    )[:, :(model.classes_per_context * (context - 1))]
            else:
                scores = None


            #####-----REPLAYED BATCH-----#####
            if not ReplayStoredData and not ReplayGeneratedData and not ReplayCurrentData:
                x_ = y_ = scores_ = None   #-> if no replay

            ##-->> Replay of stored data <<--##
            if ReplayStoredData:
                scores_ = None
                # Sample replayed training data, move to correct device
                x_, y_ = next(data_loader_previous)
                x_ = x_.to(device)
                y_ = y_.to(device) if (model.replay_targets=="hard") else None
                # If required, get target scores (i.e, [scores_])         -- using previous model, with no_grad()
                if (model.replay_targets=="soft"):
                    with torch.no_grad():
                        scores_ = previous_model.classify(x_, no_prototypes=True)
                    scores_ = scores_[:, :(model.classes_per_context*(context-1))]
                    #-> if [scores_] is not same length as [x_], zero probs are added in [loss_fn_kd]-function

            ##-->> Generative / Current Replay <<--##

            #---INPUTS---#
            if ReplayCurrentData:
                x_ = x  #--> use current context inputs

            if ReplayGeneratedData:
                # -which classes are allowed to be generated? (relevant if conditional generator / decoder-gates)
                allowed_classes = list(range(model.classes_per_context*(context-1)))
                # -generate inputs representative of previous contexts
                x_temp_ = previous_generator.sample(batch_size, allowed_classes=allowed_classes, only_x=False)
                x_ = x_temp_[0] if type(x_temp_)==tuple else x_temp_

            #---OUTPUTS---#
            if ReplayGeneratedData or ReplayCurrentData:
                # Get target scores and labels (i.e., [scores_] / [y_]) -- using previous model, with no_grad()
                # -if replay does not need to be evaluated separately for each context
                with torch.no_grad():
                    scores_ = previous_model.classify(x_, no_prototypes=True)
                scores_ = scores_[:, :(model.classes_per_context * (context - 1))]
                # -> if [scores_] is not same length as [x_], zero probs are added in [loss_fn_kd]-function
                # -also get the 'hard target'
                _, y_ = torch.max(scores_, dim=1)


                # Only keep predicted y/scores if required (as otherwise unnecessary computations will be done)
                y_ = y_ if (model.replay_targets == "hard") else None
                scores_ = scores_ if (model.replay_targets == "soft") else None


            #---> Train MAIN MODEL
            if batch_index <= iters:

                # Train the main model with this batch
                loss_dict = model.train_a_batch(x, y, x_=x_, y_=y_, scores=scores, scores_=scores_, rnt = 1./context,
                                                active_classes=active_classes, context=context)

                # Update running parameter importance estimates in W (needed for SI)
                if isinstance(model, ContinualLearner) and model.importance_weighting=='si':
                    model.update_importance_estimates(W, p_old)

                # Fire callbacks (for visualization of training-progress / evaluating performance after each context)
                for loss_cb in loss_cbs:
                    if loss_cb is not None:
                        loss_cb(progress, batch_index, loss_dict, context=context)
                for eval_cb in eval_cbs:
                    if eval_cb is not None:
                        eval_cb(model, batch_index, context=context)
                if model.label == "VAE":
                    for sample_cb in sample_cbs:
                        if sample_cb is not None:
                            sample_cb(model, batch_index, context=context)


            #---> Train GENERATOR
            if generator is not None and batch_index <= gen_iters:

                # Train the generator with this batch
                loss_dict = generator.train_a_batch(x, x_=x_, rnt=1./context)

                # Fire callbacks on each iteration
                for loss_cb in gen_loss_cbs:
                    if loss_cb is not None:
                        loss_cb(progress_gen, batch_index, loss_dict, context=context)
                for sample_cb in sample_cbs:
                    if sample_cb is not None:
                        sample_cb(generator, batch_index, context=context)


        ##----------> UPON FINISHING EACH CONTEXT...

        # Close progres-bar(s)
        progress.close()
        if generator is not None:
            progress_gen.close()

        # Parameter regularization: update and compute the parameter importance estimates
        if context<len(train_datasets) and isinstance(model, ContinualLearner):
            # -find allowed classes
            allowed_classes = active_classes
            ##--> EWC/NCL: estimate the Fisher Information matrix
            if model.importance_weighting=='fisher' and (model.weight_penalty or model.precondition):
                if model.fisher_kfac:
                    model.estimate_kfac_fisher(training_dataset, allowed_classes=allowed_classes)
                else:
                    model.estimate_fisher(training_dataset, allowed_classes=allowed_classes)
            ##--> OWM: calculate and update the projection matrix
            if model.importance_weighting=='owm' and (model.weight_penalty or model.precondition):
                model.estimate_owm_fisher(training_dataset, allowed_classes=allowed_classes)
            ##--> SI: calculate and update the normalized path integral
            if model.importance_weighting=='si' and (model.weight_penalty or model.precondition):
                model.update_omega(W, model.epsilon)

        # MEMORY BUFFER: update the memory buffer
        if checkattr(model, 'use_memory_buffer'):
            samples_per_class = model.budget_per_class if (not model.use_full_capacity) else int(
                np.floor((model.budget_per_class*len(train_datasets))/context)
            )
            # reduce examplar-sets (only needed when '--use-full-capacity' is selected)
            model.reduce_memory_sets(samples_per_class)
            # for each new class trained on, construct examplar-set
            new_classes = list(range(model.classes_per_context*(context-1), model.classes_per_context*context))
            for class_id in new_classes:
                # create new dataset containing only all examples of this class
                class_dataset = SubDataset(original_dataset=train_dataset, sub_labels=[class_id])
                # based on this dataset, construct new memory-set for this class
                allowed_classes = active_classes
                model.construct_memory_set(dataset=class_dataset, n=samples_per_class, label_set=allowed_classes)
            model.compute_means = True

        # Run the callbacks after finishing each context
        for context_cb in context_cbs:
            if context_cb is not None:
                context_cb(model, iters, context=context)

        # REPLAY: update source for replay
        if context<len(train_datasets) and hasattr(model, 'replay_mode'):
            previous_model = copy.deepcopy(model).eval()
            if model.replay_mode == 'generative':
                ReplayGeneratedData = True
                previous_generator = copy.deepcopy(generator).eval() if generator is not None else previous_model
            elif model.replay_mode == 'current':
                ReplayCurrentData = True
            elif model.replay_mode in ('buffer', 'all'):
                ReplayStoredData = True
                if model.replay_mode == "all":
                    previous_datasets = train_datasets[:context]
                else:
                    previous_datasets = [MemorySetDataset(model.memory_sets)]

#------------------------------------------------------------------------------------------------------------#

def train_fromp(model, train_datasets, iters=2000, batch_size=32,
                loss_cbs=list(), eval_cbs=list(), context_cbs=list(), **kwargs):
    '''Train a model (with a "train_a_batch" method) on multiple contexts using the FROMP algorithm.

    [model]               <nn.Module> main model to optimize across all contexts
    [train_datasets]      <list> with for each context the training <DataSet>
    [iters]               <int>, # of optimization-steps (i.e., # of mini-batches) per context
    [batch_size]          <int>, # of samples per mini-batch
    [*_cbs]               <list> of call-back functions to evaluate training-progress
    '''

    # Set model in training-mode
    model.train()

    # Use cuda?
    cuda = model._is_on_cuda()
    device = model._device()

    # Loop over all contexts.
    for context, train_dataset in enumerate(train_datasets, 1):

        # Find [active_classes]
        # --> one <list> with active classes of all contexts so far
        active_classes = list(range(model.classes_per_context * context))

        # Find [label_sets] (i.e., when replaying/revisiting/regularizing previous contexts, which labels to consider)
        label_sets = [active_classes]*context
        # NOTE: With Class-IL, when revisiting previous contexts, consider all labels up to *now*
        #       (and not up to when that context was encountered!)

        # FROMP: calculate and store regularisation-term-related quantities
        if context > 1:
            model.optimizer.init_context(context-1, reset=(model.optim_type=="adam_reset"),
                                         classes_per_context=model.classes_per_context, label_sets=label_sets)

        # Initialize # iters left on current data-loader(s)
        iters_left = 1

        # Define tqdm progress bar(s)
        progress = tqdm.tqdm(range(1, iters+1))

        # Loop over all iterations
        for batch_index in range(1, iters+1):

            # Update # iters left on current data-loader(s) and, if needed, create new one(s)
            iters_left -= 1
            if iters_left==0:
                data_loader = iter(get_data_loader(train_dataset, batch_size, cuda=cuda, drop_last=True))
                iters_left = len(data_loader)

            # -----------------Collect data------------------#
            x, y = next(data_loader)           #--> sample training data of current context
            # --> adjust the y-targets to the 'active range'
            x, y = x.to(device), y.to(device)  # --> transfer them to correct device

            #---> Train MAIN MODEL
            if batch_index <= iters:

                # Optimiser step
                loss_dict = model.optimizer.step(x, y, label_sets, context-1, model.classes_per_context)

                # Fire callbacks (for visualization of training-progress / evaluating performance after each context)
                for loss_cb in loss_cbs:
                    if loss_cb is not None:
                        loss_cb(progress, batch_index, loss_dict, context=context)
                for eval_cb in eval_cbs:
                    if eval_cb is not None:
                        eval_cb(model, batch_index, context=context)

        ##----------> UPON FINISHING EACH CONTEXT...

        # Close progres-bar(s)
        progress.close()

        # MEMORY BUFFER: update the memory buffer
        if checkattr(model, 'use_memory_buffer'):
            samples_per_class = model.budget_per_class if (not model.use_full_capacity) else int(
                np.floor((model.budget_per_class*len(train_datasets))/context)
            )
            # reduce examplar-sets (only needed when '--use-full-capacity' is selected)
            model.reduce_memory_sets(samples_per_class)
            # for each new class trained on, construct examplar-set
            new_classes = list(range(model.classes_per_context*(context-1), model.classes_per_context*context))
            for class_id in new_classes:
                # create new dataset containing only all examples of this class
                class_dataset = SubDataset(original_dataset=train_dataset, sub_labels=[class_id])
                # based on this dataset, construct new memory-set for this class
                allowed_classes = active_classes
                model.construct_memory_set(dataset=class_dataset, n=samples_per_class, label_set=allowed_classes)
            model.compute_means = True

        # FROMP: update covariance (\Sigma)
        if context<len(train_datasets):
            memorable_loader = DataLoader(dataset=train_dataset, batch_size=6, shuffle=False, num_workers=3)
            model.optimizer.update_fisher(
                memorable_loader,
                label_set=active_classes
            )

        # Run the callbacks after finishing each context
        for context_cb in context_cbs:
            if context_cb is not None:
                context_cb(model, iters, context=context)

#------------------------------------------------------------------------------------------------------------#

def train_gen_classifier(model, train_datasets, iters=2000, epochs=None, batch_size=32,
                         loss_cbs=list(), sample_cbs=list(), eval_cbs=list(), context_cbs=list(), **kwargs):
    '''Train a generative classifier with a separate VAE per class.

    [model]               <nn.Module> the generative classifier to train
    [train_datasets]      <list> with for each class the training <DataSet>
    [iters]               <int>, # of optimization-steps (i.e., # of mini-batches) per class
    [batch_size]          <int>, # of samples per mini-batch
    [*_cbs]               <list> of call-back functions to evaluate training-progress
    '''

    # Use cuda?
    device = model._device()
    cuda = model._is_on_cuda()

    # Loop over all contexts.
    classes_in_current_context = 0
    context = 1
    for class_id, train_dataset in enumerate(train_datasets):

        # Initialize # iters left on data-loader(s)
        iters_left = 1

        if epochs is not None:
            data_loader = iter(get_data_loader(train_dataset, batch_size, cuda=cuda, drop_last=False))
            iters = len(data_loader)*epochs

        # Define a tqdm progress bar(s)
        progress = tqdm.tqdm(range(1, iters+1))

        # Loop over all iterations
        for batch_index in range(1, iters+1):

            # Update # iters left on current data-loader(s) and, if needed, create new one(s)
            iters_left -= 1
            if iters_left==0:
                data_loader = iter(get_data_loader(train_dataset, batch_size, cuda=cuda,
                                                   drop_last=True if epochs is None else False))
                iters_left = len(data_loader)

            # Collect data
            x, y = next(data_loader)                                    #--> sample training data of current context
            x, y = x.to(device), y.to(device)                           #--> transfer them to correct device
            #y = y.expand(1) if len(y.size())==1 else y                 #--> hack for if batch-size is 1

            # Select model to be trained
            model_to_be_trained = getattr(model, "vae{}".format(class_id))

            # Train the VAE model of this class with this batch
            loss_dict = model_to_be_trained.train_a_batch(x)

            # Fire callbacks (for visualization of training-progress)
            for loss_cb in loss_cbs:
                if loss_cb is not None:
                    loss_cb(progress, batch_index, loss_dict, class_id=class_id)
            for eval_cb in eval_cbs:
                if eval_cb is not None:
                    eval_cb(model, batch_index+classes_in_current_context*iters, context=context)
            for sample_cb in sample_cbs:
                if sample_cb is not None:
                    sample_cb(model_to_be_trained, batch_index, class_id=class_id)

        # Close progres-bar(s)
        progress.close()

        # Did a context just finish?
        classes_in_current_context += 1
        if classes_in_current_context==model.classes_per_context:
            # Run the callbacks after finishing each context
            for context_cb in context_cbs:
                if context_cb is not None:
                    context_cb(model, iters, context=context)
            # Updated counts
            classes_in_current_context = 0
            context += 1
