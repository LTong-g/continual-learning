#!/usr/bin/env python3
import os
import numpy as np
import time
import torch
from torch import optim
# -custom-written libraries
import utils
from utils import checkattr
from data.load import get_context_set
from models import define_models as define
from models.cl.continual_learner import ContinualLearner
from train.train_task_based import train_cl
from params import options
from params.param_stamp import get_param_stamp, get_param_stamp_from_args
from params.param_values import set_method_options, check_for_errors, set_default_values
from eval import evaluate, callbacks as cb
from visual import visual_plt


## Function for specifying input-options and organizing / checking them
def handle_inputs():
    # Set indicator-dictionary for correctly retrieving / checking input options
    kwargs = {'main': True}
    # Define input options
    parser = options.define_args(filename="main", description='Run a class-incremental continual learning experiment '
                                                              'using the academic continual learning setting.')
    parser = options.add_general_options(parser, **kwargs)
    parser = options.add_eval_options(parser, **kwargs)
    parser = options.add_problem_options(parser, **kwargs)
    parser = options.add_model_options(parser, **kwargs)
    parser = options.add_train_options(parser, **kwargs)
    parser = options.add_cl_options(parser, **kwargs)
    # Parse, process and check chosen options
    args = parser.parse_args()
    set_method_options(args)                         # -if a method's "convenience"-option is chosen, select components
    set_default_values(args, also_hyper_params=True) # -set defaults, some are based on chosen scenario / experiment
    check_for_errors(args, **kwargs)                 # -check whether incompatible options are selected
    return args


def run(args, verbose=False):

    # Create plots- and results-directories if needed
    if not os.path.isdir(args.r_dir):
        os.mkdir(args.r_dir)
    if checkattr(args, 'pdf') and not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

    # If only want param-stamp, get it printed to screen and exit
    if checkattr(args, 'get_stamp'):
        print(get_param_stamp_from_args(args=args))
        exit()

    # Use cuda or mps (apple silicon)?
    cuda = torch.cuda.is_available() and args.gpu
    mps = torch.backends.mps.is_available() and args.gpu
    if cuda:
        device = torch.device("cuda")
    elif mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Report whether cuda or mps is used
    if verbose:
        if cuda:
            print("CUDA is used")
        elif mps:
            print("MPS is used (apple silicon GPU)")
        else:
            print("NO GPU is used!")

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)
    elif mps:
        torch.mps.manual_seed(args.seed)

    #-------------------------------------------------------------------------------------------------#

    #----------------#
    #----- DATA -----#
    #----------------#

    # Prepare data for chosen experiment
    if verbose:
        print("\n\n " +' LOAD DATA '.center(70, '*'))
    (train_datasets, test_datasets), config = get_context_set(
        name=args.experiment, scenario=args.scenario, contexts=args.contexts, data_dir=args.d_dir,
        normalize=checkattr(args, "normalize"), verbose=verbose, exception=(args.seed==0),
        singlehead=checkattr(args, 'singlehead')
    )

    #-------------------------------------------------------------------------------------------------#

    #----------------------#
    #----- CLASSIFIER -----#
    #----------------------#

    # Define the classifier
    if verbose:
        print("\n\n " + ' DEFINE THE CLASSIFIER '.center(70, '*'))
    model = define.define_classifier(args=args, config=config, device=device)

    # Initialize parameters and optimizer
    define.init_params(model, args)
    model.optim_list = [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr}]
    model.optim_type = args.optimizer
    if model.optim_type == "adam":
        model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))
    elif model.optim_type == "sgd":
        model.optimizer = optim.SGD(model.optim_list, momentum=args.momentum if hasattr(args, 'momentum') else 0.)

    # On what scenario will model be trained? If needed, indicate whether singlehead output / how to set active classes.
    model.scenario = args.scenario
    model.classes_per_context = config['classes_per_context']
    model.singlehead = checkattr(args, 'singlehead')
    model.neg_samples = args.neg_samples if hasattr(args, 'neg_samples') else "all"

    # Print some model-characteristics on the screen
    if verbose:
        utils.print_model_info(model)

    # -------------------------------------------------------------------------------------------------#

    #-------------------------------------------------#
    #----- CL-STRATEGY: PARAMETER REGULARIZATION -----#
    #-------------------------------------------------#

    # Options for computing the Fisher Information matrix (EWC)
    use_fisher = hasattr(args, 'importance_weighting') and args.importance_weighting == "fisher" and \
                 checkattr(args, 'weight_penalty')
    if isinstance(model, ContinualLearner) and use_fisher:
        model.fisher_n = args.fisher_n if hasattr(args, 'fisher_n') else None
        model.fisher_labels = args.fisher_labels if hasattr(args, 'fisher_labels') else 'all'
        model.fisher_batch = args.fisher_batch if hasattr(args, 'fisher_batch') else 1
        model.offline = checkattr(args, 'offline')
        if not model.offline:
            model.gamma = args.gamma if hasattr(args, 'gamma') else 1.

    # Parameter regularization by adding a weight penalty (EWC, SI)
    if isinstance(model, ContinualLearner) and checkattr(args, 'weight_penalty'):
        model.weight_penalty = True
        model.importance_weighting = args.importance_weighting
        model.reg_strength = args.reg_strength
        if model.importance_weighting == 'si':
            model.epsilon = args.epsilon if hasattr(args, 'epsilon') else 0.1

    #-------------------------------------------------------------------------------------------------#

    #--------------------------------------------------#
    #----- CL-STRATEGY: FUNCTIONAL REGULARIZATION -----#
    #--------------------------------------------------#

    # LwF: use distillation loss on current batch as soft targets
    if isinstance(model, ContinualLearner) and hasattr(args, 'replay'):
        model.replay_targets = "soft" if checkattr(args, 'distill') else "hard"
        model.KD_temp = args.temp if hasattr(args, 'temp') else 2.
        if args.replay == "current" and model.replay_targets == "soft":
            model.lwf_weighting = True
        model.replay_mode = args.replay

    #-------------------------------------------------------------------------------------------------#

    #---------------------------#
    #----- PARAMETER STAMP -----#
    #---------------------------#

    # Get parameter-stamp (and print on screen)
    if verbose:
        print('\n\n' + ' PARAMETER STAMP '.center(70, '*'))
    param_stamp = get_param_stamp(args, model.name, verbose=verbose)

    #-------------------------------------------------------------------------------------------------#

    #---------------------#
    #----- CALLBACKS -----#
    #---------------------#

    # Prepare for keeping track of performance during training for plotting in pdf
    plotting_dict = evaluate.initiate_plotting_dict(args.contexts) if (
        checkattr(args, 'pdf') or checkattr(args, 'results_dict')
    ) else None

    # Callbacks for reporting and visualizing loss
    loss_cbs = [
        cb._classifier_loss_cb(
            log=args.loss_log, model=model, contexts=args.contexts, iters_per_context=args.iters,
        )
    ]

    # Callbacks for reporting and visualizing accuracy
    eval_cbs = [
        cb._eval_cb(log=args.acc_log, test_datasets=test_datasets, iters_per_context=args.iters,
                    test_size=args.acc_n)
    ]
    context_cbs = [
        cb._eval_cb(log=args.iters, test_datasets=test_datasets, plotting_dict=plotting_dict,
                    iters_per_context=args.iters, test_size=args.acc_n)
    ]

    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- TRAINING -----#
    #--------------------#

    baseline = 'joint' if checkattr(args, 'joint') else ('cummulative' if checkattr(args, 'cummulative') else 'none')

    # Train model
    if args.train:
        if verbose:
            print('\n\n' + ' TRAINING '.center(70, '*'))
        if args.time:
            start = time.time()
        train_cl(
            model, train_datasets, iters=args.iters, batch_size=args.batch, baseline=baseline,
            eval_cbs=eval_cbs, loss_cbs=loss_cbs, context_cbs=context_cbs,
        )
        if args.time:
            training_time = time.time() - start
            time_file = open("{}/time-{}.txt".format(args.r_dir, param_stamp), 'w')
            time_file.write('{}\n'.format(training_time))
            time_file.close()
            if verbose and args.time:
                print("Total training time = {:.1f} seconds\n".format(training_time))
        if args.save:
            save_name = "mM-{}".format(param_stamp) if (
                not hasattr(args, 'full_stag') or args.full_stag == "none"
            ) else "{}-{}".format(model.name, args.full_stag)
            utils.save_checkpoint(model, args.m_dir, name=save_name, verbose=verbose)
    else:
        if verbose:
            print("\nLoading parameters of previously trained model...")
        load_name = "mM-{}".format(param_stamp) if (
            not hasattr(args, 'full_ltag') or args.full_ltag == "none"
        ) else "{}-{}".format(model.name, args.full_ltag)
        utils.load_checkpoint(model, args.m_dir, name=load_name, verbose=verbose, strict=False)

    #-------------------------------------------------------------------------------------------------#

    #----------------------#
    #----- EVALUATION -----#
    #----------------------#

    if verbose:
        print('\n\n' + ' EVALUATION '.center(70, '*'))

    if verbose:
        print("\n Accuracy of final model on test-set:")
    accs = []
    for i in range(args.contexts):
        acc = evaluate.test_acc(
            model, test_datasets[i], verbose=False, test_size=None, context_id=i, allowed_classes=list(
                range(config['classes_per_context']*i, config['classes_per_context']*(i+1))
            ) if (args.scenario == "task" and not checkattr(args, 'singlehead')) else None,
        )
        if verbose:
            print(" - Context {}: {:.4f}".format(i + 1, acc))
        accs.append(acc)
    average_accs = sum(accs) / args.contexts
    if verbose:
        print('=> average accuracy over all {} contexts: {:.4f}\n\n'.format(args.contexts, average_accs))
    file_name = "{}/acc-{}.txt".format(args.r_dir, param_stamp)
    output_file = open(file_name, 'w')
    output_file.write('{}\n'.format(average_accs))
    output_file.close()
    if checkattr(args, 'results_dict'):
        file_name = "{}/dict-{}--n{}".format(args.r_dir, param_stamp, "All" if args.acc_n is None else args.acc_n)
        utils.save_object(plotting_dict, file_name)

    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- PLOTTING -----#
    #--------------------#

    if checkattr(args, 'pdf'):
        plot_name = "{}/{}.pdf".format(args.p_dir, param_stamp)
        pp = visual_plt.open_pdf(plot_name)
        figure_list = []
        plot_list = []
        for i in range(args.contexts):
            plot_list.append(plotting_dict['acc per context']['context {}'.format(i+1)])
        figure_list.append(visual_plt.plot_lines(
            plot_list, x_axes=plotting_dict['x_iteration'], line_names=['context {}'.format(i + 1) for i in range(args.contexts)],
            ylabel='test accuracy', xlabel='iterations', title='accuracy per context',
        ))
        figure_list.append(visual_plt.plot_lines(
            [plotting_dict['average']], x_axes=plotting_dict['x_iteration'], line_names=['average'],
            ylabel='test accuracy', xlabel='iterations', title='average accuracy',
        ))
        for figure in figure_list:
            pp.savefig(figure)
        pp.close()


if __name__ == '__main__':
    args = handle_inputs()
    run(args, verbose=True)
