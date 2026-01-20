from utils import checkattr


def set_method_options(args, **kwargs):
    if checkattr(args, 'ewc'):
        args.weight_penalty = True
        args.importance_weighting = 'fisher'
        args.offline = True
    if checkattr(args, 'si'):
        args.weight_penalty = True
        args.importance_weighting = 'si'
    if checkattr(args, 'lwf'):
        args.replay = "current"
        args.distill = True


def set_default_values(args, also_hyper_params=True, **kwargs):
    # -set default-values for certain arguments based on chosen experiment
    args.normalize = args.normalize if args.experiment in ('CIFAR10', 'CIFAR100') else False
    args.depth = (5 if (args.experiment in ('CIFAR10', 'CIFAR100')) else 0) if args.depth is None else args.depth
    args.fc_lay = 3 if args.fc_lay is None else args.fc_lay
    args.channels = 16 if args.channels is None else args.channels
    args.rl = args.rl
    args.gp = args.gp
    if args.contexts is None:
        args.contexts = 5 if args.experiment in ('splitMNIST', 'CIFAR10') else 10
    if args.iters is None:
        args.iters = 2000 if args.experiment == 'splitMNIST' else 5000
    args.lr = (0.001 if args.experiment == 'splitMNIST' else 0.0001) if args.lr is None else args.lr
    args.batch = (128 if args.experiment in ('splitMNIST', 'permMNIST') else 256) if args.batch is None else args.batch
    args.fc_units = (400 if args.experiment == 'splitMNIST' else (
        1000 if args.experiment == 'permMNIST' else 2000
    )) if args.fc_units is None else args.fc_units

    if not hasattr(args, 'replay'):
        args.replay = 'none'
    if not hasattr(args, 'distill'):
        args.distill = False

    # -unless the number of iterations after which to log is explicitly set, set them equal to # of iters per context
    args.acc_log = args.iters if (not hasattr(args, 'acc_log')) or args.acc_log is None else args.acc_log
    args.loss_log = args.iters if (not hasattr(args, 'loss_log')) or args.loss_log is None else args.loss_log

    if also_hyper_params:
        if not hasattr(args, 'si_c'):
            args.si_c = None
        if not hasattr(args, 'ewc_lambda'):
            args.ewc_lambda = None
        args.si_c = (5000. if args.experiment == 'splitMNIST' else 5.) if args.si_c is None else args.si_c
        args.ewc_lambda = (
            1000000000. if args.experiment == 'splitMNIST' else 100.
        ) if args.ewc_lambda is None else args.ewc_lambda
        if hasattr(args, 'reg_strength'):
            args.reg_strength = (
                args.si_c if checkattr(args, 'si') else (args.ewc_lambda if checkattr(args, 'ewc') else 1.)
            ) if args.reg_strength is None else args.reg_strength


def check_for_errors(args, **kwargs):
    if not hasattr(args, 'scenario') or args.scenario != 'class':
        raise ValueError("This simplified project only supports class-incremental learning.")

    if checkattr(args, 'ewc') and checkattr(args, 'si'):
        raise ValueError("Select only one of EWC or SI at a time.")

    if checkattr(args, 'lwf') and (checkattr(args, 'ewc') or checkattr(args, 'si')):
        raise ValueError("LwF should be run alone in this simplified project.")
