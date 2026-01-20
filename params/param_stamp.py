from data.load import get_context_set
from models import define_models as define


def get_param_stamp_from_args(args):
    '''To get param-stamp a bit quicker.'''
    config = get_context_set(
        name=args.experiment, scenario=args.scenario, contexts=args.contexts, data_dir=args.d_dir, only_config=True,
        normalize=getattr(args, "normalize", False), verbose=False, singlehead=getattr(args, 'singlehead', False),
    )
    model = define.define_classifier(args=args, config=config, device='cpu')
    param_stamp = get_param_stamp(args, model.name, verbose=False)
    return param_stamp


def get_param_stamp(args, model_name, verbose=True):
    '''Based on the input-arguments, produce a "parameter-stamp".'''

    # -for problem specification
    multi_n_stamp = "{n}{joint}{cum}-{sce}".format(
        n=args.contexts,
        joint="-Joint" if getattr(args, 'joint', False) else "",
        cum="-Cummulative" if getattr(args, 'cummulative', False) else "",
        sce=args.scenario,
    )
    problem_stamp = "{exp}{norm}{multi_n}".format(
        exp=args.experiment, norm="-N" if getattr(args, 'normalize', False) else "", multi_n=multi_n_stamp
    )
    if verbose:
        print(" --> problem:       " + problem_stamp)

    # -for model
    model_stamp = model_name
    if verbose:
        print(" --> model:         " + model_stamp)

    # -for training settings
    train_stamp = "i{num}-lr{lr}-b{bsz}-{optim}{mom}".format(
        num=args.iters, lr=args.lr, bsz=args.batch, optim=args.optimizer,
        mom="-m{}".format(args.momentum) if args.optimizer == 'sgd' and getattr(args, 'momentum', 0) > 0 else "",
    )
    if verbose:
        print(" --> train-params:  " + train_stamp)

    # -for parameter regularization
    param_reg_stamp = ""
    if getattr(args, 'weight_penalty', False):
        if args.importance_weighting == 'fisher':
            param_reg_stamp = "--EWC{}".format(args.reg_strength)
        elif args.importance_weighting == 'si':
            param_reg_stamp = "--SI{}".format(args.reg_strength)

    # -for LwF
    replay_stamp = "--LwF" if getattr(args, 'replay', 'none') == 'current' else ""

    param_stamp = "{}--{}--{}{}{}{}".format(
        problem_stamp, model_stamp, train_stamp, param_reg_stamp, replay_stamp,
        "-s{}".format(args.seed) if not args.seed == 0 else ""
    )

    if verbose:
        print(param_stamp)
    return param_stamp
