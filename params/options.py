import argparse

##-------------------------------------------------------------------------------------------------------------------##

# Where to store the data / results / models / plots
store = "./store"

##-------------------------------------------------------------------------------------------------------------------##

####################
## Define options ##
####################

def define_args(filename, description):
    parser = argparse.ArgumentParser('./{}.py'.format(filename), description=description)
    return parser


def add_general_options(parser, main=False, **kwargs):
    if main:
        parser.add_argument('--get-stamp', action='store_true', help='print param-stamp & exit')
    parser.add_argument('--seed', type=int, default=0, help='[first] random seed (for each random-module used)')
    parser.add_argument('--no-gpus', action='store_false', dest='gpu', help="don't use GPUs")
    parser.add_argument('--no-save', action='store_false', dest='save', help="don't save trained models")
    parser.add_argument('--full-stag', type=str, metavar='STAG', default='none', help="tag for saving full model")
    parser.add_argument('--full-ltag', type=str, metavar='LTAG', default='none', help="tag for loading full model")
    if main:
        parser.add_argument('--test', action='store_false', dest='train', help='evaluate previously saved model')
    parser.add_argument('--data-dir', type=str, default='{}/datasets'.format(store), dest='d_dir',
                        help="default: %(default)s")
    parser.add_argument('--model-dir', type=str, default='{}/models'.format(store), dest='m_dir',
                        help="default: %(default)s")
    parser.add_argument('--plot-dir', type=str, default='{}/plots'.format(store), dest='p_dir',
                        help="default: %(default)s")
    parser.add_argument('--results-dir', type=str, default='{}/results'.format(store), dest='r_dir',
                        help="default: %(default)s")
    return parser

##-------------------------------------------------------------------------------------------------------------------##


def add_eval_options(parser, main=False, **kwargs):
    eval_params = parser.add_argument_group('Evaluation Parameters')
    eval_params.add_argument('--time', action='store_true', help="keep track of total training time")
    if main:
        eval_params.add_argument('--pdf', action='store_true', help="generate pdf with results")
    eval_params.add_argument('--results-dict', action='store_true', help="output dict with results after each task")
    eval_params.add_argument('--loss-log', type=int, metavar="N",
                             help="# iters after which to plot loss (def: # iters)")
    eval_params.add_argument('--acc-log', type=int, metavar="N",
                             help="# iters after which to plot accuracy (def: # iters)")
    eval_params.add_argument('--acc-n', type=int, default=1024,
                             help="# samples to evaluate accuracy (after each context)")
    return parser

##-------------------------------------------------------------------------------------------------------------------##


def add_problem_options(parser, **kwargs):
    problem_params = parser.add_argument_group('Problem Specification')
    cl_protocols = ['splitMNIST', 'permMNIST', 'CIFAR10', 'CIFAR100']
    problem_params.add_argument('--experiment', type=str, default='splitMNIST', choices=cl_protocols)
    problem_params.add_argument('--scenario', type=str, default='class', choices=['class'])
    problem_params.add_argument('--contexts', type=int, metavar='N', help='number of contexts')
    problem_params.add_argument('--iters', type=int, help="# iterations (mini-batches) per context")
    problem_params.add_argument('--batch', type=int, help="mini batch size (# observations per iteration)")
    problem_params.add_argument('--no-norm', action='store_false', dest='normalize',
                                help="don't normalize images (only for CIFAR)")
    return parser

##-------------------------------------------------------------------------------------------------------------------##


def add_model_options(parser, **kwargs):
    model = parser.add_argument_group('Parameters Main Model')
    # -convolutional layers
    model.add_argument('--conv-type', type=str, default="standard", choices=["standard", "resNet"])
    model.add_argument('--n-blocks', type=int, default=2, help="# blocks per conv-layer (only for 'resNet')")
    model.add_argument('--depth', type=int, default=None, help="# of convolutional layers (0 = only fc-layers)")
    model.add_argument('--reducing-layers', type=int, dest='rl', default=None,
                       help="# of layers with stride (=image-size halved)")
    model.add_argument('--channels', type=int, default=None, help="# of channels 1st conv-layer (doubled every 'rl')")
    model.add_argument('--conv-bn', type=str, default="yes", help="use batch-norm in the conv-layers (yes|no)")
    model.add_argument('--conv-nl', type=str, default="relu", choices=["relu", "leakyrelu"])
    model.add_argument('--global-pooling', action='store_true', dest='gp', help="ave global pool after conv-layers")
    # -fully connected layers
    model.add_argument('--fc-layers', type=int, default=None, dest='fc_lay', help="# of fully-connected layers")
    model.add_argument('--fc-units', type=int, metavar="N", help="# of units in hidden fc-layers")
    model.add_argument('--fc-drop', type=float, default=0., help="dropout probability for fc-units")
    model.add_argument('--fc-bn', type=str, default="no", help="use batch-norm in the fc-layers (no|yes)")
    model.add_argument('--fc-nl', type=str, default="relu", choices=["relu", "leakyrelu", "none"])
    return parser

##-------------------------------------------------------------------------------------------------------------------##


def add_train_options(parser, **kwargs):
    train_params = parser.add_argument_group('Training Parameters')
    train_params.add_argument('--lr', type=float, help="learning rate")
    train_params.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    train_params.add_argument("--momentum", type=float, default=0., help="momentum (if using SGD optimizer)")
    return parser

##-------------------------------------------------------------------------------------------------------------------##


def add_cl_options(parser, main=False, **kwargs):
    if main:
        baseline_options = parser.add_argument_group('Baseline Options')
        baseline_options.add_argument('--joint', action='store_true', help="train once on data of all contexts")
        baseline_options.add_argument('--cummulative', action='store_true',
                                      help="train incrementally on data of all contexts so far")

    cl_options = parser.add_argument_group('Continual Learning Options')
    cl_options.add_argument('--ewc', action='store_true', help="use Elastic Weight Consolidation")
    cl_options.add_argument('--si', action='store_true', help="use Synaptic Intelligence")
    cl_options.add_argument('--lwf', action='store_true', help="use Learning without Forgetting")

    # Parameter-regularization specific options
    cl_options.add_argument('--ewc-lambda', type=float, default=None, help="EWC regularization strength")
    cl_options.add_argument('--si-c', type=float, default=None, help="SI regularization strength")
    cl_options.add_argument('--offline', action='store_true', help='use Offline EWC rather than Online EWC')
    cl_options.add_argument('--fisher-n', type=int, default=None, help='number of samples to estimate Fisher')
    cl_options.add_argument('--fisher-labels', type=str, default='all', choices=['all', 'sample', 'pred', 'true'])
    cl_options.add_argument('--fisher-batch', type=int, default=1, help='batch size to estimate Fisher')
    cl_options.add_argument('--gamma', type=float, default=1.0, help='Online EWC decay factor')
    cl_options.add_argument('--epsilon', type=float, default=0.1, help='SI dampening parameter')
    return parser
