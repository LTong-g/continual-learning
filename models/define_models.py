import utils


##-------------------------------------------------------------------------------------------------------------------##

def define_classifier(args, config, device, depth=0):
    return define_standard_classifier(args=args, config=config, device=device, depth=depth)


##-------------------------------------------------------------------------------------------------------------------##

## Function for defining discriminative classifier model

def define_standard_classifier(args, config, device, depth=0):
    # Import required model
    from models.classifier import Classifier
    # Specify model
    model = Classifier(
        image_size=config['size'],
        image_channels=config['channels'],
        classes=config['output_units'],
        # -conv-layers
        depth=depth,
        conv_type=args.conv_type if depth > 0 else None,
        start_channels=args.channels if depth > 0 else None,
        reducing_layers=args.rl if depth > 0 else None,
        num_blocks=args.n_blocks if depth > 0 else None,
        conv_bn=(True if args.conv_bn == "yes" else False) if depth > 0 else None,
        conv_nl=args.conv_nl if depth > 0 else None,
        no_fnl=True if depth > 0 else None,
        global_pooling=args.gp if depth > 0 else None,
        # -fc-layers
        fc_layers=args.fc_lay,
        fc_units=args.fc_units,
        fc_drop=args.fc_drop,
        fc_bn=True if args.fc_bn == "yes" else False,
        fc_nl=args.fc_nl,
        excit_buffer=True,
        phantom=False,
    ).to(device)
    return model


##-------------------------------------------------------------------------------------------------------------------##

## Function for (re-)initializing the parameters of [model]

def init_params(model, args, verbose=False):

    ## Initialization
    # - reinitialize all parameters according to default initialization
    model.apply(utils.weight_reset)
    # - initialize parameters according to chosen custom initialization (if requested)
    if hasattr(args, 'init_weight') and not args.init_weight == "standard":
        utils.weight_init(model, strategy="xavier_normal")
    if hasattr(args, 'init_bias') and not args.init_bias == "standard":
        utils.bias_init(model, strategy="constant", value=0.01)

    ## Use pre-training
    if hasattr(args, "pre_convE") and args.pre_convE and hasattr(model, 'depth') and model.depth > 0:
        load_name = model.convE.name if (
            not hasattr(args, 'convE_ltag') or args.convE_ltag == "none"
        ) else "{}-{}{}".format(model.convE.name, args.convE_ltag,
                                "-s{}".format(args.seed) if hasattr(args, 'seed_to_ltag') and args.seed_to_ltag else "")
        utils.load_checkpoint(model.convE, model_dir=args.m_dir, name=load_name, verbose=verbose)

    ## Freeze some parameters?
    if hasattr(args, "freeze_convE") and args.freeze_convE and hasattr(model, 'convE'):
        for param in model.convE.parameters():
            param.requires_grad = False
        model.convE.frozen = True
