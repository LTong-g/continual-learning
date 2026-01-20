from eval import evaluate


#########################################################
## Callback-functions for evaluating model-performance ##
#########################################################

def _sample_cb(log, config, visdom=None, test_datasets=None, sample_size=64):
    '''Initiates function for evaluating samples of generative model.

    [test_datasets]     None or <list> of <Datasets> (if provided, also reconstructions are shown)'''

    def sample_cb(generator, batch, context=1, class_id=None, **kwargs):
        '''Callback-function, to evaluate sample (and reconstruction) ability of the model.'''

        if batch % log == 0:

            if test_datasets is not None:
                evaluate.show_reconstruction(generator, test_datasets[context-1], config, size=int(sample_size/2),
                                             visdom=None, context=context)

            evaluate.show_samples(
                generator, config, visdom=None, size=sample_size,
                visdom_title='Samples{}'.format(" VAE-{}".format(class_id) if class_id is not None else "")
            )

    return sample_cb


def _eval_cb(log, test_datasets, visdom=None, plotting_dict=None, iters_per_context=None, test_size=None,
             summary_graph=True, S='mean'):
    '''Initiates function for evaluating performance of classifier (in terms of accuracy).

    [test_datasets]       <list> of <Datasets>; also if only 1 context, it should be presented as a list!
    '''

    def eval_cb(classifier, batch, context=1):
        '''Callback-function, to evaluate performance of classifier.'''

        iteration = batch if (context is None or context==1) else (context-1)*iters_per_context + batch

        if iteration % log == 0:

            if (S is not None) and hasattr(classifier, 'S'):
                classifier.S = S

            evaluate.test_all_so_far(classifier, test_datasets, context, iteration, test_size=test_size,
                                     visdom=None, summary_graph=summary_graph, plotting_dict=plotting_dict)

    return eval_cb


##------------------------------------------------------------------------------------------------------------------##

########################################################################
## Callback-functions for keeping track of loss and training progress ##
########################################################################

def _classifier_loss_cb(log=1, visdom=None, model=None, contexts=None, iters_per_context=None, progress_bar=True):
    '''Initiates function for keeping track of, and reporting on, the progress of the classifier's training.'''

    def cb(bar, iter, loss_dict, context=1):
        '''Callback-function, to call on every iteration to keep track of training progress.'''

        iteration = iter if context==1 else (context-1)*iters_per_context + iter

        if progress_bar and bar is not None:
            context_stm = "" if (contexts is None) else " Context: {}/{} |".format(context, contexts)
            bar.set_description(
                '<CLASSIFIER> |{t_stm} training loss: {loss:.3} | training accuracy: {prec:.3} |'
                    .format(t_stm=context_stm, loss=loss_dict['loss_total'], prec=loss_dict['accuracy'])
            )
            bar.update(1)

    return cb
