def print_maximal_degradation_warning(config, acc_aware_statistics, logger):
    acc_aware_params = config.accuracy_aware_training.params
    maximal_relative_accuracy_degradation = acc_aware_params.get('maximal_relative_accuracy_degradation', None)
    maximal_absolute_accuracy_degradation = acc_aware_params.get('maximal_absolute_accuracy_degradation', None)
    if maximal_relative_accuracy_degradation is not None:
        final_relative_accuracy_degradation = acc_aware_statistics['relative_accuracy_degradation']
        if final_relative_accuracy_degradation > maximal_relative_accuracy_degradation:
            logger.warning(f'Was not able to compress a model to fit the required maximal relative accuracy '
                           f'degradation. Actual: {final_relative_accuracy_degradation:.4f}. '
                           f'Required: {maximal_relative_accuracy_degradation:.4f}.')
    if maximal_absolute_accuracy_degradation is not None:
        final_absolute_accuracy_degradation = acc_aware_statistics['absolute_accuracy_degradation']
        if final_absolute_accuracy_degradation > maximal_absolute_accuracy_degradation:
            logger.warning(f'Was not able to compress a model to fit the required maximal absolute accuracy '
                           f'degradation. Actual: {final_absolute_accuracy_degradation:.4f}. '
                           f'Required: {maximal_absolute_accuracy_degradation:.4f}.')
