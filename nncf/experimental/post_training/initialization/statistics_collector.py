class StatisticsCollector:
    def __init__(self, compressed_model, dataloader, engine):
        self.compressed_model = compressed_model
        self.dataloader = dataloader
        self.engine = engine
        self.is_calculate_metric = False

    def collect_statistics(self, layers_to_collect_statistics, num_iters):
        model_with_additional_outputs = self._create_model_for_statistics_collection()
        self.engine.set_model(model_with_additional_outputs)
        for i in range(num_iters):
            output, target = self.engine.infer_model(i)
            # output should be topologicly sorted
            self._agregate_statistics(output)
            if self.is_calculate_metric:
                self._calculate_metric(target)

    def _create_model_for_statistics_collection(self):
        pass

    def _agregate_statistics(self):
        pass

    def _calculate_metric(self, target):
        pass
