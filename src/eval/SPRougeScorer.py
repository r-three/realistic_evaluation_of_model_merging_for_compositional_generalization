from rouge_score import rouge_scorer
from rouge_score.scoring import BootstrapAggregator


class SPRougeScorer(object):
    def __init__(self, evaluation_config):
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True,
            lang=evaluation_config.get_datasetConfig().language,
        )
        self.aggregator = BootstrapAggregator()

    def add_batch(self, batch_predictedTxt: list[str], batch_goldTxt: list[str]):
        """

        Args:
            batch_predictedTxt:
            batch_goldTxt:
        """
        for predicted_txt, gold_txt in zip(batch_predictedTxt, batch_goldTxt):
            scores = self.scorer.score(gold_txt, predicted_txt)
            self.aggregator.add_scores(scores)

    def compute(self) -> dict[str, float]:
        result = self.aggregator.aggregate()
        return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}
