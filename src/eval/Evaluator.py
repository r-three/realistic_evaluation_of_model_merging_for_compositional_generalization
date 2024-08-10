from typing import Any, Dict

from src.eval.EvaluationConfig import EvaluationConfig
from src.eval.PredictionWriter import PredictionWriter
from src.eval.Scorer import Scorer


class Evaluator(object):
    def __init__(
        self, evaluation_config: EvaluationConfig, metrics: Any, prediction_dir: str
    ):
        """

        Args:
            evaluation_config:
            metrics:
            prediction_dir:
        """
        self.evaluation_config = evaluation_config
        self.scorer = Scorer(evaluation_config, prediction_dir, metrics)
        self.writer = PredictionWriter(evaluation_config, prediction_dir)

        self.seen_idxs = {}

    def add_batch(self, batchOf_evalInfo: Dict):
        """
        Add batch to scorer and writer

        Args:
            batchOf_evalInfo:
        """
        batchOf_idxs = batchOf_evalInfo["idx"]

        # For distributed setup, the batch might have duplicate examples due to padding that we have to remove.
        # 1) Compute the indices we have to remove
        idx_toRemove = []
        for batch_idx, idx in enumerate(batchOf_idxs):
            if idx in self.seen_idxs:
                idx_toRemove.append(batch_idx)
            self.seen_idxs[idx] = True

        # 2) Remove these indices
        filteredBatch_ofEvalInfo = {}
        for key, batchOf_values in batchOf_evalInfo.items():
            filtered_value = []
            for batch_idx, value in enumerate(batchOf_values):
                if batch_idx not in idx_toRemove:
                    filtered_value.append(value)

            filteredBatch_ofEvalInfo[key] = filtered_value

        self.scorer.add_batch(filteredBatch_ofEvalInfo)
        self.writer.log_batch(filteredBatch_ofEvalInfo)

    def get_result(self) -> Dict:
        """
        Get evaluation results

        Returns:
            score
        """
        return self.scorer.get_score()

    def get_evaluationRunDir(self) -> str:
        """
        Get directory of evaluation run

        Returns:
            evaluationRun_dir
        """
        return self.writer.get_evaluationRunDir()
