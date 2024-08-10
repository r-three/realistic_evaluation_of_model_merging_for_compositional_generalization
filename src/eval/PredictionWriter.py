import json
import os
from typing import Dict

from src.eval.EvaluationConfig import EvaluationConfig
from src.utils.io import *
from src.utils.utils import *


class PredictionWriter(object):
    def __init__(self, evaluation_config: EvaluationConfig, prediction_dir: str):
        """
        Args:
            evaluation_config:
            prediction_dir:
        """
        self.evaluation_config = evaluation_config
        self.predictions_file = None
        if prediction_dir is not None:
            self.evaluationRun_dir = self._open_writer(prediction_dir)

    def _get_nextRunIdx(self, finalPrediction_dir: str):
        """
        Args:
            finalPrediction_dir:

        Returns:

        """
        run_idx = 0
        while True:
            run_dir = os.path.join(finalPrediction_dir, f"run_{run_idx}")
            if not os.path.exists(run_dir):
                return run_idx
            run_idx += 1

    def _open_writer(self, prediction_dir: str):
        """
        Get the corect file for the writer under prediction_dir and open it

        Args:
            prediction_dir:

        Returns:

        """
        dataset = self.evaluation_config.get_datasetConfig().dataset
        if self.evaluation_config.get_datasetConfig().language_code is not None:
            dataset += f"_{self.evaluation_config.get_datasetConfig().language_code}"
        if self.evaluation_config.get_datasetConfig().language is not None:
            dataset += f"_{self.evaluation_config.get_datasetConfig().language}"
        if self.evaluation_config.get_datasetConfig().domain is not None:
            assert self.evaluation_config.get_datasetConfig().task is not None
            dataset += f"_{self.evaluation_config.get_datasetConfig().domain}_{self.evaluation_config.get_datasetConfig().task}"
        prediction_dir = os.path.join(prediction_dir, dataset)

        next_runIdx = self._get_nextRunIdx(prediction_dir)
        # Use the previous run if we want to overwrite it
        if self.evaluation_config.overwrite_previous_run and next_runIdx > 0:
            next_runIdx -= 1

        evaluationRun_dir = os.path.join(prediction_dir, f"run_{next_runIdx}")
        os.makedirs(evaluationRun_dir, exist_ok=True)

        self.predictions_fp = os.path.join(evaluationRun_dir, f"predictions.json")
        self.predictions_file = open(self.predictions_fp, "w+", encoding="utf-8")
        return evaluationRun_dir

    def log_batch(self, batchOf_evalInfo: Dict):
        """
        Write batch to rpediction file

        Args:
            batchOf_evalInfo:
        """
        if self.predictions_file is not None:
            listOf_evalInfo = convert_dictOfLists_to_listOfDicts(batchOf_evalInfo)

            for eval_info in listOf_evalInfo:
                self.predictions_file.write(
                    json.dumps(eval_info, ensure_ascii=False) + "\n"
                )
            self.predictions_file.flush()

    def get_evaluationRunDir(self):
        """
        Get directory of evaluation_run
        """
        if self.predictions_file is not None:
            return self.evaluationRun_dir
        else:
            return None
