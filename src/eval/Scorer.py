from typing import Any, Dict

from evaluate import load

from src.eval.EvaluationConfig import EvaluationConfig
from src.eval.SPRougeScorer import SPRougeScorer
from src.utils.stats import *
from src.utils.utils import *


class Scorer(object):
    def __init__(
        self, evaluation_config: EvaluationConfig, prediction_dir: str, metrics: Any
    ):
        """

        Args:
            evaluation_config:
            prediction_dir:
            metrics:
        """
        self.evaluation_config = evaluation_config
        self.metrics_toCompute = {
            "accuracy": False,
            "squad": False,
            "kv_substitution": False,
            "arithmetic": False,
            "kv_substitution_arithmetic": False,
            "sp_rouge": False,
        }

        experiment_id = prediction_dir.replace("/", "-")

        if "Accuracy" in metrics:
            self.metrics_toCompute["accuracy"] = True
            self.metric = load("accuracy", experiment_id=experiment_id)

        if "Squad" in metrics:
            self.metrics_toCompute["squad"] = True
            self.metric = load("squad", experiment_id=experiment_id)

        if "sp_rouge" in metrics:
            self.metrics_toCompute["sp_rouge"] = True
            self.metric = SPRougeScorer(self.evaluation_config)

    def add_batch(self, batchOf_evalInfo: Dict):
        """
        Add batch to scorer

        Args:
            batchOf_evalInfo:

        Returns:

        """
        if self.metrics_toCompute["accuracy"]:

            # Hot fix to account for DomainNet setup where bandage lbl is shared between tool and office and nail lbl is shared between tool and office when we shift the labels and share a classifier head between all tasks.
            # Bandage is idx 62 in tool and idx 159 in office. Nail is idx 73 in tool and idx 173 in office. We force the predicted choice to be correct if the predicted choice is originally wrong, but should be correct after accounting for this mapping.
            if self.evaluation_config.get_datasetConfig().shift_lbls:
                corrected_predictedChoice = []
                for predicted_choice, lbl in zip(
                    batchOf_evalInfo["predicted_choice"], batchOf_evalInfo["lbl"]
                ):
                    if predicted_choice == 62 and lbl == 159:
                        corrected_predictedChoice.append(159)
                    elif predicted_choice == 159 and lbl == 62:
                        corrected_predictedChoice.append(62)
                    elif predicted_choice == 73 and lbl == 173:
                        corrected_predictedChoice.append(173)
                    elif predicted_choice == 173 and lbl == 73:
                        corrected_predictedChoice.append(73)
                    else:
                        corrected_predictedChoice.append(predicted_choice)
                batchOf_evalInfo["predicted_choice"] = corrected_predictedChoice

            self.metric.add_batch(
                predictions=batchOf_evalInfo["predicted_choice"],
                references=batchOf_evalInfo["lbl"],
            )

        if self.metrics_toCompute["squad"]:
            # Have to format the answer correctly for record since record
            # also has an answer key which promptsource requires and cannot be overwritten
            if self.evaluation_config.get_datasetConfig().dataset == "record":
                converted_answers = convert_dictOfLists_to_listOfDicts(
                    {
                        "text": batchOf_evalInfo["text"],
                        "answer_start": batchOf_evalInfo["answer_start"],
                    }
                )
                for answer in converted_answers:
                    answer["text"] = answer["text"]
                    answer["answer_start"] = answer["answer_start"]
                batchOf_evalInfo["answers"] = converted_answers

            self.metric.add_batch(
                predictions=convert_dictOfLists_to_listOfDicts(
                    {
                        "id": batchOf_evalInfo["id"],
                        "prediction_text": batchOf_evalInfo["prediction_text"],
                    }
                ),
                references=convert_dictOfLists_to_listOfDicts(
                    {
                        "id": batchOf_evalInfo["id"],
                        "answers": batchOf_evalInfo["answers"],
                    }
                ),
            )

        if self.metrics_toCompute["sp_rouge"]:
            self.metric.add_batch(
                batchOf_evalInfo["prediction_text"], batchOf_evalInfo["summary"]
            )

    def get_score(self) -> Dict:
        """
        Get the final score

        Returns:
            score
        """
        score = {}

        if self.metrics_toCompute["squad"] or self.metrics_toCompute["sp_rouge"]:
            metric_scores = self.metric.compute()
            # Scale SQUAD metrics to be between 0 and 1
            for metric, value in metric_scores.items():
                metric_scores[metric] = value / 100
            score.update(metric_scores)

        if self.metrics_toCompute["accuracy"]:
            score.update(self.metric.compute())

        for key, value in score.items():
            score[key] = float("%.3f" % value)

        score["average"] = get_average(score.values())

        return score
