from typing import Dict, Any


class Evaluator:
    """
    Simple wrapper in case we add additional evaluation modes or test-only runs.
    Currently unused; kept for API completeness.
    """
    def __init__(self, task):
        self.task = task

    def evaluate(self, model, dataloader) -> Dict[str, Any]:
        raise NotImplementedError("Use Trainer._validate for now.")


