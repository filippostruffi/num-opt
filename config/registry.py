from typing import Callable, Dict, Any, Tuple, List, Iterable

import torch

# Registries
TASK_REGISTRY: Dict[str, Any] = {}
DATASET_REGISTRY: Dict[str, Callable[..., Tuple[Any, Any]]] = {}
MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_task(name: str):
    def _decorator(cls):
        TASK_REGISTRY[name] = cls
        return cls
    return _decorator


def register_dataset(name: str):
    def _decorator(builder: Callable[..., Tuple[Any, Any]]):
        DATASET_REGISTRY[name] = builder
        return builder
    return _decorator


def register_model(name: str):
    def _decorator(builder: Callable[..., Any]):
        MODEL_REGISTRY[name] = builder
        return builder
    return _decorator


# Optimizers
def build_optimizers_from_names(params: Iterable[torch.nn.Parameter], names: List[str], lr):
    from benchmark.optimizers.factory import OPTIMIZER_REGISTRY  # local import to avoid cycles
    optimizers = []
    for name in names:
        name_lower = name.lower()
        if name_lower not in OPTIMIZER_REGISTRY:
            raise ValueError(f"Unknown optimizer: {name}. Available: {list(OPTIMIZER_REGISTRY.keys())}")
        # Allow lr to be a mapping (per-optimizer) or a scalar
        if isinstance(lr, dict):
            used_lr = lr.get(name_lower, lr.get("*", 1e-3))
        else:
            used_lr = lr
        optimizers.append(OPTIMIZER_REGISTRY[name_lower](params, lr=used_lr))
    return optimizers


def ensure_imports() -> None:
    # Import all task modules (registration happens via decorators)
    import benchmark.tasks.image_classification  # noqa: F401
    import benchmark.tasks.semantic_segmentation  # noqa: F401
    import benchmark.tasks.sentiment_analysis  # noqa: F401
    import benchmark.tasks.machine_translation  # noqa: F401
    import benchmark.tasks.ner  # noqa: F401
    import benchmark.tasks.text_generation  # noqa: F401
    import benchmark.tasks.text_summarization  # noqa: F401
    import benchmark.tasks.question_answering  # noqa: F401

    # Import datasets (at least one per task)
    # image
    import benchmark.datasets.image_classification.mnist  # noqa: F401
    import benchmark.datasets.image_classification.cifar10  # noqa: F401
    # segmentation
    import benchmark.datasets.segmentation.oxford_iiit_pet  # noqa: F401
    import benchmark.datasets.segmentation.ade20k  # noqa: F401
    # sentiment
    import benchmark.datasets.sentiment.imdb  # noqa: F401
    import benchmark.datasets.sentiment.sst2  # noqa: F401
    # translation
    import benchmark.datasets.machine_translation.europarl_bilingual  # noqa: F401
    import benchmark.datasets.machine_translation.iwslt14_en_de  # noqa: F401
    # ner
    import benchmark.datasets.ner.conll2003  # noqa: F401
    import benchmark.datasets.ner.wikiann_en  # noqa: F401
    # generation
    import benchmark.datasets.generation.wikitext2  # noqa: F401
    import benchmark.datasets.generation.ptb_text_only_hf  # noqa: F401
    # summarization
    import benchmark.datasets.summarization.cnn_dailymail  # noqa: F401
    import benchmark.datasets.summarization.aeslc  # noqa: F401
    # qa
    import benchmark.datasets.qa.squad_v1  # noqa: F401
    import benchmark.datasets.qa.tweet_qa  # noqa: F401
    # removed unimplemented QA datasets

    # Import models (at least required ones)
    import benchmark.models.image.cnn_scratch  # noqa: F401
    import benchmark.models.image.resnet18  # noqa: F401
    import benchmark.models.segmentation.unet  # noqa: F401
    import benchmark.models.segmentation.deeplabv3_resnet50  # noqa: F401
    import benchmark.models.sentiment.lstm_sentiment  # noqa: F401
    import benchmark.models.sentiment.distilbert_sentiment  # noqa: F401
    import benchmark.models.translation.transformer_seq2seq  # noqa: F401
    import benchmark.models.translation.lstm_seq2seq  # noqa: F401
    import benchmark.models.ner.lstm_crf_ner  # noqa: F401
    import benchmark.models.ner.lstm_crf_charcnn_ner  # noqa: F401
    import benchmark.models.generation.gpt_small  # noqa: F401
    import benchmark.models.generation.gru_lm  # noqa: F401
    import benchmark.models.summarization.bart_small  # noqa: F401
    import benchmark.models.summarization.tiny_transformer_seq2seq  # noqa: F401
    import benchmark.models.qa.transformer_qa  # noqa: F401
    import benchmark.models.qa.bilstm_attention_qa  # noqa: F401


