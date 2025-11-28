from typing import Any
from benchmark.config.registry import register_model


@register_model("distilbert_sentiment")
def build_model(task: Any, dataset: Any):
    """
    Build a DistilBERT-based sentiment classifier via HuggingFace Transformers.
    Requires `transformers` to be installed and internet or cached weights/tokenizer.
    Populates task.runtime_objects['hf_tokenizer'] for the task collate_fn to use.
    """
    try:
        from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
    except Exception as e:
        raise NotImplementedError(
            "distilbert_sentiment requires the 'transformers' package. "
            "Add `transformers>=4.40.0` to requirements and reinstall."
        ) from e

    model_name = "distilbert-base-uncased"
    num_labels = 2

    # Try pretrained; if that fails (offline), try local cache; if still fails, raise cleanly
    tokenizer = None
    model = None
    last_err = None
    for kwargs in ({"local_files_only": False}, {"local_files_only": True}):
        try:
            tok = AutoTokenizer.from_pretrained(model_name, **kwargs)
            cfg = AutoConfig.from_pretrained(model_name, num_labels=num_labels, **kwargs)
            mdl = AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg, **kwargs)
            tokenizer = tok
            model = mdl
            break
        except Exception as ex:
            last_err = ex
            continue

    if tokenizer is None or model is None:
        # As a last resort, try to instantiate from config only (random init)
        try:
            cfg = AutoConfig.from_pretrained(model_name, num_labels=num_labels, local_files_only=True)
            from transformers import AutoModelForSequenceClassification as AM
            model = AM.from_config(cfg)
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        except Exception as ex:
            raise NotImplementedError(
                "Unable to load DistilBERT weights/tokenizer (offline and no cache). "
                "Pre-download `distilbert-base-uncased` into HF cache (set HF_DATASETS_CACHE/TRANSFORMERS_CACHE) "
                "or run with internet access."
            ) from (ex if ex else last_err)

    # Expose tokenizer to the task so its collate_fn can produce tokenized inputs
    task.runtime_objects["hf_tokenizer"] = tokenizer
    task.runtime_objects["hf_num_labels"] = num_labels
    return model


