## Commands (epochs=20, task-appropriate max-samples, per-opt LR policy)

Template (use per-optimizer defaults; append overrides for HF models when needed):
```bash
python -m benchmark.main --task <task> --dataset <dataset> --model <model> \
  --optimizers all --epochs 20 --batch-size 64 --max-samples <N> \
  --lr-policy per_opt --output-dir ./results
```

HuggingFace encoder models (e.g., DistilBERT) – recommended LR overrides:
```bash
--lr-overrides "sgd=5e-4,sgd_momentum=5e-4,rmsprop=3e-5,adagrad=3e-5,adadelta=3e-4,adam=3e-5,adamw=3e-5,radam=3e-5,lion=5e-5,lars=1e-4,lamb=3e-5,adabelief=3e-5,yogi=3e-5,adafactor=5e-5,sam=2e-5,gsam=2e-5"
```

### Image Classification
| Task | Dataset | Model | Command |
|---|---|---|---|
| image_classification | mnist | cnn_scratch | python -m benchmark.main --task image_classification --dataset mnist --model cnn_scratch --optimizers all --epochs 20 --batch-size 128 --max-samples 60000 --lr-policy per_opt --output-dir ./results |
| image_classification | mnist | resnet18 | python -m benchmark.main --task image_classification --dataset mnist --model resnet18 --optimizers all --epochs 20 --batch-size 128 --max-samples 60000 --lr-policy per_opt --output-dir ./results |
| image_classification | cifar10 | cnn_scratch | python -m benchmark.main --task image_classification --dataset cifar10 --model cnn_scratch --optimizers all --epochs 20 --batch-size 128 --max-samples 50000 --lr-policy per_opt --output-dir ./results |
| image_classification | cifar10 | resnet18 | python -m benchmark.main --task image_classification --dataset cifar10 --model resnet18 --optimizers all --epochs 20 --batch-size 128 --max-samples 50000 --lr-policy per_opt --output-dir ./results |

### Semantic Segmentation
| Task | Dataset | Model | Command |
|---|---|---|---|
| semantic_segmentation | oxford_iiit_pet | unet_vanilla | python -m benchmark.main --task semantic_segmentation --dataset oxford_iiit_pet --model unet_vanilla --optimizers all --epochs 20 --batch-size 64 --max-samples 10000 --lr-policy per_opt --output-dir ./results |
| semantic_segmentation | oxford_iiit_pet | deeplabv3_resnet50 | python -m benchmark.main --task semantic_segmentation --dataset oxford_iiit_pet --model deeplabv3_resnet50 --optimizers all --epochs 20 --batch-size 64 --max-samples 10000 --lr-policy per_opt --output-dir ./results |
| semantic_segmentation | ade20k | unet_vanilla | python -m benchmark.main --task semantic_segmentation --dataset ade20k --model unet_vanilla --optimizers all --epochs 20 --batch-size 64 --max-samples 20000 --lr-policy per_opt --output-dir ./results |
| semantic_segmentation | ade20k | deeplabv3_resnet50 | python -m benchmark.main --task semantic_segmentation --dataset ade20k --model deeplabv3_resnet50 --optimizers all --epochs 20 --batch-size 64 --max-samples 20000 --lr-policy per_opt --output-dir ./results |

### Sentiment Analysis
| Task | Dataset | Model | Command |
|---|---|---|---|
| sentiment_analysis | imdb | lstm_sentiment | python -m benchmark.main --task sentiment_analysis --dataset imdb --model lstm_sentiment --optimizers all --epochs 20 --batch-size 64 --max-samples 25000 --lr-policy per_opt --output-dir ./results |
| sentiment_analysis | imdb | distilbert_sentiment | python -m benchmark.main --task sentiment_analysis --dataset imdb --model distilbert_sentiment --optimizers all --epochs 20 --batch-size 16 --max-samples 300000 --lr-policy per_opt --lr-overrides "sgd=5e-4,sgd_momentum=5e-4,rmsprop=3e-5,adagrad=3e-5,adadelta=3e-4,adam=3e-5,adamw=3e-5,radam=3e-5,lion=5e-5,lars=1e-4,lamb=3e-5,adabelief=3e-5,yogi=3e-5,adafactor=5e-5,sam=2e-5,gsam=2e-5" --output-dir ./results |
| sentiment_analysis | sst2 | lstm_sentiment | python -m benchmark.main --task sentiment_analysis --dataset sst2 --model lstm_sentiment --optimizers all --epochs 20 --batch-size 64 --max-samples 60000 --lr-policy per_opt --output-dir ./results |
| sentiment_analysis | sst2 | distilbert_sentiment | python -m benchmark.main --task sentiment_analysis --dataset sst2 --model distilbert_sentiment --optimizers all --epochs 20 --batch-size 16 --max-samples 60000 --lr-policy per_opt --lr-overrides "sgd=5e-4,sgd_momentum=5e-4,rmsprop=3e-5,adagrad=3e-5,adadelta=3e-4,adam=3e-5,adamw=3e-5,radam=3e-5,lion=5e-5,lars=1e-4,lamb=3e-5,adabelief=3e-5,yogi=3e-5,adafactor=5e-5,sam=2e-5,gsam=2e-5" --output-dir ./results |

### Machine Translation
| Task | Dataset | Model | Command |
|---|---|---|---|
| machine_translation | europarl_bilingual | lstm_seq2seq | python -m benchmark.main --task machine_translation --dataset europarl_bilingual --model lstm_seq2seq --optimizers all --epochs 20 --batch-size 64 --max-samples 200000 --lr-policy per_opt --output-dir ./results |
| machine_translation | europarl_bilingual | transformer_seq2seq | python -m benchmark.main --task machine_translation --dataset europarl_bilingual --model transformer_seq2seq --optimizers all --epochs 20 --batch-size 64 --max-samples 200000 --lr-policy per_opt --output-dir ./results |
| machine_translation | iwslt14_en_de | lstm_seq2seq | python -m benchmark.main --task machine_translation --dataset iwslt14_en_de --model lstm_seq2seq --optimizers all --epochs 20 --batch-size 64 --max-samples 100000 --lr-policy per_opt --output-dir ./results |
| machine_translation | iwslt14_en_de | transformer_seq2seq | python -m benchmark.main --task machine_translation --dataset iwslt14_en_de --model transformer_seq2seq --optimizers all --epochs 20 --batch-size 64 --max-samples 100000 --lr-policy per_opt --output-dir ./results |

### NER
| Task | Dataset | Model | Command |
|---|---|---|---|
| ner | conll2003 | lstm_crf_ner | python -m benchmark.main --task ner --dataset conll2003 --model lstm_crf_ner --optimizers all --epochs 20 --batch-size 64 --max-samples 14000 --lr-policy per_opt --output-dir ./results |
| ner | conll2003 | bilstm_crf_charcnn_ner | python -m benchmark.main --task ner --dataset conll2003 --model bilstm_crf_charcnn_ner --optimizers all --epochs 20 --batch-size 64 --max-samples 14000 --lr-policy per_opt --output-dir ./results |
| ner | wikiann_en | lstm_crf_ner | python -m benchmark.main --task ner --dataset wikiann_en --model lstm_crf_ner --optimizers all --epochs 20 --batch-size 64 --max-samples 100000 --lr-policy per_opt --output-dir ./results |
| ner | wikiann_en | bilstm_crf_charcnn_ner | python -m benchmark.main --task ner --dataset wikiann_en --model bilstm_crf_charcnn_ner --optimizers all --epochs 20 --batch-size 64 --max-samples 100000 --lr-policy per_opt --output-dir ./results |

### Text Generation
| Task | Dataset | Model | Command |
|---|---|---|---|
| text_generation | wikitext2 | gpt_small | python -m benchmark.main --task text_generation --dataset wikitext2 --model gpt_small --optimizers all --epochs 20 --batch-size 64 --max-samples 100000 --lr-policy per_opt --output-dir ./results |
| text_generation | wikitext2 | gru_lm | python -m benchmark.main --task text_generation --dataset wikitext2 --model gru_lm --optimizers all --epochs 20 --batch-size 64 --max-samples 100000 --lr-policy per_opt --output-dir ./results |
| text_generation | ptb_text_only | gpt_small | python -m benchmark.main --task text_generation --dataset ptb_text_only --model gpt_small --optimizers all --epochs 20 --batch-size 64 --max-samples 100000 --lr-policy per_opt --output-dir ./results |
| text_generation | ptb_text_only | gru_lm | python -m benchmark.main --task text_generation --dataset ptb_text_only --model gru_lm --optimizers all --epochs 20 --batch-size 64 --max-samples 100000 --lr-policy per_opt --output-dir ./results |

### Text Summarization
| Task | Dataset | Model | Command |
|---|---|---|---|
| text_summarization | cnn_dailymail | bart_small | python -m benchmark.main --task text_summarization --dataset cnn_dailymail --model bart_small --optimizers all --epochs 20 --batch-size 32 --max-samples 200000 --lr-policy per_opt --output-dir ./results |
| text_summarization | cnn_dailymail | tiny_transformer_seq2seq | python -m benchmark.main --task text_summarization --dataset cnn_dailymail --model tiny_transformer_seq2seq --optimizers all --epochs 20 --batch-size 32 --max-samples 200000 --lr-policy per_opt --output-dir ./results |
| text_summarization | aeslc | bart_small | python -m benchmark.main --task text_summarization --dataset aeslc --model bart_small --optimizers all --epochs 20 --batch-size 64 --max-samples 14000 --lr-policy per_opt --output-dir ./results |
| text_summarization | aeslc | tiny_transformer_seq2seq | python -m benchmark.main --task text_summarization --dataset aeslc --model tiny_transformer_seq2seq --optimizers all --epochs 20 --batch-size 64 --max-samples 14000 --lr-policy per_opt --output-dir ./results |

### Question Answering
| Task | Dataset | Model | Command |
|---|---|---|---|
| question_answering | squad_v1 | transformer_qa | python -m benchmark.main --task question_answering --dataset squad_v1 --model transformer_qa --optimizers all --epochs 20 --batch-size 64 --max-samples 80000 --lr-policy per_opt --output-dir ./results |
| question_answering | squad_v1 | bilstm_attention_qa | python -m benchmark.main --task question_answering --dataset squad_v1 --model bilstm_attention_qa --optimizers all --epochs 20 --batch-size 64 --max-samples 80000 --lr-policy per_opt --output-dir ./results |
| question_answering | tweet_qa | transformer_qa | python -m benchmark.main --task question_answering --dataset tweet_qa --model transformer_qa --optimizers all --epochs 20 --batch-size 64 --max-samples 10000 --lr-policy per_opt --output-dir ./results |
| question_answering | tweet_qa | bilstm_attention_qa | python -m benchmark.main --task question_answering --dataset tweet_qa --model bilstm_attention_qa --optimizers all --epochs 20 --batch-size 64 --max-samples 10000 --lr-policy per_opt --output-dir ./results |



