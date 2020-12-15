### Repository for SciMON Code

SciMON is a dataset containing medical systematic reviews, their constituent studies, and a large amount of related markup. This repository contains code for attempting to produce summaries from this data.

All commands are assumed to be run in the same terminal session, so variables such as `PYTHONPATH` are assumed to be carried between components.

### Set Up

You might wish to create a conda env:
```
conda create -n scimon python=3.8
# or conda create -p scimon python=3.8
conda activate scimon
```

You will need to install these packages:
```
pip install -r requirements.txt
wget https://ai2-s2-research.s3-us-west-2.amazonaws.com/longformer/longformer-encdec-base-16384.tar.gz
wget https://ai2-s2-research.s3-us-west-2.amazonaws.com/longformer/longformer-encdec-large-16384.tar.gz
```

### Input Prep

The first step is to convert model inputs for the summarizer. This converts the review structure into tensorized versions of inputs and outputs; either text or table inputs or outputs. The primary versions of interest are the text-to-text version and the table-to-table versions. See [sample.json](sample.json) for an example of the raw inputs.

This will need to be repeated for each subset:
```
input_subset=...
output_reviews_file=...
MAX_LENGTH="--max_length 500"
# Run from either the SciMON root or specify the path of the SciMON repository.
export PYTHONPATH=./
# text-text version
python scripts/modeling/summarizer_input_prep.py --input $input_subset --output $output_reviews_file --tokenizer facebook/bart-base $MAX_LENGTH"
# table-table version
python scripts/modeling/tabular_summarizer_input_prep.py --input $input_subset --output $output_reviews_file --tokenizer facebook/bart-base $MAX_LENGTH"
# text-table version
python scripts/modeling/text_to_table_input_prep.py --input $input_subset --output $output_reviews_file --tokenizer facebook/bart-base $MAX_LENGTH"
# table-text version
python scripts/modeling/table_to_text_summarizer_input.py --input $input_subset --output $output_reviews_file --tokenizer facebook/bart-base $MAX_LENGTH"
```

### Modeling

All model training uses the same script. Run with `--help` for all options. This requires at least one RTX8000 (users of just one will need to adjust GRAD_ACCUM appropriately).
```
training_reviews_file="result of input prep"
validation_reviews_file="result of input prep"
training_root="place to store model artificats"
EPOCHS=8      # more doesn't seem to do much
GRAD_ACCUM=16 # if using 2x RTX8000, otherwise set for batch sizes of 32
MODEL_NAME=   # options are facebook/bart-base, facebook/bart-large, /path/to/longformer/base, /path/to/longformer/large
python scimon/models/transformer_summarizer.py \
    --train $training_reviews_file \
    --val $validation_reviews_file \
    --training_root $training_dir \
    --epochs=$EPOCHS \
    --grad_accum=$GRAD_ACCUM \
    --fp16 \
    --model_name $MODEL_NAME
```

### Decoding

Make predictions via:
```
INPUT=$validation_reviews_file
OUTPUT="well, you want this to go somewhere?"
CHECKPOINT="trained model"
NUM_BEAMS=6
MODEL_NAME="same as in modeling"
python scripts/modeling/decode.py --input $INPUT --output $OUTPUT --checkpoint $CHECKPOINT --num_beams=$NUM_BEAMS --model_name $MODEL_NAME
```

The tabular target settings should have the extra arguments: `--min_length 2 --max_length 10`

### Scoring

For tabular scoring:
```
f="$OUTPUT from above"
python scripts/modeling/f1_scorer.py --input $f --output $f.scores
```

For textual scoring (requires a GPU):
```
f="$OUTPUT from above"
evidence_inference_dir=...
evidence_inference_classifier_params=...
python scripts/modeling/consistency_scorer.py --model_outputs $f --output $f.scores --evidence_inference_dir $evidence_inference_dir --evidence_inference_classifier_params $evidence_inference_params &
```

### Evidence Inference

This section uses a modified version of the evidence inference dataset that discards the comparator. Clone evidence inference fom the [scimon tag](https://github.com/jayded/evidence-inference/releases/tag/scimon). Once installing the requirements.txt file, the models may be trained via:
```
python evidence_inference/models/pipeline.py --params params/sampling_abstracts/bert_pipeline_8samples.json --output_dir $evidence_inference_dir
```
