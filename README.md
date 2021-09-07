## MS^2: Multi-Document Summarization of Medical Studies

MS^2 is a dataset containing medical systematic reviews, their constituent studies, and a large amount of related markup. This repository contains code for attempting to produce summaries from this data. To find out more about how we created this dataset, please read our [preprint](https://arxiv.org/abs/2104.06486).

This dataset is created as an annotated subset of the Semantic Scholar research corpus. MS^2 is licensed under the following license agreement: [Semantic Scholar API and Dataset License Agreement](http://s2-public-api-prod.us-west-2.elasticbeanstalk.com/corpus/legal/)

All following commands are assumed to be run in the same terminal session, so variables such as `PYTHONPATH` are assumed to be carried between components.

### Set Up

You might wish to create a conda env:
```
conda create -n ms2 python=3.8
# or conda create -p ms2 python=3.8
conda activate ms2
```

You will need to install these packages:
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
wget https://ai2-s2-research.s3-us-west-2.amazonaws.com/longformer/longformer-encdec-base-16384.tar.gz
wget https://ai2-s2-research.s3-us-west-2.amazonaws.com/longformer/longformer-encdec-large-16384.tar.gz
```

### Data & Checkpoints

| File        | Description | sha1        | md5         |
| ----------- | ----------- | ----------- | ----------- |
| [ms_data_2021-04-12.zip](https://ai2-s2-ms2.s3-us-west-2.amazonaws.com/ms_data_2021-04-12.zip)      | MS^2 Dataset Files | 6090fbea | 7cf243af |
| [bart_base_ckpt_7.ckpt](https://ai2-s2-ms2.s3-us-west-2.amazonaws.com/bart_base_ckpt_7.ckpt)        | BART checkpoint | 9698478c | 4a0d5630 |
| [longformer_base_ckpt_7.ckpt](https://ai2-s2-ms2.s3-us-west-2.amazonaws.com/longformer_base_ckpt_7.ckpt)        | Longformer (LED) checkpoint | 327f9f41 | 4558b0d4 |
| [evidence_inference_models.zip](https://ai2-s2-ms2.s3-us-west-2.amazonaws.com/evidence_inference_models.zip)        | EI models | bc7fecdc | 2bc1bdaf |
| [decoded.zip](https://ai2-s2-ms2.s3-us-west-2.amazonaws.com/decoded.zip)      |  | a9e023e2 | 0725f2a4 |
| [decoded_with_scores.zip](https://ai2-s2-ms2.s3-us-west-2.amazonaws.com/decoded_with_scores.zip)      |  | 38715772 | 5808924e |

All files are on AWS S3, so you can also acquire them using the AWS cli, e.g. `aws s3 cp s3://ai2-s2-ms2/ms_data_2021-04-12.zip ./`

[comment]: <> (```)
[comment]: <> (sha1sum ms_data_2021-04-12.zip)
[comment]: <> (6090fbea367c7c52a4c3a9418591792d8dea90e7  ms_data_2021-04-12.zip)
[comment]: <> (md5sum ms_data_2021-04-12.zip)
[comment]: <> (7cf243af2373ad496d948fc73d7dcf31  ms_data_2021-04-12.zip)
[comment]: <> (```)

### Input Prep

The first step is to convert model inputs for the summarizer. This converts the review structure into tensorized versions of inputs and outputs; either text or table inputs or outputs. The primary versions of interest are the text-to-text version and the table-to-table versions. See [sample.json](sample.json) for an example of the raw inputs.

This will need to be repeated for each subset:
```
input_subset=...
output_reviews_file=...
MAX_LENGTH="--max_length 500"
# Run from either the ms2 root or specify the path of the ms2 repository.
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
training_root="place to store model artifacts"
EPOCHS=8      # more doesn't seem to do much
GRAD_ACCUM=16 # if using 2x RTX8000, otherwise set for batch sizes of 32
MODEL_NAME=   # options are facebook/bart-base, facebook/bart-large, /path/to/longformer/base, /path/to/longformer/large
python ms2/models/transformer_summarizer.py \
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

This section uses a modified version of the evidence inference dataset that discards the comparator. Clone evidence inference fom the [ms2 tag](https://github.com/jayded/evidence-inference/releases/tag/ms2). Once installing the requirements.txt file, the models may be trained via:
```
python evidence_inference/models/pipeline.py --params params/sampling_abstracts/bert_pipeline_8samples.json --output_dir $evidence_inference_dir
```

### Citation

If using this dataset, please cite:

```
@misc{deyoung2021ms2,
  title={MS2: Multi-Document Summarization of Medical Studies}, 
  author={Jay DeYoung and Iz Beltagy and Madeleine van Zuylen and Bailey Kuehl and Lucy Lu Wang},
  year={2021},
  eprint={2104.06486},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```
