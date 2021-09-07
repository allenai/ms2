#!/bin/bash

set -e
set -x
PYTHONPATH=$(readlink -e .):$PYTHONPATH
export PYTHONPATH
PYTHON=/home/deyoung.j/local/ms2_repro/bin/python
set -o nounset
set -o pipefail

function ckpt {
    local cmd="$1"
    local name="$2"
    local ckpt_file="$ARTIFACTS/logs/$name.ckpt"
    local partial_ckpt_file="$ARTIFACTS/logs/$name.partial"
    local log_file_base="$ARTIFACTS/logs/$name"
    mkdir -p "$(dirname $ckpt_file)" "$(dirname $log_file_base)"
    if [ -e "$partial_ckpt_file" ] ; then
        cat "$partial_ckpt_file" >> "$partial_ckpt_file".old
    fi
    if [ ! -e "$ckpt_file" ] ; then
        echo "running $name; $cmd"
        echo "$cmd" > "$partial_ckpt_file"
        if [ -e "${log_file_base}.e" ]; then
            mv "${log_file_base}.e" "${log_file_base}.e.old"
        fi
        if [ -e "${log_file_base}.o" ]; then
            mv "${log_file_base}.o" "${log_file_base}.o.old"
        fi
        # shellcheck disable=SC2086
        eval $cmd > >(tee "${log_file_base}.o") 2> >(tee "${log_file_base}.e" >&2) && touch $ckpt_file || (echo "failed $name ; $cmd" ; exit 1)
        #else
        #echo "already ran '$name'; clear '$ckpt_file' to rerun"
        fi
}

ARTIFACTS="/scratch/deyoung.j/ms2_repro/"
OUTPUTS="$ARTIFACTS/outputs"

MAX_LENGTH="--max_length 500"
for subset in training validation testing ; do 
    cmd="srun -p short -t 16:00:00 --mem 24G $PYTHON scripts/modeling/summarizer_input_prep.py --input /scratch/deyoung.j/ms2_repro/ms2_data/${subset}_reviews.jsonl --output $OUTPUTS/text_to_text/${subset}.jsonl  --tokenizer facebook/bart-base $MAX_LENGTH"
    ckpt "$cmd" "text_to_text/$subset"
done

training_reviews_file=$OUTPUTS/text_to_text/training.jsonl
validation_reviews_file=$OUTPUTS/text_to_text/validation.jsonl
testing_reviews_file=$OUTPUTS/text_to_text/testing.jsonl

training_dir=$OUTPUTS/text_to_text/training/bart-base/
EPOCHS=8
GRAD_ACCUM=16
MODEL_NAME='facebook/bart-base'
cmd="srun -p frink --gres gpu:1 --mem=64G --cpus-per-task=16 \
    $PYTHON ms2/models/transformer_summarizer.py \
    --train $training_reviews_file \
    --val $validation_reviews_file \
    --training_root $training_dir \
    --epochs=$EPOCHS \
    --grad_accum=$GRAD_ACCUM \
    --fp16 \
    --model_name $MODEL_NAME \
    --max_num_refs 25"

ckpt "$cmd" "text_to_text/training/bart-base"

NUM_BEAMS=6
CHECKPOINT=$training_dir/_ckpt_epoch_7.ckpt
INPUT=$validation_reviews_file
OUTPUT=$training_dir/decode/validation
mkdir -p $(dirname $OUTPUT)
cmd="srun -p frink --gres gpu:1 --mem 32G --cpus-per-task=8 \
    $PYTHON \
        scripts/modeling/decode.py \
        --input $INPUT --output $OUTPUT \
        --checkpoint $CHECKPOINT \
        --num_beams=$NUM_BEAMS \
        --model_name $MODEL_NAME"
ckpt "$cmd" "text_to_text/training/bart_base/decode/validation" &


INPUT=$training_reviews_file
OUTPUT=$training_dir/decode/training
cmd="srun -p frink --gres gpu:1 --mem 32G --cpus-per-task=8 \
    $PYTHON \
        scripts/modeling/decode.py \
        --input $INPUT --output $OUTPUT \
        --checkpoint $CHECKPOINT \
        --num_beams=$NUM_BEAMS \
        --model_name $MODEL_NAME"
ckpt "$cmd" "text_to_text/training/bart_base/decode/training" &


INPUT=$testing_reviews_file
OUTPUT=$training_dir/decode/testing
cmd="srun -p frink --gres gpu:1 --mem 32G --cpus-per-task=8 \
    $PYTHON \
        scripts/modeling/decode.py \
        --input $INPUT --output $OUTPUT \
        --checkpoint $CHECKPOINT \
        --num_beams=$NUM_BEAMS \
        --model_name $MODEL_NAME"
ckpt "$cmd" "text_to_text/training/bart_base/decode/testing" &

wait

training_dir=$OUTPUTS/text_to_text/training/longformer_base/
# longformer, bart-large
MODEL_NAME=/scratch/deyoung.j/ms2_repro/source_models/longformer-encdec-base-16384
cmd="srun -p frink --gres gpu:1 --mem=64G --cpus-per-task=16 \
    $PYTHON ms2/models/transformer_summarizer.py \
    --train $training_reviews_file \
    --val $validation_reviews_file \
    --training_root $training_dir \
    --epochs=$EPOCHS \
    --grad_accum=$GRAD_ACCUM \
    --fp16 \
    --model_name $MODEL_NAME \
    --max_num_refs 25"

ckpt "$cmd" "text_to_text/training/longformer_base" &

NUM_BEAMS=6
CHECKPOINT=$training_dir/_ckpt_epoch_7.ckpt
INPUT=$validation_reviews_file
OUTPUT=$training_dir/decode/validation
mkdir -p $(dirname $OUTPUT)
cmd="srun -p frink --gres gpu:1 --mem 32G --cpus-per-task=8 \
    $PYTHON \
        scripts/modeling/decode.py \
        --input $INPUT --output $OUTPUT \
        --checkpoint $CHECKPOINT \
        --num_beams=$NUM_BEAMS \
        --model_name $MODEL_NAME"
ckpt "$cmd" "text_to_text/training/longformer_base/decode/validation" &


INPUT=$training_reviews_file
OUTPUT=$training_dir/decode/training
cmd="srun -p frink --gres gpu:1 --mem 32G --cpus-per-task=8 \
    $PYTHON \
        scripts/modeling/decode.py \
        --input $INPUT --output $OUTPUT \
        --checkpoint $CHECKPOINT \
        --num_beams=$NUM_BEAMS \
        --model_name $MODEL_NAME"
ckpt "$cmd" "text_to_text/training/longformer_base/decode/training" &


INPUT=$testing_reviews_file
OUTPUT=$training_dir/decode/testing
cmd="srun -p frink --gres gpu:1 --mem 32G --cpus-per-task=8 \
    $PYTHON \
        scripts/modeling/decode.py \
        --input $INPUT --output $OUTPUT \
        --checkpoint $CHECKPOINT \
        --num_beams=$NUM_BEAMS \
        --model_name $MODEL_NAME"
ckpt "$cmd" "text_to_text/training/longformer_base/decode/testing" &

wait
