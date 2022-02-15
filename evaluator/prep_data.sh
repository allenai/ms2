#!/bin/bash

tokenizer=facebook/bart-base
max_length=500

# TRAINING SET
name=training
input_file=/net/nfs2.corp/s2-research/lucyw/ms2/ms2_data/${name}_reviews.jsonl
intermediate_file=/net/nfs2.corp/s2-research/lucyw/ms2/ms2_data_intermediate/${name}.jsonl
output_file=/net/nfs2.corp/s2-research/lucyw/ms2/ms2_data_processed/${name}.jsonl

python scripts/modeling/summarizer_input_prep.py --input $input_file --intermediate $intermediate_file --output $output_file \
 --tokenizer $tokenizer --max_length $max_length

 # VALIDATION SET
name=validation
input_file=/net/nfs2.corp/s2-research/lucyw/ms2/ms2_data/${name}_reviews.jsonl
intermediate_file=/net/nfs2.corp/s2-research/lucyw/ms2/ms2_data_intermediate/${name}.jsonl
output_file=/net/nfs2.corp/s2-research/lucyw/ms2/ms2_data_processed/${name}.jsonl

python scripts/modeling/summarizer_input_prep.py --input $input_file --intermediate $intermediate_file --output $output_file \
 --tokenizer $tokenizer --max_length $max_length

# TESTING SET
name=testing
input_file=/net/nfs2.corp/s2-research/lucyw/ms2/ms2_data/${name}_reviews.jsonl
intermediate_file=/net/nfs2.corp/s2-research/lucyw/ms2/ms2_data_intermediate/${name}.jsonl
output_file=/net/nfs2.corp/s2-research/lucyw/ms2/ms2_data_processed/${name}.jsonl

python scripts/modeling/summarizer_input_prep.py --input $input_file --intermediate $intermediate_file --output $output_file \
 --tokenizer $tokenizer --max_length $max_length

