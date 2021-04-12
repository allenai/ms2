"""Produces instances for the transformer summarizer

Uses textual inputs from the references, 
From each review, this extracts all possible I/O tuplets that are *not* in the Review Effect statements and:
* turns every reference into a format like | intervention | outcome | evidence sentence | significance class |
* keeping only tuplets like ^^ that have an evidence sentence score above some threshold
* turns the preamble into a format like | intervention | outcome |
* turns the target into the literal text of the significance class

"""
import argparse
import functools
import itertools
import json
import logging
import multiprocessing
import re

from collections import defaultdict
from dataclasses import asdict
from typing import Dict, List, Set, Tuple

from ms2.utils import (
    get_tokenizer,
    Review,
    TargetSummary,
    START_BACKGROUND,
    END_BACKGROUND,
    START_INTERVENTION,
    END_INTERVENTION,
    START_OUTCOME,
    END_OUTCOME,
)

from summarizer_input_prep import (
    select_reference_from_study,
    process_reference,
    tokenize_target_summary,
    valid_review,
)

from tabular_summarizer_input_prep import (
    review_ios,
    sig_class,
)

NUM_PROCS = 4
INTERVENTION_RE = START_INTERVENTION + '(.*?)' + END_INTERVENTION
OUTCOME_RE = START_OUTCOME + '(.*?)' + END_OUTCOME

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

def process_review(review: Review) -> List[TargetSummary]:
    ret = []
    review_io_parts = set(itertools.product(*review_ios(review)))

    selected_study_references = map(select_reference_from_study, review.included_studies)
    processed_references = map(process_reference, selected_study_references)
    refs = list(filter(lambda x: x is not None, processed_references))

    for sig in review.significances:
        i, o = sig.intervention, sig.outcome
        if (i, o) not in review_io_parts:
            continue
        clazz = sig_class(sig.classification)
        if len(refs) > 0:
            ret.append(TargetSummary(
                preface='\n'.join([
                    START_BACKGROUND,
                    START_INTERVENTION + ' ' + i + ' ' + END_INTERVENTION,
                    START_OUTCOME + ' ' + o + ' '  + END_OUTCOME,
                    END_BACKGROUND,
                ]),
                target_texts=[clazz],
                review_id=review.docid + '_int_' + i + '_out_' + o,
                references=refs,
                s2id=review.s2id,
                s2hash=review.s2hash
            ))
    return ret

def main():
    parser = argparse.ArgumentParser(description='Convert Reviews into TargetSummary objects')
    parser.add_argument('--input', required=True, help='jsonl serialized input reviews')
    parser.add_argument('--output', required=True, help='file for jsonl serialized output targets')
    parser.add_argument('--tokenizer', required=True, help='tokenizer type, e.g. BART')
    parser.add_argument('--max_length', type=int, default=None, required=False, help='truncate sequence lengths?')
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.tokenizer)
    review_count = 0
    written = 0
    with open(args.input, 'r') as inf, \
        open(args.output, 'w') as of:
        for line in inf:
            if len(line.strip()) == 0:
                continue
            review = Review.from_json(line)
            review_count += 1
            instances = process_review(review)
            instances = list(filter(valid_review, instances))
            tensorized_reviews = map(functools.partial(tokenize_target_summary, tokenizer=tokenizer, max_length=args.max_length), instances)
            tensorized_reviews = list(tensorized_reviews)
            for review in tensorized_reviews:
                of.write(json.dumps(asdict(review)))
                of.write('\n')
                written += 1
    assert written > 0

if __name__ == '__main__':
    main()

