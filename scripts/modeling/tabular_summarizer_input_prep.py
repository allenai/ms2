"""Produces instances for the transformer summarizer

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

from scimon.utils import (
    get_tokenizer,
    Review,
    TargetReference,
    TargetSummary,
    START_BACKGROUND,
    END_BACKGROUND,
    START_INTERVENTION,
    END_INTERVENTION,
    START_OUTCOME,
    END_OUTCOME,
    START_REFERENCE,
    END_REFERENCE,
    SEP_TOKEN,
    START_EVIDENCE,
    END_EVIDENCE,
    EXTRA_TOKENS
)

from summarizer_input_prep import (
    TARGET_FIELDS,
    tokenize_target_summary,
    valid_review,
)

NUM_PROCS = 4
INTERVENTION_RE = START_INTERVENTION + '(.*?)' + END_INTERVENTION
OUTCOME_RE = START_OUTCOME + '(.*?)' + END_OUTCOME

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

def clean_str(s):
    for elem in EXTRA_TOKENS:
        s = s.replace(elem, '')
        s = s.replace('  ', ' ')
    return s

def extract_non_target_parts(summary_parts: List[Tuple[str, str]]) -> Set[str]:
    ret = set()
    for field, value, _ in summary_parts:
        if field not in TARGET_FIELDS and field not in {'FURTHER_STUDY', 'RECOMMENDATION', 'EVIDENCE_QUALITY', 'DETAILED_FINDINGS', 'RESULT'}:
            ret.add(value)
    return ret

def review_ios(review: Review) -> Tuple[Set[str], Set[str]]:
    assert review.structured_abstract is not None and len(review.structured_abstract) > 0
    non_summary_inputs = extract_non_target_parts(review.structured_abstract)
    non_summary_inputs = '\n'.join(non_summary_inputs)
    intervention_groups = list(map(str.strip, map(clean_str, re.findall(INTERVENTION_RE, non_summary_inputs))))
    outcome_groups = list(map(str.strip, map(clean_str, re.findall(OUTCOME_RE, non_summary_inputs))))
    return intervention_groups, outcome_groups

def sig_class(dist: Dict[str, float]) -> str:
    best_score = float('-inf')
    best_class = None
    for clazz, score in dist.items():
        if score > best_score:
            best_score = score
            best_class = clazz
    return best_class

def process_review(review: Review, evidence_threshold) -> List[TargetSummary]:
    ret = []
    ref_opinions = defaultdict(list)
    review_io_parts = set(itertools.product(*review_ios(review)))
    for ref in review.extract_references():
        if ref.significances is None:
            continue
        for sig in ref.significances:
            if sig.evidence_sentence_score < evidence_threshold:
                continue
            i, o = sig.intervention, sig.outcome
            if (i, o) not in review_io_parts:
                continue
            clazz = sig_class(sig.classification)
            fake_text = '\n'.join([
                START_REFERENCE,
                START_INTERVENTION + ' ' + i + ' ' + END_INTERVENTION,
                START_OUTCOME + ' ' + o + ' ' + END_OUTCOME,
                START_EVIDENCE + ' ' + sig.evidence_sentence + ' ' + END_EVIDENCE,
                SEP_TOKEN + ' ' + clazz,
                END_REFERENCE,
            ])
            ref_opinions[(i,o)].append(TargetReference(
                title_abstract=fake_text,
                s2id=ref.s2id,
                s2hash=ref.s2hash,
                full_text=None,
            ))

    for sig in review.significances:
       i, o = sig.intervention, sig.outcome
       if (i, o) not in review_io_parts:
           continue
       clazz = sig_class(sig.classification)
       refs = ref_opinions[(i, o)]
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
    parser.add_argument('--evidence_sentence_threshold', default=0.0, type=float, help='Evidence sentence score minimum thresholds')
    parser.add_argument('--tokenizer', required=True, help='tokenizer type, e.g. BART')
    parser.add_argument('--max_length', type=int, default=None, required=False, help='truncate sequence lengths?')
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.tokenizer)
    review_count = 0
    with open(args.input, 'r') as inf, \
        open(args.output, 'w') as of:
        for line in inf:
            if len(line.strip()) == 0:
                continue
            review = Review.from_json(line)
            instances = process_review(review, evidence_threshold=args.evidence_sentence_threshold)
            instances = list(filter(valid_review, instances))
            tensorized_reviews = map(functools.partial(tokenize_target_summary, tokenizer=tokenizer, max_length=args.max_length), instances)
            tensorized_reviews = list(tensorized_reviews)
            for review in tensorized_reviews:
                of.write(json.dumps(asdict(review)))
                of.write('\n')
                review_count += 1
        logging.info(f'Wrote {review_count} instances')
        assert review_count > 0

if __name__ == '__main__':
    main()
