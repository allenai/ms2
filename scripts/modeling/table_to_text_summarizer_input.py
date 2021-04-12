"""Produces instances for the transformer summarizer

From each review, this extracts all possible I/O tuplets that are *not* in the Review Effect statements and:
* turns every reference into a format like | intervention | outcome | evidence sentence | significance class |
* keeping only tuplets like ^^ that have an evidence sentence score above some threshold
* uses the preamble 
* targets the EFFECT statement

"""
import argparse
import functools
import itertools
import json
import logging
import multiprocessing

from dataclasses import asdict
from typing import List

from ms2.utils import (
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
    END_EVIDENCE
)

from summarizer_input_prep import (
    clean_targets,
    extract_summary_parts,
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

def process_review(review: Review, evidence_threshold) -> List[TargetSummary]:
    ret = []
    refs = []
    review_io_parts = set(itertools.product(*review_ios(review)))
    preface, target, _ = extract_summary_parts(review.structured_abstract)
    for ref in review.extract_references():
        if ref.significances is None:
            continue
        for sig in ref.significances:
            if sig.evidence_sentence_score < evidence_threshold:
                continue
            i, o = sig.intervention, sig.outcome
            if (i, o) not in review_io_parts:
                continue
                #raise ValueError('Impossible!')
            clazz = sig_class(sig.classification)
            fake_text = '\n'.join([
                START_REFERENCE,
                START_INTERVENTION + ' ' + i + ' ' + END_INTERVENTION,
                START_OUTCOME + ' ' + o + ' ' + END_OUTCOME,
                START_EVIDENCE + ' ' + sig.evidence_sentence + ' ' + END_EVIDENCE,
                SEP_TOKEN + ' ' + clazz,
                END_REFERENCE,
            ])
            refs.append(TargetReference(
                title_abstract=fake_text,
                full_text=None,
                s2id=ref.s2id,
                s2hash=ref.s2hash,
            ))

    ret.append(TargetSummary(
           preface=START_BACKGROUND + ' ' + preface + ' ' + END_BACKGROUND,
           target_texts=clean_targets([target]),
           review_id=review.docid,
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
    written = 0
    with open(args.input, 'r') as inf, \
        open(args.output, 'w') as of:
        for line in inf:
            if len(line.strip()) == 0:
                continue
            review = Review.from_json(line)
            review_count += 1
            instances = process_review(review, evidence_threshold=args.evidence_sentence_threshold)
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