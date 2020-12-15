import argparse
import functools
import itertools
import json
import logging
import multiprocessing
import os
import re
import shutil

from dataclasses import asdict, replace
from typing import List, Optional, Set, Tuple

import tqdm

from scimon.data.munge import fields_re, spaces_re
from scimon.utils import (
    get_tokenizer,
    Review,
    Study,
    Reference,
    TargetReference,
    TargetSummary,
    EXTRA_TOKENS,
    START_BACKGROUND,
    END_BACKGROUND,
    START_REFERENCE,
    END_REFERENCE,
    SEP_TOKEN,
)

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

NUM_PROCS = 8

# fields that begin a review but we aren't attempting to generate
PREAMBLE_FIELDS = set([
    'BACKGROUND',
    'GOAL',
])
# fields that we exclude both generation and targeting - 
# either because they are out of scope, or too detailed, or useless
EXCLUDED_FIELDS = set([
    'RESULT',
    'METHODS',
    'ETC',
    'DETAILED_FINDINGS', 'RECOMMENDATION', 'FURTHER_STUDY', 'EVIDENCE_QUALITY'
])
# fields we want to generate
# eventually will include 'EVIDENCE_QUALITY', potentially 'FURTHER_STUDY'
TARGET_FIELDS = set([
    'CONCLUSION',
    'EFFECT',
])

def extract_summary_parts(summary_parts: List[Tuple[str, str]]) -> Tuple[str, str, List[str]]:
    # TODO(jayd) filter for results-like, preamble-like, conclusions-line
    preambles = []
    targets = []
    unknown_fields = []

    for field, value, _ in summary_parts:
        if field in PREAMBLE_FIELDS:
            preambles.append(value)
        elif field in TARGET_FIELDS:
            targets.append(value)
        elif field in EXCLUDED_FIELDS:
            pass
        else:
            unknown_fields.append(field)

    return '\n'.join(preambles), '\n'.join(targets), unknown_fields

def select_reference_from_study(study: Study) -> Optional[Reference]:
    if len(study.references) == 1:
        return study.references[0]
    else:
        have_abstract = list(filter(lambda x: x.abstract is not None, study.references))
        have_abstract_and_body = list(filter(lambda x: x.content is not None, have_abstract))
        if len(have_abstract_and_body) > 0:
            # TODO is there a better choice?
            return have_abstract_and_body[0]
        elif len(have_abstract) > 0:
            return have_abstract[0]
        else:
            return None

def process_reference(reference: Reference) -> Optional[TargetReference]:
    if reference.abstract is None:
        return None
    title = reference.title.strip()
    abstract = reference.abstract.strip()
    title_abstract = START_REFERENCE + ' ' + title + ' ' + SEP_TOKEN + ' ' + abstract + ' ' + END_REFERENCE
    return TargetReference(
        title_abstract=title_abstract,
        full_text=None,
        s2id=reference.s2id,
        s2hash=reference.s2hash,
    )

def process_review(review: Review, max_refs: int, tokenizer) -> Tuple[Optional[TargetSummary], Set[str]]:
    input_text = None
    target_texts = []
    summaries = []
    if review.structured_abstract is not None and len(review.structured_abstract) > 1:
        abs_preamble, abs_target, abs_unknown_fields = extract_summary_parts(review.structured_abstract)
        if len(abs_preamble.strip()) == 0:
            return None
        abs_preamble = START_BACKGROUND + ' ' + abs_preamble + ' ' + END_BACKGROUND
    else:
        abs_preamble, abs_target, abs_unknown_fields = None, None, []
    if review.structured_summary is not None and len(review.structured_summary) > 1:
        sum_preamble, sum_target, sum_unknown_fields = extract_summary_parts(review.structured_summary)
        sum_preamble = START_BACKGROUND + ' ' + sum_preamble + ' ' + END_BACKGROUND
    else:
        sum_preamble, sum_target, sum_unknown_fields = None, None, []

    unknown_fields = set(abs_unknown_fields) or set(sum_unknown_fields)
    if len(unknown_fields) > 0:
        logging.info(f'For review {review.docid}, found unknown fields {unknown_fields} in abstract and summary!')

    if abs_target is None and sum_target is None:
        return None, unknown_fields

    if abs_target is not None and len(abs_target) > 0:
        target_texts.append(abs_target)
    if sum_target is not None and len(sum_target) > 0:
        target_texts.append(sum_target)

    if abs_preamble is not None:
        input_text = abs_preamble
    elif sum_preamble is not None:
        input_text = sum_preamble
    else:
        input_text = ''

    if len(input_text.strip()) == 0:
        return None, set()

    selected_study_references = map(select_reference_from_study, review.included_studies)
    processed_references = map(process_reference, selected_study_references)
    references = list(filter(lambda x: x is not None, processed_references))

    if max_refs is not None and len(references) > max_refs:
        logging.info('Truncating review {} references from {} to {}'.format(review.docid, len(references), max_refs))
        references = references[:max_refs]
    target_texts = clean_targets(target_texts)

    return TargetSummary(
        preface=input_text.strip(),
        target_texts=target_texts,
        review_id=review.docid,
        references=references,
        s2id=review.s2id,
        s2hash=review.s2hash,
    ), unknown_fields

def clean_targets(target_texts: List[str]) -> List[str]:
    # remove fancy markers from the target
    for elem in EXTRA_TOKENS:
        cleaned_targets = []
        for t in target_texts:
            cleaned_targets.append(t.replace(elem, ''))
        target_texts = cleaned_targets
    # remove standard section markers from the start of the 
    cleaned_targets = []
    for t in target_texts:
        beginning, end = t[:50], t[50:]
        beginning = re.sub(fields_re, '', beginning)
        beginning = re.sub(spaces_re, ' ', beginning)
        cleaned_targets.append(beginning + end)
    return cleaned_targets

def tokenize_target_summary(summary: TargetSummary, tokenizer, max_length: Optional[int]) -> TargetSummary:
    end_reference = tokenizer.encode(END_REFERENCE, add_special_tokens=False)[0]
    def tokenize_target_reference(target_reference: TargetReference) -> TargetReference:
        title_abstract = tokenizer.encode(
            target_reference.title_abstract,
            add_special_tokens=False,
            truncation=max_length is not None,
            max_length=max_length
        )
        if title_abstract[-1] != end_reference:
            title_abstract = title_abstract + [end_reference]
        return replace(target_reference,
            title_abstract=title_abstract,
            full_text=tokenizer.encode(
                target_reference.full_text,
                add_special_tokens=False,
                truncation=max_length is not None, max_length=max_length
            ) if target_reference.full_text is not None else None,
        )
    
    end_background = tokenizer.encode(END_BACKGROUND, add_special_tokens=False)[0]
    if len(summary.preface) > 0:
        preface = tokenizer.encode(summary.preface,
                                   truncation=max_length is not None,
                                   max_length=max_length,
                                   add_special_tokens=False)
        if preface[-1] != end_background:
            preface = preface + [end_background]
    else:
        preface = None

    target_texts = []
    eos_token = tokenizer.encode(tokenizer._eos_token.content, add_special_tokens=False)[0]
    for target_text in summary.target_texts:
        if len(target_text) == 0:
            continue
        target_text = tokenizer.encode(
            target_text,
            add_special_tokens=True,
            truncation=max_length is not None, max_length=max_length
        )
        if target_text[-1] != eos_token:
            target_text = target_text + [eos_token]
        target_texts.append(target_text)

    return replace(summary,
        preface=preface,
        target_texts=target_texts,
        references=list(map(tokenize_target_reference, summary.references)),
    )

def valid_review(summary: TargetSummary) -> bool:
    if summary is None:
        return False
    if len(summary.references) == 0:
        return False
    if len(summary.target_texts) == 0:
        return False
    return True

def total_reference_lengths(summary: TargetSummary) -> int:
    return sum((len(ref.title_abstract) for ref in summary.references))

def total_decoding_lengths(summary: TargetSummary) -> int:
    preface_length = len(summary.preface) if summary.preface is not None else 0
    target_lengths = max(map(len, summary.target_texts))
    return preface_length + target_lengths

def main():
    parser = argparse.ArgumentParser(description='Convert Reviews into TargetSummary objects')
    parser.add_argument('--input', required=True, help='jsonl serialized input reviews')
    parser.add_argument('--output', required=True, help='file for jsonl serialized output targets')
    parser.add_argument('--tokenizer_save', required=False, default=None, help='Where should we save the tokenizer to?')
    parser.add_argument('--tokenizer', required=True, help='tokenizer type, e.g. BART')
    parser.add_argument('--max_length', type=int, default=None, required=False, help='truncate sequence lengths?')
    parser.add_argument('--max_refs', type=int, default=None, required=False, help='truncate number of included refs?')
    args = parser.parse_args()

    # TODO(jayd) assign ids to these elements!
    tokenizer = get_tokenizer(args.tokenizer)
    if args.tokenizer_save is not None:
        logging.info(f'Saving tokenizer with extended vocab to {args.tokenizer_save}')
        tokenizer.save_pretrained(args.tokenizer_save)
        # workaround for what seems to be a huggingface bug
        likely_misnamed_tokenizer_config = os.path.join(args.tokenizer_save, 'tokenizer_config.json')
        if os.path.exists(likely_misnamed_tokenizer_config):
            shutil.copyfile(likely_misnamed_tokenizer_config, os.path.join(args.tokenizer_save, 'config.json'))
    reviews = Review.read_reviews(args.input)
    logging.info(f'Loaded {len(reviews)}')
    with multiprocessing.Pool(processes=NUM_PROCS) as p:
        logging.info('Processing reviews')
        processed = p.imap(functools.partial(process_review, max_refs=args.max_refs, tokenizer=tokenizer), reviews, chunksize=1)
        processed = filter(lambda x: x is not None, processed)
        processed = list(processed)
        target_reviews, unknown_fields = zip(*processed)
        all_unknown_fields = set(itertools.chain.from_iterable(unknown_fields))
        if len(all_unknown_fields) > 0:
            logging.info(f'Unable to process fields {all_unknown_fields}')
        non_empty_reviews = list(filter(valid_review, target_reviews))
        logging.info('Tensorizing reviews')
        tensorized_reviews = p.imap(functools.partial(tokenize_target_summary, tokenizer=tokenizer, max_length=args.max_length), non_empty_reviews, chunksize=1)
        tensorized_reviews = list(tensorized_reviews)
        logging.info(f'After processing, a total of {len(non_empty_reviews)} reviews are left, with {len(tensorized_reviews)} reviews for input')
        review_target_lengths = list(p.map(total_decoding_lengths, tensorized_reviews))
        review_reference_lengths = list(p.map(total_reference_lengths, tensorized_reviews))
    total_lengths = [sum(x) for x in zip(review_reference_lengths, review_target_lengths)]
    min_length = min(total_lengths)
    max_length = max(total_lengths)
    avg_length = sum(total_lengths) / len(total_lengths)
    logging.info(f'Input/Output lengths are a minimum of {min_length} wordpieces long, maximum of {max_length} wordpieces, and average of {avg_length} wordpieces.')
    with open(args.output, 'w') as of:
        for review in tqdm.tqdm(tensorized_reviews):
            of.write(json.dumps(asdict(review)))
            of.write('\n')

if __name__ == '__main__':
    main()