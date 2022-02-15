import argparse
import itertools
import json
import math
import os
import re

import torch

from ms2.models.evidence_inference_models import initialize_models
from ms2.models.utils import rouge_scores
from ms2.evaluation.utils import clean, entailment_scores
from ms2.utils import get_tokenizer


def main():
    parser = argparse.ArgumentParser(description='Score model outputs based on a consistency score between target and prediction')
    parser.add_argument('--model_outputs', required=True, help='json file of model outputs with "target", "generated", and "preamble" fields')
    parser.add_argument('--evidence_inference_dir', required=True, help='Directory containing trained evidence inference models')
    parser.add_argument('--evidence_inference_classifier_params', required=True, help='Params to initialize evidence inference models')
    parser.add_argument('--unconditioned_classifier', action='store_true', help='Use an unconditioned evidence inference classifier')
    parser.add_argument('--output', required=True, help='Output file for scores')
    args = parser.parse_args()
    
    with open(args.model_outputs, 'r') as inf:
        outputs = [json.loads(line) for line in inf]
        generated = [x['generated'] for x in outputs]
        targets = [x['target'] for x in outputs]
        preambles = [x['preamble'] for x in outputs]
    generated = list(map(clean, generated))
    targets = list(map(clean, targets))
    
    # rouge scoring
    tokenizer = get_tokenizer('facebook/bart-base')
    rouge_results = rouge_scores([[x] for x in generated], [[x] for x in targets], tokenizer, use_aggregator=True)
    print('Rouge')
    print(rouge_results)

    # evidence inference scoring
    with open(args.evidence_inference_classifier_params, 'r') as inf:
        params = json.loads(inf.read())
    _, evidence_inference_classifier, _, _, _, evidence_inference_tokenizer = initialize_models(params)
    if args.unconditioned_classifier:
        classifier_file = os.path.join(args.evidence_inference_dir, 'unconditioned_evidence_classifier', 'unconditioned_evidence_classifier.pt')
    else:
        classifier_file = os.path.join(args.evidence_inference_dir, 'evidence_classifier', 'evidence_classifier.pt')
    #evidence_inference_classifier.load_state_dict(torch.load(classifier_file))
    # pooler parameters are added by default in an older transformers, so we have to ignore that those are uninitialized.
    evidence_inference_classifier.load_state_dict(torch.load(classifier_file), strict=False)
    evidence_inference_classifier.cuda()

    entailment_results = entailment_scores(evidence_inference_classifier, evidence_inference_tokenizer, generated, targets, preambles, use_ios=not args.unconditioned_classifier)
    print('entailment')
    print(entailment_results)
    
    assert args.output != args.model_outputs
    with open(args.output, 'w') as of:
        of.write('rouge\n')
        of.write(json.dumps(rouge_results))
        of.write('\n\n')
        of.write('entailment\n')
        of.write(json.dumps(entailment_results))
        of.write('\n')


if __name__ == '__main__':
    main()