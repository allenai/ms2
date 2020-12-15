import itertools
import random

from collections import namedtuple
from dataclasses import replace
from typing import List, Tuple

import torch

from torch.utils.data import Dataset

from scimon.utils import TargetReference, TargetSummary
from scimon.models.utils import pad_tensors

random.seed(12345)
# TODO allow reading from disk
# TODO allow memory pinning
class ReviewDataset(Dataset):
    """A Dataset of Partially generated Reviews"""
    #An element of the dataset is a three-tuple consisting of:
    #* a representation of the references
    #* a representation of the review question + a partial summary/conclusion state
    #* the target next word to generate

    Instance = namedtuple('Instance', ['refs', 'preface', 'target'])

    def __init__(self, data: List[TargetSummary], format_function):
        super(ReviewDataset).__init__()
        self.data = data
        random.shuffle(self.data)
        self.instances = list(itertools.chain.from_iterable(map(format_function, self.data)))

    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        return self.instances[idx]

    @staticmethod
    def from_file(f: str, format_function) -> 'ReviewDataset':
        def tensorize_reference(reference: TargetReference) -> TargetReference:
            title_abstract = torch.LongTensor(reference.title_abstract)
            full_text = torch.LongTensor(reference.full_text) if reference.full_text is not None else None
            return replace(reference,
                title_abstract=title_abstract,
                full_text=full_text,
            )
        def tensorize(summary: TargetSummary) -> TargetSummary:
            # TODO what about summaries with no preface
            preface = torch.LongTensor(summary.preface) if summary.preface is not None and len(summary.preface) > 0 else torch.LongTensor([0])
            references = list(map(tensorize_reference, summary.references))
            target_texts = list(map(torch.LongTensor, summary.target_texts))
            return replace(
                summary,
                preface=preface,
                target_texts=target_texts,
                references=references,
            )

        summaries = TargetSummary.read_summaries(f)
        summaries = list(map(tensorize, summaries))
        return ReviewDataset(summaries, format_function)
 
    @staticmethod
    def to_flattened_model_inputs(instance: TargetSummary) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # TODO this is a dumb concatenation. be smarter. add separators or something.
        ref_texts = torch.cat([ref.title_abstract for ref in instance.references], dim=0)
        preface = instance.preface
        ret = []
        for txt in instance.target_texts:
            ret.append(ReviewDataset.Instance(ref_texts, preface, txt))
        return ret

class ToUnflattenedModelInputsFunction(object):
    def __init__(self, padding_value):
        self.padding_value = padding_value

    def __call__(self, instance: TargetSummary) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        ref_texts = [ref.title_abstract for ref in instance.references]
        ref_texts = pad_tensors(ref_texts, padding_value=self.padding_value)
        preface = instance.preface
        ret = []
        for txt in instance.target_texts:
            ret.append((ReviewDataset.Instance(ref_texts, preface, txt)))
        return ret