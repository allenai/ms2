import itertools
import logging

from collections import defaultdict
from dataclasses import dataclass, asdict, is_dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from dataclasses_json import dataclass_json
from transformers import AutoTokenizer

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
NUM_PROCS = 8

START_POPULATION='<pop>'
END_POPULATION='</pop>'
START_INTERVENTION='<int>'
END_INTERVENTION='</int>'
START_OUTCOME='<out>'
END_OUTCOME='</out>'
START_BACKGROUND = '<background>'
END_BACKGROUND = '</background>'
START_REFERENCE = '<ref>'
END_REFERENCE = '</ref>'
START_EVIDENCE = '<evidence>'
END_EVIDENCE = '</evidence>'
SEP_TOKEN = '<sep>'
EXTRA_TOKENS = [
    START_BACKGROUND,
    END_BACKGROUND,
    START_REFERENCE,
    END_REFERENCE,
    SEP_TOKEN,
    START_POPULATION,
    END_POPULATION,
    START_INTERVENTION,
    END_INTERVENTION,
    START_OUTCOME,
    END_OUTCOME,
    START_EVIDENCE,
    END_EVIDENCE,
]

def get_tokenizer(tokenizer_type: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_type, additional_special_tokens=EXTRA_TOKENS)
    return tokenizer

@dataclass_json
@dataclass
class Significance:
    intervention: str
    outcome: str
    classification: Dict[str, float]
    evidence_sentence: Optional[str]=None
    evidence_sentence_score: Optional[float]=None

@dataclass_json
@dataclass
class TargetReference:
    title_abstract: Union[torch.Tensor, str]
    full_text: Optional[Union[torch.Tensor, List[int], str]]
    s2id: Optional[str]
    s2hash: Optional[str]

@dataclass_json
@dataclass
class TargetSummary:
    """Target input/output for a summarization model

    Preface:
    """
    preface: Optional[Union[str, List[int], torch.Tensor]]
    target_texts: Union[List[str], List[int], List[torch.Tensor]]
    review_id: str
    references: List[TargetReference]
    s2id: Optional[str]
    s2hash: Optional[str]

    @staticmethod
    def read_summaries(f: str) -> List['TargetSummary']:
        with open(f, 'r') as inf:
            summaries = map(TargetSummary.from_json, inf)
            summaries = list(summaries)
        return summaries

@dataclass_json
@dataclass
class Reference:
    """Any kind of scientific paper
    
    Unfortunately, none of the fields are always present in the data, so we
    will be left guessing what the best way to find the actual text of any
    given reference.
    """
    identifiers: List[Dict[str, str]]
    metadata: Dict[str, str]
    title: Optional[str]=None
    doi: Optional[str]=None
    pmid: Optional[str]=None
    # these must be populated later
    s2id: Optional[str]=None
    s2hash: Optional[str]=None
    abstract: Optional[str]=None
    content: Optional[str]=None
    publication_types: Optional[List[str]]=None
    significances: Optional[List[Significance]]=None
    interventions: Optional[List[str]]=None
    outcomes: Optional[List[str]]=None
    populations: Optional[List[str]]=None
    in_doc_significances: Optional[List[Significance]]=None

@dataclass_json
@dataclass
class Study:
    """Any scientific study

    Typically there is a one-to-one mapping between references and studies,
    although not always. Some studies appear to be published multiple ways, at
    multiple times (e.g. a full paper and a later conference abstract).

    In this dataset, a study should always contain exactly one reference element.
    """
    references: List[Reference]
    identifiers: List[Dict[str, str]]
    metadata: Dict[str, str]
    pmid: Optional[str]=None
    doi: Optional[str] = None

@dataclass_json
@dataclass
class Review:
    """Systematic Review representation

    All reviews should have a structured abstract, a document name, and title.
    One would expect to be able to always have access to included studies, and
    one would be wrong. At least one review exists in the data where no studies
    could be found, and thus no review was performed
    """
    docid: str
    title: str
    authors: str
    abstract: str
    # the final field is an optional distribution over model labels
    structured_abstract: List[Tuple[str, str, Optional[Dict[str, float]]]]
    summary: Optional[str]
    structured_summary: List[Tuple[str, str, Optional[Dict[str, float]]]]
    included_studies: List[Study]
    ongoing_studies: List[Study]
    awaiting_studies: List[Study]
    excluded_studies: List[Study]
    general_references: List[Reference]
    unattributed_references: Optional[List[Reference]]=None
    content: Optional[str]=None
    doi: Optional[str]=None
    content: Optional[str]=None # separate from the abstract
    s2id: Optional[str]=None
    s2hash: Optional[str]=None
    pmid: Optional[str]=None
    interventions: Optional[List[str]]=None
    outcomes: Optional[List[str]]=None
    populations: Optional[List[str]]=None
    significances: Optional[List[Significance]]=None

    def __post_init__(self):
        assert self.docid is not None
        assert self.title is not None
        assert self.abstract is not None and len(self.abstract.strip()) > 0
        #assert len(self.structured_abstract) > 0

    def extract_references(self) -> List[Reference]:
        studies = itertools.chain.from_iterable([
            self.included_studies,
            self.ongoing_studies,
            self.excluded_studies,
            self.awaiting_studies,
        ])
        study_refs = itertools.chain.from_iterable(map(lambda x: x.references, studies))
        all_refs = list(study_refs) + self.general_references + self.unattributed_references
        return all_refs

    @staticmethod
    def read_reviews(f) -> List['Review']:
        with open(f, 'r') as inf:
            inf = filter(lambda line: len(line.strip()) > 0, inf)
            reviews = map(Review.from_json, inf)
            reviews = list(reviews)
        return reviews