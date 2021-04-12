from dataclasses import dataclass
from typing import List

import torch

from rouge_score import rouge_scorer
from rouge_score import scoring

"""Citation
@inproceedings{lin-2004-rouge,
    title = "{ROUGE}: A Package for Automatic Evaluation of Summaries",
    author = "Lin, Chin-Yew",
    booktitle = "Text Summarization Branches Out",
    month = jul,
    year = "2004",
    address = "Barcelona, Spain",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W04-1013",
    pages = "74--81",
}
"""

from torch import nn
from torch.nn.utils.rnn import pad_sequence, PackedSequence, pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence


def rouge_scores(preds: List[List[torch.Tensor]], targets: List[List[torch.Tensor]], tokenizer, use_stemmer=False, use_aggregator=False):
    # largely copied from https://github.com/huggingface/nlp/blob/master/metrics/rouge/rouge.py#L84
    rouge_types = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=use_stemmer)
    refs, hyps = [], []
    for p, t in zip(preds, targets):
        assert len(p) == len(t)
        refs.extend(p)
        hyps.extend(t)

    if use_aggregator:
        aggregator = scoring.BootstrapAggregator()
        scores = None
    else:
        aggregator = None
        scores = []

    for ref, pred in zip(refs, hyps):
        if isinstance(ref, torch.Tensor):
            ref = tokenizer.decode(ref).lower()
        if isinstance(pred, torch.Tensor):
            pred = tokenizer.decode(pred).lower()
        score = scorer.score(ref, pred)
        if use_aggregator:
            aggregator.add_scores(score)
        else:
            scores.append(score)

    if use_aggregator:
        result = aggregator.aggregate()
    else:
        result = {}
        for key in scores[0]:
            result[key] = list(score[key] for score in scores)

    return result

def pad_tensors(data: List[torch.Tensor], padding_value) -> torch.Tensor:
    data_ = []
    for d in data:
        if len(d.size()) == 0:
            d = d.unsqueeze(0)
        data_.append(d)
    padded = pad_sequence(data_, batch_first=True, padding_value=padding_value)
    return padded

# TODO(jayd) memory pinning?
# borrowed from the EraserBenchmark
@dataclass(eq=True, frozen=True)
class PaddedSequence:
    """A utility class for padding variable length sequences mean for RNN input
    This class is in the style of PackedSequence from the PyTorch RNN Utils,
    but is somewhat more manual in approach. It provides the ability to generate masks
    for outputs of the same input dimensions.
    The constructor should never be called directly and should only be called via
    the autopad classmethod.
    We'd love to delete this, but we pad_sequence, pack_padded_sequence, and
    pad_packed_sequence all require shuffling around tuples of information, and some
    convenience methods using these are nice to have.
    """

    data: torch.Tensor
    batch_sizes: torch.Tensor
    batch_first: bool = False

    @classmethod
    def autopad(cls, data, batch_first: bool = False, padding_value=0, device=None) -> 'PaddedSequence':
        # handle tensors of size 0 (single item)
        data_ = []
        for d in data:
            if len(d.size()) == 0:
                d = d.unsqueeze(0)
            data_.append(d)
        padded = pad_sequence(data_, batch_first=batch_first, padding_value=padding_value)
        if batch_first:
            batch_lengths = torch.LongTensor([x.size()[0] for x in data_])
            if any([x == 0 for x in batch_lengths]):
                raise ValueError(
                    "Found a 0 length batch element, this can't possibly be right: {}".format(batch_lengths))
        else:
            # TODO actually test this codepath
            batch_lengths = torch.LongTensor([len(x) for x in data])
        return PaddedSequence(padded, batch_lengths, batch_first).to(device=device)

    def pack_other(self, data: torch.Tensor):
        return pack_padded_sequence(data, self.batch_sizes, batch_first=self.batch_first, enforce_sorted=False)

    @classmethod
    def from_packed_sequence(cls, ps: PackedSequence, batch_first: bool, padding_value=0) -> 'PaddedSequence':
        padded, batch_sizes = pad_packed_sequence(ps, batch_first, padding_value)
        return PaddedSequence(padded, batch_sizes, batch_first)

    def cuda(self) -> 'PaddedSequence':
        return PaddedSequence(self.data.cuda(), self.batch_sizes.cuda(), batch_first=self.batch_first)

    def to(self, device=None, dtype=None, copy=False, non_blocking=False) -> 'PaddedSequence':
        # TODO make to() support all of the torch.Tensor to() variants
        return PaddedSequence(
            self.data.to(device=device, dtype=dtype, copy=copy, non_blocking=non_blocking),
            self.batch_sizes.to(device=device, copy=copy, non_blocking=non_blocking),
            batch_first=self.batch_first)

    def mask(self, on=int(0), off=int(0), device='cpu', size=None, dtype=None) -> torch.Tensor:
        if size is None:
            size = self.data.size()
        out_tensor = torch.zeros(*size, dtype=dtype)
        # TODO this can be done more efficiently
        out_tensor.fill_(off)
        # note to self: these are probably less efficient than explicilty populating the off values instead of the on values.
        if self.batch_first:
            for i, bl in enumerate(self.batch_sizes):
                out_tensor[i, :bl] = on
        else:
            for i, bl in enumerate(self.batch_sizes):
                out_tensor[:bl, i] = on
        return out_tensor.to(device)

    def unpad(self, other: torch.Tensor=None) -> List[torch.Tensor]:
        if other is None:
            other = self
        if isinstance(other, PaddedSequence):
            other = other.data
        out = []
        for o, bl in zip(other, self.batch_sizes):
            out.append(o[:bl])
        return out

    def flip(self) -> 'PaddedSequence':
        return PaddedSequence(self.data.transpose(0, 1), not self.batch_first, self.padding_value)

def mangle_bart_with_longformer(bart_model, extend_encoder=True, extend_decoder=True):
    # TODO fix with https://github.com/allenai/longformer/blob/encoderdecoder/scripts/convert_bart_to_longformerencoderdecoder.py
    def replace_layers(model, config):
        for i, layer in enumerate(model.layers):
            self_attn = LongformerSelfAttention(config, layer_id=i)
            self_attn.query = layer.self_attn.q_proj
            self_attn.key = layer.self_attn.k_proj
            self_attn.value = layer.self_attn.v_proj
            # TODO should these parameters be tied? they aren't in the longformer source
            self_attn.query_global = layer.self_attn.q_proj
            self_attn.key_global = layer.self_attn.k_proj
            self_attn.value_global = layer.self_attn.v_proj
            # TODO longformer has no out_proj which seems odd
            layer.self_attn.self = self_attn
    bart_model.config.max_position_embeddings = 16 * bart_model.config.max_position_embeddings
    # this is a hack.
    # it might even get fixed eventually.
    if extend_decoder:
        new_decoder_embeds = torch.cat(
           [bart_model.model.decoder.embed_positions.weight] +
           15 * [bart_model.model.decoder.embed_positions.weight[2:]],
        dim=0).clone()
        # TODO experiment with adding additional tokens, e.g. separating documents (or maybe an embedding per document?) + separating the prompt + ??
        bart_model.model.decoder.embed_positions = LearnedPositionalEmbedding(
           num_embeddings=new_decoder_embeds.size()[0] - 2,
           embedding_dim=new_decoder_embeds.size()[1],
           padding_idx=bart_model.model.decoder.embed_positions.padding_idx,
           offset=0,
        )
        bart_model.model.decoder.embed_positions.weight = nn.Parameter(new_decoder_embeds, requires_grad=True)

    if extend_encoder:
        new_encoder_embeds = torch.cat(
           [bart_model.model.encoder.embed_positions.weight] +
           15 * [bart_model.model.encoder.embed_positions.weight[2:]],
           dim=0).clone()
        bart_model.model.encoder.embed_positions = LearnedPositionalEmbedding(
           num_embeddings=new_encoder_embeds.size()[0] - 2,
           embedding_dim=new_encoder_embeds.size()[1],
           padding_idx=bart_model.model.encoder.embed_positions.padding_idx,
           offset=0,
        )
        bart_model.model.encoder.embed_positions.weight = nn.Parameter(new_encoder_embeds, requires_grad=True)
    # TODO(jayd,iz) are any of these parameters even in the right ballpark?
    bart_model.config.attention_probs_dropout_prob = bart_model.config.dropout
    bart_model.config.attention_window = [64] * len(bart_model.model.encoder.layers)
    bart_model.config.attention_dilation = [1] * len(bart_model.model.encoder.layers)
    bart_model.config.attention_mode = 'tvm'
    # bart_model.config.attention_mode = 'sliding_chunks' # TODO is this the right choice?
    bart_model.config.autoregressive = False
    if extend_encoder:
        replace_layers(bart_model.model.encoder, bart_model.config)
    if extend_decoder:
        replace_layers(bart_model.model.decoder, bart_model.config)
