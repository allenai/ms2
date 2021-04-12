import argparse
import glob
import itertools
import logging
import os
import re
import json

from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import classification_report
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler

from transformers import BartForConditionalGeneration, AutoConfig
from transformers.file_utils import ModelOutput
from transformers.modeling_bart import _prepare_bart_decoder_inputs
from transformers.modeling_outputs import BaseModelOutput
from transformers.optimization import get_linear_schedule_with_warmup

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateLogger
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel
from pytorch_lightning.core.lightning import LightningModule

from longformer import LongformerEncoderDecoderForConditionalGeneration, LongformerEncoderDecoderConfig
from longformer.sliding_chunks import pad_to_window_size

from ms2.data.review_datasets import ReviewDataset, ToUnflattenedModelInputsFunction
from ms2.models.utils import rouge_scores
from ms2.utils import get_tokenizer

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


class ReferenceInteractingBartSummarizer(nn.Module):
    # tokenizer passed separately to allow adding for special characters
    def __init__(self, model_type, tokenizer, args):
        super().__init__()
        config = AutoConfig.from_pretrained(model_type)
        config.attention_dropout = args.attention_dropout
        config.gradient_checkpointing = args.grad_ckpt
        if model_type == 'facebook/bart-base':  # bug in HF configuration
            config.encoder_attention_heads = 12
            config.decoder_attention_heads = 12
        self.model = BartForConditionalGeneration.from_pretrained(model_type, config=config)
        self.model.resize_token_embeddings(len(tokenizer.get_vocab()))
        self.tokenizer = tokenizer
        self.config = self.model.config
        self.args = args

    def _encode_multiple(
            self,
            inputs: torch.LongTensor,
            preamble: torch.LongTensor,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False
        ):
        """
        inputs: Padded reference texts
        preamble: single beginning/prompt
        """
        inputs = inputs[:self.args.max_num_refs]
        preamble = preamble.repeat(inputs.size()[0], 1)
        encoder_input = torch.cat([preamble, inputs], dim=1)[:, :self.config.max_position_embeddings]
        encoder_outputs = self.model.model.encoder(
            encoder_input,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            attention_mask=None,
            return_dict=False
        )
        selection_mask = encoder_input != self.config.pad_token_id
        input_ids = torch.masked_select(encoder_input, selection_mask).unsqueeze(0)
        if len(encoder_outputs) == 1:
            encoded = encoder_outputs[0]
            encoder_states, all_attentions = None, None
        else:
            encoded, encoder_states, all_attentions = encoder_outputs
            encoder_states = tuple(torch.masked_select(hs, selection_mask) for hs in encoder_states)
            all_attentions = tuple(torch.masked_select(attn, selection_mask) for attn in all_attentions)
        encoded_sequences = torch.masked_select(encoded, selection_mask.unsqueeze(-1)).reshape(1, -1, encoded.size()[-1])
        if torch.any(torch.isnan(encoded)):
            raise RuntimeError('Found nans while encoding inputs!')
        if return_dict:
            return input_ids, BaseModelOutput(
                last_hidden_state=encoded_sequences,
                hidden_states=encoder_states,
                attentions=all_attentions,
            )
        else:
            return input_ids, (encoded_sequences, encoder_states, all_attentions)

    def forward(self, inputs: torch.Tensor, preambles: torch.Tensor, targets: torch.Tensor):
        # prep the decoder inputs and the loss labels and masks
        # Note that `lm_labels` is similar to `decoder_input_ids` but shifted one step to the left.
        # There's also a small difference in the use of <s> and </s> as shown in the following example
        # For example,
        #   decoder_input_ids = '<s> some text .'
        #   lm_labels         = ' some text .</s>'
        targets = targets[:, :self.args.max_length]  # limit target length for memory
        decoder_input_ids = targets[:, :-1].contiguous()
        lm_labels = targets[:, 1:].clone()

        decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
            self.model.config,
            None,  # this would be the input ids but we very much do not want them here
            decoder_input_ids=decoder_input_ids,
            decoder_padding_mask=None,
            causal_mask_dtype=self.model.model.shared.weight.dtype,
        )
        _, (encoded_sequences, _, _) = self._encode_multiple(inputs, preambles, return_dict=False)
        if torch.any(torch.isnan(encoded_sequences.data)):
            raise RuntimeError('Found nans while encoding inputs!')

        # decoder output!
        decoder_outputs = self.model.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoded_sequences,
            encoder_padding_mask=None,
            decoder_padding_mask=decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            decoder_cached_states=None,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=False,
        )
        if torch.any(torch.isnan(decoder_outputs[0])):
            raise RuntimeError('Found nans while decoding!')

        lm_logits = F.linear(decoder_outputs[0], self.model.model.shared.weight, bias=self.model.final_logits_bias)
        if torch.any(torch.isnan(lm_logits)):
            raise RuntimeError('Found nans while predicting lm weights!')
        outputs = (lm_logits,) + decoder_outputs[1:]  # Add cache, hidden states and attention if they are here
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        # Note: masking will need to be re-added if bs > 1 (currently not possible!)
        masked_lm_loss = loss_fct(lm_logits.view(-1, self.model.config.vocab_size), lm_labels.view(-1))
        if torch.any(torch.isnan(masked_lm_loss)):
            raise RuntimeError('Invalid loss!')
        masked_lm_loss = masked_lm_loss.mean()
        outputs = (masked_lm_loss,) + outputs

        return outputs

    @torch.no_grad()
    def generate_summary(
        self,
        inputs: torch.Tensor,
        preambles: torch.Tensor,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        **model_kwargs
    ) -> torch.LongTensor:

        # We cannot generate if the model does not have a LM head
        if self.model.get_output_embeddings() is None:
            raise AttributeError(
                "You tried to generate sequences with a model that does not have a LM Head."
                "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`, `XLMWithLMHeadModel`, `BartForConditionalGeneration` )"
            )

        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
        )
        bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        )

        # just decode one at a time for sanity's sake!
        batch_size = 1
        #if input_ids is not None:
        #    batch_size = input_ids.shape[0]  # overriden by the input batch_size
        #else:
        #    batch_size = 1

        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
        assert isinstance(min_length, int) and min_length >= 0, "`min_length` should be a positive integer."
        assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
        assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
        assert isinstance(use_cache, bool), "`use_cache` should be a boolean."
        assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
        assert temperature > 0, "`temperature` should be strictly positive."
        assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
        assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
        assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert inputs is not None or preambles is not None
        assert inputs is not None or (
            isinstance(bos_token_id, int) and bos_token_id >= 0
        ), "If inputs is not defined, `bos_token_id` should be a positive integer."
        assert pad_token_id is None or (
            isinstance(pad_token_id, int) and (pad_token_id >= 0)
        ), "`pad_token_id` should be a positive integer."
        assert (eos_token_id is None) or (
            isinstance(eos_token_id, int) and (eos_token_id >= 0)
        ), "`eos_token_id` should be a positive integer."
        assert length_penalty > 0, "`length_penalty` should be strictly positive."
        assert (
            isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0
        ), "`no_repeat_ngram_size` should be a positive integer."
        assert (
            isinstance(num_return_sequences, int) and num_return_sequences > 0
        ), "`num_return_sequences` should be a strictly positive integer."
        assert (
            bad_words_ids is None or isinstance(bad_words_ids, list) and isinstance(bad_words_ids[0], list)
        ), "`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated"

        # not allow to duplicate outputs when greedy decoding
        if do_sample is False:
            if num_beams == 1:
                # no_beam_search greedy generation conditions
                assert (
                    num_return_sequences == 1
                ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

            else:
                # beam_search greedy generation conditions
                assert (
                    num_beams >= num_return_sequences
                ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

        # create attention mask if necessary
        # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
        # if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
        #    attention_mask = input_ids.ne(pad_token_id).long()
        # elif attention_mask is None:
        #    attention_mask = input_ids.new_ones(input_ids.shape)

        # set pad_token_id to eos_token_id if not set. Important that this is done after
        # attention_mask is created
        if pad_token_id is None and eos_token_id is not None:
            logging.warning(
                "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_id)
            )
            pad_token_id = eos_token_id

        # current position and vocab size
        if hasattr(self.config, "vocab_size"):
            vocab_size = self.config.vocab_size
        elif (
            self.config.is_encoder_decoder
            and hasattr(self.config, "decoder")
            and hasattr(self.config.decoder, "vocab_size")
        ):
            vocab_size = self.config.decoder.vocab_size

        # set effective batch size and effective batch multiplier according to do_sample
        if do_sample:
            effective_batch_size = batch_size * num_return_sequences
            effective_batch_mult = num_return_sequences
        else:
            effective_batch_size = batch_size
            effective_batch_mult = 1

        assert self.config.is_encoder_decoder
        if decoder_start_token_id is None:
            # see if BOS token can be used for decoder_start_token_id
            if bos_token_id is not None:
                decoder_start_token_id = bos_token_id
            elif hasattr(self.config, "decoder") and hasattr(self.config.decoder, "bos_token_id"):
                decoder_start_token_id = self.config.decoder.bos_token_id
            else:
                raise ValueError(
                    "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
                )

        assert hasattr(self.model, "get_encoder"), "{} should have a 'get_encoder' function defined".format(self.model)
        assert callable(self.model.get_encoder), "{} should be a method".format(self.model.get_encoder)

        # get encoder and store encoder outputs
        encoder_outputs: ModelOutput
        input_ids, encoder_outputs = self._encode_multiple(inputs, preambles, return_dict=True)

        # Expand input ids if num_beams > 1 or num_return_sequences > 1
        if num_return_sequences > 1 or num_beams > 1:
            input_ids_len = input_ids.shape[-1]
            input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)
            #attention_mask = attention_mask.unsqueeze(1).expand(
            #    batch_size, effective_batch_mult * num_beams, input_ids_len
            #)

            input_ids = input_ids.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
            #attention_mask = attention_mask.contiguous().view(
            #    effective_batch_size * num_beams, input_ids_len
            #)  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

        attention_mask = None
        if self.config.is_encoder_decoder:
            # create empty decoder_input_ids
            input_ids = torch.full(
                (effective_batch_size * num_beams, 1),
                decoder_start_token_id,
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
            cur_len = 1

            assert (
                batch_size == encoder_outputs.last_hidden_state.shape[0]
            ), f"expected encoder_outputs.last_hidden_state to have 1st dimension bs={batch_size}, got {encoder_outputs.last_hidden_state.shape[0]} "

            # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
            expanded_batch_idxs = (
                torch.arange(batch_size)
                .view(-1, 1)
                .repeat(1, num_beams * effective_batch_mult)
                .view(-1)
                .to(input_ids.device)
            )

            # expand encoder_outputs
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_batch_idxs
            )

            # save encoder_outputs in `model_kwargs`
            model_kwargs["encoder_outputs"] = encoder_outputs

        else:
            cur_len = input_ids.shape[-1]

        assert (
            cur_len < max_length
        ), f"The context has {cur_len} number of tokens, but `max_length` is only {max_length}. Please make sure that `max_length` is bigger than the number of tokens, by setting either `generate(max_length=...,...)` or `config.max_length = ...`"

        if num_beams > 1:
            output = self.model._generate_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                early_stopping=early_stopping,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                num_return_sequences=num_return_sequences,
                length_penalty=length_penalty,
                num_beams=num_beams,
                vocab_size=vocab_size,
                attention_mask=attention_mask,
                use_cache=use_cache,
                model_kwargs=model_kwargs,
            )
        else:
            output = self.model._generate_no_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                attention_mask=attention_mask,
                use_cache=use_cache,
                model_kwargs=model_kwargs,
            )

        return output

    def collate_fn(self, batch: List[ReviewDataset.Instance]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (inst,) = batch
        return inst.refs, inst.preface.unsqueeze(0), inst.target.unsqueeze(0)

# TODO add a simple transformer
# TODO add a SciBERT or BiomedRoBERTa transformer
# TODO uh oh, size problems!
class LightningBartSummarizer(LightningModule):

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.tokenizer = get_tokenizer('facebook/bart-base')
        if 'long' in args.model_name:
            self.summarizer = SingleStreamBartSummarizer(args.model_name, self.tokenizer, args)
        else:
            self.summarizer = ReferenceInteractingBartSummarizer(args.model_name, self.tokenizer, args)

        self.config = self.summarizer.config
        self.generation_params = {
            'num_beams': args.num_beams,
            'length_penalty': args.length_penalty,
            'no_repeat_ngram_size': args.no_repeat_ngram_size,
            'early_stopping': True,
            'decoder_start_token_id': self.config.bos_token_id,
            'min_length': args.min_length,
            'max_length': args.max_length,
            'temperature': args.temperature,
            'repetition_penalty': args.repetition_penalty,
        }
        self.predictions_file = None

    def forward(self, inputs, preambles, targets):
        return self.summarizer.forward(inputs=inputs, preambles=preambles, targets=targets)

    @torch.no_grad()
    def generate_summary(self, *args, **kwargs) -> torch.LongTensor:
        for p in self.summarizer.parameters():
            p.requires_grad = False
        ret = self.summarizer.generate_summary(*args, **kwargs)
        for p in self.summarizer.parameters():
            p.requires_grad = True
        return ret

    def training_step(self, batch, batch_idx):
        outputs = self.forward(*batch)
        loss = outputs[0]
        output = {
            'loss': loss,
            'train_loss': loss,
            'log': {
                'train_loss': loss,
                'lr': loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]['lr'],
            },
        }
        return output

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        for p in self.summarizer.parameters():
            p.requires_grad = False
        outputs = self.forward(*batch)
        for p in self.summarizer.parameters():
            p.requires_grad = True
        loss = outputs[0]
        lm_logits = outputs[1]
        generations = self.generate_summary(
            batch[0],
            batch[1],
            **self.generation_params,
        )
        targets = batch[2]
        output = {
            'val_loss': loss.cpu(),
            'progress_bar': {
                'val_loss': loss.cpu(),
            },
            'preambles': [x.cpu() for x in batch[1]],
            'generations': [[x.cpu()] for x in generations],
            'teacher_forced_generations': [torch.argmax(lm_logits, dim=1).detach().cpu()],
            'targets': [[x.cpu()] for x in targets],
        }
        assert len(output['generations']) == len(output['targets'])
        assert list(map(len, output['generations'])) == list(map(len, output['targets']))
        return output

    def validation_epoch_end(self, outputs):
        losses = np.mean([output['val_loss'] for output in outputs])
        generated, teacher_forced_generations, targets = self._accumulate_generations(outputs)
        assert len(generated) > 0
        scores = rouge_scores(generated, targets, self.summarizer.tokenizer, use_aggregator=True)
        tf_scores = rouge_scores(teacher_forced_generations, targets, self.summarizer.tokenizer, use_aggregator=True)
        # TODO: if self.use_ddp: sync val_loss and rouge scores across GPUs
        output = {
            'val_loss': losses,
            'log': {
                'val_loss': losses,
            },
        }
        for rouge_type, prf_scores in scores.items():
            output['val_' + rouge_type + '_p'] = prf_scores.mid.precision
            output['val_' + rouge_type + '_r'] = prf_scores.mid.recall
            output['val_' + rouge_type + '_f'] = prf_scores.mid.fmeasure
            output['log']['val_' + rouge_type + '_p'] = prf_scores.mid.precision
            output['log']['val_' + rouge_type + '_r'] = prf_scores.mid.recall
            output['log']['val_' + rouge_type + '_f'] = prf_scores.mid.fmeasure
        for rouge_type, prf_scores in tf_scores.items():
            output['val_tf' + rouge_type + '_p'] = prf_scores.mid.precision
            output['val_tf' + rouge_type + '_r'] = prf_scores.mid.recall
            output['val_tf' + rouge_type + '_f'] = prf_scores.mid.fmeasure
        output['progress_bar'] = {
            'val_loss': losses,
            'val_rougeL_f': output['val_rougeL_f'],
            'val_rougeLsum_f': output['val_rougeLsum_f'],
            'val_rouge1_f': output['val_rouge1_f'],
            'val_rouge2_f': output['val_rouge2_f'],
            #'val_rougeL_r': output['val_rougeL_r'],
            #'val_rouge1_r': output['val_rouge1_r'],
            #'val_rouge2_r': output['val_rouge2_r'],
            #'val_rougeL_p': output['val_rougeL_p'],
            #'val_rouge1_p': output['val_rouge1_p'],
            #'val_rouge2_p': output['val_rouge2_p'],
        }
        for k, v in output.items():
            if 'rouge' in k:
                output['log'][k] = v
        if self.args.evidence_inference_eval:
            scores = self._evidence_inference_score(generated, targets)
            output['progress_bar']['macro_f1'] = scores['macro avg']['f1-score']
            output['log']['macro_f1'] = scores['macro avg']['f1-score']
            output['log']['macro_r'] = scores['macro avg']['recall']
            output['log']['macro_p'] = scores['macro avg']['precision']
            for i, j in scores.items():
                if not isinstance(j, dict):
                    output['log'][i] = j
                else:
                    for k, l in j.items():
                        ik = i + '_' + k
                        output['log'][ik] = l

        return output

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        if self.predictions_file is None:
            self.predictions_file = open(os.path.join(self.args.training_root, 'predictions.json'), 'w')
        output = self.validation_step(batch, batch_idx)
        data = {
            'batch_idx': batch_idx,
            'preamble': self.tokenizer.decode(batch[1].squeeze(), skip_special_tokens=False),
            'generated': self.tokenizer.decode(output['generations'][0][0], skip_special_tokens=False),
            'target': self.tokenizer.decode(output['targets'][0][0], skip_special_tokens=False)
        }
        json_record = json.dumps(data)
        self.predictions_file.write(json_record + '\n')
        self.predictions_file.flush()
        return {'test_loss': output['val_loss'], 'progress_bar': {'val_loss': output['val_loss']}, }

    def test_epoch_end(self, outputs):
        self.predictions_file.close()

    def _accumulate_generations(self, outputs) -> Tuple[List[List[torch.IntTensor]], List[List[torch.IntTensor]], List[List[torch.IntTensor]]]:
        generated = []
        teacher_forced_generations = []
        targets = []
        for output in outputs:
            # both the generated and targets should be lists of lists of inttensors
            gen = output.get('generations', [])
            tf = output.get('teacher_forced_generations', [])
            if len(gen) > 0:
                generated.extend(gen)
                teacher_forced_generations.extend(tf)
                tgt = output.get('targets', [])
                targets.extend(tgt)
                assert len(tgt) == len(gen)
        return generated, teacher_forced_generations, targets

    def _evidence_inference_score(self, generations, truths):
        generations_labels = ['significantly decreased', 'no significant difference', 'significantly increased', 'broken generation']
        generations_mapping = {
            'significantly decreased': 0,
            'no significant difference': 1,
            'significantly increased': 2,
            'broken generation': 3
        }
        generations = list(map(lambda s: s.replace('<s>', '').replace('</s>', ''), map(str.lower, map(self.tokenizer.decode, itertools.chain.from_iterable(generations)))))
        truths = list(map(lambda s: s.replace('<s>', '').replace('</s>', ''), map(str.lower, map(self.tokenizer.decode, itertools.chain.from_iterable(truths)))))
        pretty_generations = []
        pretty_truths = []
        for gen in generations:
            pretty_generations.append(generations_mapping.get(gen, 3))
        for t in truths:
            pretty_truths.append(generations_mapping.get(t, 3))
        all_labels = set(generations_labels[x] for x in (set(pretty_generations) | set(pretty_truths)))
        assert len(generations) == len(truths)
        return classification_report(
            pretty_truths,
            pretty_generations,
            target_names=all_labels,
            output_dict=True,
            digits=3,
            zero_division=0,
        )

    def configure_optimizers(self):
        if self.args.debug:
            return torch.optim.Adam(self.summarizer.parameters(), lr=self.args.lr)  # const LR
        optimizer = torch.optim.Adam(self.summarizer.parameters(), lr=self.args.lr, eps=self.args.adam_epsilon)
        num_gpus = torch.cuda.device_count()
        num_steps = self.args.dataset_size * self.args.epochs / num_gpus / self.args.grad_accum
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def configure_ddp(self, model, device_ids):
        # Needs to override the default ddp to set `find_unused_parameters=False` for gradient checkpointing to work
        model = LightningDistributedDataParallel(
            model,
            device_ids=device_ids,
            find_unused_parameters=False
        )
        return model

    def _get_loader(self, dataset, is_train):
        if self.trainer.use_ddp:
            sampler = DistributedSampler(dataset, shuffle=is_train)
            shuffle = False
        else:
            sampler = None
            shuffle = is_train
        loader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=0,
                collate_fn=self.summarizer.collate_fn,
                drop_last=False,
        )
        return loader

    def train_dataloader(self):
        return self._get_loader(self.train_dataset, True)

    def val_dataloader(self):
        return self._get_loader(self.val_dataset, False)

    def test_dataloader(self):
        return self._get_loader(self.val_dataset, False)

    def grad_norm(self, norm_type):
        # Override PTL `grad_norm` function to only return `total_grad_norm` instead norms of individual params
        # TODO: grad_norm reporting needs to take fp16 loss scale into account
        parameters = [p for p in self.parameters() if p.grad is not None]
        device = parameters[0].device
        total_norm = torch.zeros([], device=device if parameters else None)
        norm_type = float(norm_type)
        for p in parameters:
            param_norm = p.grad.data.pow(norm_type).sum()
            total_norm.add_(param_norm)
        total_norm = (total_norm ** (1.0 / norm_type))
        return {'total_grad_norm': total_norm}

    @classmethod
    def add_args(cls, parser):
        # generation args
        parser.add_argument('--num_beams', default=4, type=int, help='How many beams to use when decoding during validation')
        parser.add_argument('--min_length', type=int, default=20, help='Minimum summary lengths')
        parser.add_argument('--max_length', type=int, default=512, help='Maximum target lengths')
        parser.add_argument('--max_num_refs', type=int, default=25, help='Maximum number of reference text')
        parser.add_argument('--temperature', type=float, help="Sampling temperature")
        parser.add_argument('--repetition_penalty', type=float, help="")
        parser.add_argument('--length_penalty', default=2.0, type=float, help='Length penalty when decoding during validation')
        parser.add_argument('--no_repeat_ngram_size', default=3, type=int, help='Size of ngram not to repeat when decoding during validation')
        # training args
        parser.add_argument('--train_rouge_eval_batches', default=100, type=int, help='How often (in batches) to generate in the training data for rouge scoring?')
        parser.add_argument('--grad_ckpt', action='store_true', help='Enable gradient checkpointing to save memory')
        # model args
        parser.add_argument('--attention_dropout', default=0.1, type=float, help='Length penalty when decoding during validation')
        parser.add_argument('--lr', default=1e-5, type=float, help='Learning rate')
        parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Adam epsilon')
        parser.add_argument('--warmup_steps', default=1000, type=int, help='Batches for warmup')
        parser.add_argument('--fp16', action='store_true', help='Use fp16')
        parser.add_argument('--model_name', default='facebook/bart-base', help='name of path of a model')
        parser.add_argument('--evidence_inference_eval', default=False, action='store_true', help='When producing a significance classification, ')

        parser.add_argument('--debug', action='store_true', help='Debugging')


class SingleStreamBartSummarizer(nn.Module):

    def __init__(self, model_path, tokenizer, args):
        super().__init__()
        # TODO(jayd) look into DistilBART https://github.com/huggingface/transformers/blob/5543b30aa6b52da3c8f7d9e525b0edc26226d717/examples/seq2seq/
        config = LongformerEncoderDecoderConfig.from_pretrained(
            model_path,
            attention_mode='sliding_chunks_no_overlap',
            attention_dropout=args.attention_dropout,
            gradient_checkpointing=args.grad_ckpt,
        )
        # with `sliding_chunks_no_overlap`, attention size is 3 * attention_window. Use 340 if total amount of attention is 1024 (as in BART) or use 170 if you feel 170*3=510 is the average length of ref. I used 340 in other experiments and it works well and haven't tried 170
        attention_size = 340
        config.attention_window = [attention_size] * config.encoder_layers
        logging.info('config:' + str(config))
        self.model = LongformerEncoderDecoderForConditionalGeneration.from_pretrained(
            model_path,
            config=config,
        )
        self.max_input_length = (config.max_encoder_position_embeddings // (2 * attention_size)) * 2 * attention_size
        self.model.resize_token_embeddings(len(tokenizer.get_vocab()))
        self.tokenizer = tokenizer
        self.config = self.model.config
        self.args = args

    def _prepare_input_ids(self, inputs, preambles):
        # TODO fix the global attention mask
        assert inputs.size(0) == preambles.size(0) == 1
        input_ids = torch.cat([preambles, inputs], dim=1)  # combine preamble and refs in one long sequence
        input_ids = input_ids[:, :self.max_input_length]  # limit to max input size
        attention_mask = input_ids.new_ones(input_ids.shape, dtype=torch.long)
        attention_mask[input_ids == self.tokenizer.pad_token_id] = 0
        attention_mask[0, :preambles.size()[1]] = 2  # global attention on preamble
        input_ids, attention_mask = pad_to_window_size(  # ideally, should be moved inside the LongformerModel
            input_ids, attention_mask, self.config.attention_window[0], self.tokenizer.pad_token_id)
        assert all(list(map(lambda x: x <= self.max_input_length, input_ids.size())))
        return input_ids, attention_mask

    def forward(self, inputs, preambles, targets):
        input_ids, attention_mask = self._prepare_input_ids(inputs, preambles)
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            decoder_input_ids=targets[:, :-1], labels=targets[:, 1:])

    def collate_fn(self, batch: List[ReviewDataset.Instance]) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(batch) == 1
        instance = batch[0]
        refs = instance.refs.data
        refs = refs.masked_select(refs != 0)  # remove padding and combine in one long sequence
        preface = instance.preface
        target = instance.target
        return refs.unsqueeze(dim=0), preface.unsqueeze(dim=0), target.unsqueeze(dim=0)  # batch of size 1

    @torch.no_grad()
    def generate_summary(self, inputs, preambles, **kwargs):
        input_ids, attention_mask = self._prepare_input_ids(inputs, preambles)
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)


def get_args():
    parser = argparse.ArgumentParser(description='Train a BART based summarization model!')
    parser.add_argument('--epochs', default=10, type=int, help='Train for how many epochs?')
    parser.add_argument('--train', required=True, help='jsonl serialized training files')
    parser.add_argument('--val', required=True)
    parser.add_argument('--training_root', required=True, help='Where to save checkpoints, etc.')
    parser.add_argument('--grad_accum', default=4, type=int, help='Number of gradient accumulation steps')
    parser.add_argument('--test', action='store_true', help='Skip training. Run prediction on the validation set')
    parser.add_argument('--test_all_ckpts', action='store_true', help='Skip training. Run prediction on the validation set over all ckpts')
    parser.add_argument('--dataset_size', default=14607, type=int, help='Number of instances in the training set')  # TODO: read from the data

    LightningBartSummarizer.add_args(parser)
    return parser


def main():
    parser = get_args()
    args = parser.parse_args()
    model = LightningBartSummarizer(args)

    if args.test or args.test_all_ckpts:
        # loading the training dataset is expensive and unnecessary if we're only evaluating
        model.train_dataset = []
    else:
        model.train_dataset = ReviewDataset.from_file(args.train, format_function=ToUnflattenedModelInputsFunction(model.config.pad_token_id))
    model.val_dataset = ReviewDataset.from_file(args.val, format_function=ToUnflattenedModelInputsFunction(model.config.pad_token_id))
    logging.info(f'Loaded training dataset of length {len(model.train_dataset)}, val: {len(model.val_dataset)}')

    resume_from_checkpoint = None
    ckpts = glob.glob(os.path.join(args.training_root, '*.ckpt'))
    logging.info('Found {} pre-existing checkpoints: {}'.format(len(ckpts), ckpts))
    if len(ckpts) > 0:
        epochs = map(lambda ckpt: re.match('.*_([0-9]+)\.ckpt', ckpt).group(1), ckpts)
        ckpts = {int(e): c for (e, c) in zip(epochs, ckpts)}
        best = max(ckpts.keys())
        resume_from_checkpoint = ckpts[best]
        logging.info('Resuming from existing checkpoint {}'.format(resume_from_checkpoint))

    # single machine for the moment
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.training_root, 'model.ckpt'),
        verbose=True,
        # save_best_only=False,
        save_top_k=-1,
        monitor='val_loss',
        mode='min',
    )
    trainer = Trainer(
        gpus=-1,
        num_sanity_val_steps=2,
        val_check_interval=0.5 if not args.debug else 1.0,
        check_val_every_n_epoch=1 if not args.debug else 10,
        distributed_backend='ddp',
        replace_sampler_ddp=False,
        num_nodes=1,
        default_root_dir=args.training_root,
        max_epochs=args.epochs,
        log_gpu_memory=True,
        show_progress_bar=True,
        log_save_interval=10,
        accumulate_grad_batches=args.grad_accum,
        precision=16 if args.fp16 else 32, amp_level='O2',
        checkpoint_callback=checkpoint_callback,
        callbacks=[LearningRateLogger()],
        resume_from_checkpoint=resume_from_checkpoint,
        track_grad_norm=2,
    )

    if not (args.test or args.test_all_ckpts):
        trainer.fit(model)
    # Possibly a CUDA/pytorch bug: it seems after a recent update of the S2 servers
    # this code block would reliably trigger a crash of the nvidia drivers and require
    # a reboot to restore the server. scripts/modeling/decode.py still works.

    #trainer.test(model)
    #if args.test_all_ckpts:
    #    ckpts = glob.glob(os.path.join(args.training_root, '*.ckpt'))
    #    epochs = map(lambda ckpt: re.match('.*_([0-9]+)\.ckpt', ckpt).group(1), ckpts)
    #    ckpts = {int(e): c for (e, c) in zip(epochs, ckpts)}
    #    if len(ckpts) == 0:
    #        raise ValueError('Cannot restore from 0 checkpoints!')
    #    logging.info('Testing over {} pre-existing checkpoints: {}'.format(len(ckpts), ckpts))
    #    for epoch, ckpt in ckpts.items():
    #        resume_from_checkpoint = ckpt
    #        trainer = Trainer(
    #            gpus=-1,
    #            distributed_backend=None,
    #            replace_sampler_ddp=False,
    #            num_nodes=1,
    #            default_root_dir=args.training_root,
    #            log_gpu_memory=True,
    #            show_progress_bar=True,
    #            log_save_interval=10,
    #            precision=16 if args.fp16 else 32, amp_level='O2',
    #            resume_from_checkpoint=resume_from_checkpoint,
    #            track_grad_norm=2,
    #        )
    #        to_evaluate = LightningBartSummarizer.load_from_checkpoint(checkpoint_path=ckpt)
    #        to_evaluate.eval()
    #        to_evaluate.predictions_file = open(os.path.join(args.training_root, f'epoch_{epoch}_predictions.json'), 'w')
    #        trainer.test(to_evaluate, test_dataloaders=model.val_dataloader())


if __name__ == '__main__':
    main()
