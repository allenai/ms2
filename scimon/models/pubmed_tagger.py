"""Multiclass tagger for Pubmed Publication Types

The pubmed publication types category seems to be under-populated in pubmed
articles, so we developed this tagger based on abstracts.

The abstracts classifier is an unsuitable replacement for this because this is
(1) multiclass and (2) trained via negative sampling.

There's no real way of evaluating performance at this time, and for the moment
it is not used in the main review processing pipeline.
"""
import argparse
import glob
import json
import logging
import operator
import os
import random

from collections import defaultdict
from typing import Dict, List, Union

import _jsonnet
import pytorch_lightning as pl

import torch
import torch.nn as nn

from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizerFast

from scimon.models.utils import PaddedSequence

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

random.seed(2468)

class PubmedTagger(pl.LightningModule):

    def __init__(self, bert_name, classes):
        super(PubmedTagger, self).__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.lin = nn.Linear(self.bert.config.hidden_size, len(classes))
        self.classes = classes
    
    def forward(self, tokens, token_mask, labels=None, labels_mask=None):
        """

        Args:
            tokens (torch.LongTensor): bs * len token ids
            token_mask (torch.FloatTensor): bs * len mask; elements are 1.0 for on, 0.0 for off
            labels (torch.LongTensor, optional): bs * num_classes. The true classes associated with each instance in the batch. Defaults to None.
            labels_mask (torch.LongTensor, optional): bs * num_classes. A mask for classes to ignore for each instance in the batch. Elements are 1.0 for on, 0.0 for off. Defaults to None.

        Returns:
            Tuple matching HuggingFace BERTs: (?loss, logits, hidden states, attentions)
            loss (torch.FloatTensor of shape (1,), optional): returned when `labels` are present
            logits (torch.FloatTensor of shape (bs * num_classes)): pre-sigmoid output from the multiclass layer
            hidden states: as in HuggingFace BERT
            attentions: as in HuggingFace BERT
        """
        bert_outputs = self.bert(tokens, token_mask)
        last_hidden_states = bert_outputs[0]
        cls_tokens = bert_outputs[0][:,0]
        logits = self.lin(cls_tokens)
        outputs = (logits, bert_outputs[-2], bert_outputs[-1])
        if labels is not None:
            if labels_mask is not None:
                logits *= labels_mask
                #logits = logits.to(dtype=torch.LongTensor)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            outputs = (loss,) + outputs
        else:
            loss = None
        return outputs

    def training_step(self, batch, batch_idx):
        text: PaddedSequence
        labels: torch.LongTensor
        labels_mask: torch.FloatTensor
        text, labels, labels_mask = batch
        device = next(self.parameters()).device
        text = text.to(device=device)
        mask = text.mask(on=1, off=0, dtype=torch.float, device=device)
        labels = labels.to(device=device)
        loss, logits, _, _ = self.forward(text.data, mask, labels=labels, labels_mask=labels_mask)
        preds = torch.round(nn.functional.sigmoid(logits))
        report = classification_report(labels.cpu().numpy(), preds.detach().cpu().numpy(), target_names=self.classes, output_dict=True, zero_division=0)
        acc = sum(preds.masked_select(labels_mask.to(torch.bool)) == labels.masked_select(labels_mask.to(torch.bool))) / labels_mask.sum()
        return {
            'batch_log_metrics': loss.item(),
            'loss': loss,
            'acc': acc,
            'labels': labels_mask.sum().item(),
            'log': {
                'f1': report['macro avg']['f1-score'],
                'loss': loss.item(),
                'train_loss': loss.item(),
            }
        }

    def validation_step(self, batch, batch_idx):
        text: PaddedSequence
        labels: torch.LongTensor
        labels_mask: torch.FloatTensor
        text, labels, labels_mask = batch
        device = next(self.parameters()).device
        text = text.to(device=device)
        mask = text.mask(on=1, off=0, dtype=torch.float, device=device)
        labels = labels.to(device)
        loss, logits, _, _ = self.forward(text.data, mask, labels=labels, labels_mask=None)
        preds = torch.round(nn.functional.sigmoid(logits))
        report = classification_report(labels.cpu().numpy(), preds.detach().cpu().numpy(), target_names=self.classes, output_dict=True, zero_division=0)
        acc = sum(preds.masked_select(labels_mask.to(torch.bool)) == labels.masked_select(labels_mask.to(torch.bool))) / labels_mask.sum()
        return {
            'val_loss': loss,
            'val_acc': acc,
            'labels': labels_mask.sum().item(),
        }

    def validation_epoch_end(self, outputs):
        return {
            'val_loss': sum(output['val_loss'] for output in outputs),
            'val_acc': torch.mean(torch.tensor([output['val_acc'] for output in outputs])),
            'labels': torch.sum(torch.tensor([output['labels'] for output in outputs])),
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class PubmedDataset(Dataset):

    def __init__(
            self,
            inputs_dir: str,
            sample_negatives: bool,
            tokenizer,
            classes: List[str],
            max_length: int,
            truncate_extra: bool=True):
        self.classes = classes
        self.intern_class = dict(((x,i) for (i,x) in enumerate(classes)))
        self.tokenizer = tokenizer
        self.max_length = max_length
        positives = []
        negatives = []
        for f in glob.glob(os.path.join(inputs_dir, 'targets', '*')):
            positives.extend(read_jsonl(f))
            if len(positives) > 1000000:
                logging.info('not loading all data due to memory constraints')
                break
        for f in glob.glob(os.path.join(inputs_dir, 'etc', '*')):
            negatives.extend(read_jsonl(f))
            if len(negatives) > 5 * len(positives):
                logging.info('not loading all data due to memory constraints')
                break
        for p in positives:
            p['source'] = 'positive'
        for n in negatives:
            n['source'] = 'negative'
        if sample_negatives:
            negatives = random.sample(negatives, len(positives))
        self.positives = positives
        self.negatives = negatives
        all_data = positives + negatives
        random.shuffle(all_data)
        # turn the data into a list of [text, labels, label_mask]
        self.instances = list(map(self._elem_to_training_instance, all_data))

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]

    def _elem_to_training_instance(self, elem):
        # auto trim instances
        text = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(elem['text'])[:self.max_length])
        publication_types = elem['publication_types']
        publication_types = list(map(lambda t: self.intern_class[t], publication_types))
        types = torch.zeros((len(self.classes),))
        types[publication_types] = 1
        if elem['source'] == 'positive':
            mask = types
        elif elem['source'] == 'negative':
            mask = torch.ones((len(self.classes,)))
        else:
            raise ValueError('impossible state with unknown elem {}'.format(elem))
        return text, types, mask
    
    @staticmethod
    def collate_fn(instances):
        texts, labels, label_masks = zip(*instances)
        texts = PaddedSequence.autopad(texts, batch_first=True, padding_value=0)
        labels = torch.stack(labels)
        label_masks = torch.stack(label_masks)
        return texts, labels, label_masks

def read_jsonl(f: str) -> List[Union[Dict, List]]:
    with open(f, 'r') as inf:
        return list(map(json.loads, inf))
 
def main():
    parser = argparse.ArgumentParser(description='train and evaluate pubmed multiclass tagger')
    parser.add_argument('--model_dir', required=True, help='Training dir')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads(_jsonnet.evaluate_file(args.config))
    logging.info(config)

    model_name = config['model_name']
    train_dir = config['train']
    val_dir = config['val']
    batch_size = config['batch_size']
    max_epochs = config['epochs']
    classes = config['classes']
    max_length = config['max_length']

    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        version=1,
        name='lightning_logs'
    )

    logging.info('Loading model')
    model = PubmedTagger(model_name, classes)
    model = model.cuda()

    logging.info('Loading data')
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    train_data = PubmedDataset(train_dir, True, tokenizer, classes, max_length)
    val_data = PubmedDataset(val_dir, False, tokenizer, classes, max_length)
    logging.info('Loaded {} training examples, {} validation_examples'.format(len(train_data), len(val_data)))
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=PubmedDataset.collate_fn)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=PubmedDataset.collate_fn)
    logging.info('Creating trainer')
    trainer = pl.Trainer(default_root_dir=args.model_dir, max_epochs=max_epochs, gpus=1, logger=logger)
    # TODO resume from checkpoint!
    logging.info('Training!')
    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == '__main__':
    main()
