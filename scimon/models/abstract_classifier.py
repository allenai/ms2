"""Models and simple code for working with a CSV to generate classifications.

This code exists to train/evaluate two types of classifiers:
- one on entire abstracts, as whether or not they are suitable for inclusions
- one on abstract *sentences* to specify their types (e.g. background/goal/
  results/varying types of conclusion statements)

The classes are inferred from the input CSVs
"""
import argparse
import logging
import os
import random
import numpy as np

from collections import Counter
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl

import torch
import torch.nn as nn

from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from transformers.optimization import get_linear_schedule_with_warmup

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)


# this separate model exists because PyTorch lightning wants more things in and
# out of its serialization than we need
class AbstractTagger(nn.Module):
    def __init__(self, model_name, classes, tokenizer, model_dir):
        super(AbstractTagger, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(self.model.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.model.config.hidden_size, len(classes))
        self.classes = classes
        self.tokenizer = tokenizer
        self.model_dir = model_dir

    def forward(self, input_ids, labels=None):
        attention_mask = (input_ids != self.tokenizer.pad_token_id)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        sent_sep_embeddings = outputs[0][input_ids == self.tokenizer.sep_token_id]
        sent_sep_embeddings = self.dropout(sent_sep_embeddings)
        logits = self.classifier(sent_sep_embeddings)
        loss = None
        flat_labels = labels
        if labels is not None:
            flat_labels = labels[labels != -100]
            assert flat_labels.size(0) == sent_sep_embeddings.size(0)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, flat_labels)
        return loss, logits, flat_labels

    def decode(self, texts: List[str], max_length: int, group_in_one_abstract: bool):
        input_ids_list = []
        input_ids = []
        for text in texts:
            tokens = self.tokenizer.encode(text, truncation=True, max_length=max_length)
            if len(input_ids) + len(tokens) > max_length or not group_in_one_abstract:
                input_ids_list.append(input_ids)
                input_ids = []
            if len(input_ids) > 0:
                tokens = tokens[1:]  # drop the leading <s>
            input_ids.extend(tokens)
        if len(input_ids) > 0:
            input_ids_list.append(input_ids)
        input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(t) for t in input_ids_list], batch_first=True, padding_value=self.tokenizer.pad_token_id).long()
        loss, logits, flat_labels = self.forward(input_ids.cuda())
        assert loss is None
        assert flat_labels is None
        pred_labels = torch.argmax(logits, dim=1)
        dist = torch.softmax(logits, dim=1)
        assert len(pred_labels) == len(texts)
        return pred_labels, dist


class LightningAbstractTagger(pl.LightningModule):

    def __init__(self, args, model_name, classes, tokenizer, model_dir):
        super(LightningAbstractTagger, self).__init__()
        self.save_hyperparameters()
        self.args = args
        self.model = AbstractTagger(model_name, classes, tokenizer, model_dir)

    def training_step(self, batch, batch_idx):
        names, input_ids, labels = batch
        loss, logits, flat_labels = self.forward(input_ids=input_ids, labels=labels)
        preds = torch.argmax(logits, dim=1)
        report = classification_report(
            flat_labels.cpu().numpy(),
            preds.detach().cpu().numpy(),
            labels=list(range(len(self.model.classes))),
            target_names=self.model.classes,
            output_dict=True,
            zero_division=0)
        accuracy = accuracy_score(flat_labels.cpu().numpy(), preds.detach().cpu().numpy())

        return {
            'scores': torch.softmax(logits, dim=1).detach().cpu(),
            'accuracy': accuracy,
            'loss': loss,
            'preds': preds,
            'labels': flat_labels,
            'log': {
                **report['macro avg'],
                'loss': loss,
                'accuracy': accuracy,
            }
        }

    def forward(self, input_ids, labels=None):
        return self.model.forward(input_ids=input_ids, labels=labels)

    def decode(self, texts: List[str], max_length: int):
        return self.model.decode(texts, max_length, group_in_one_abstract=self.args.seq)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        dataset_size = self.train_dataloader.dataloader.dataset.__len__()
        num_steps = dataset_size * self.args.epochs / self.args.grad_accum / self.args.batch_size
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_steps * 0.1, num_training_steps=num_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().cpu()
        preds = torch.cat([x['preds'] for x in outputs]).cpu().numpy()
        labels = torch.cat([x['labels'] for x in outputs]).cpu().numpy()
        report = classification_report(
            labels,
            preds,
            labels=list(range(len(self.model.classes))),
            target_names=self.model.classes,
            output_dict=True,
            zero_division=0)
        accuracy = accuracy_score(labels, preds)
        logging.info(f'loss: {avg_loss}, accuracy: {accuracy}, macro avg: {report["macro avg"]}')
        return {
            'accuracy': accuracy,
            'val_loss': avg_loss,
            'log': report['macro avg'],
        }

    def test_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().cpu()
        preds = torch.cat([x['preds'] for x in outputs]).cpu().numpy()
        labels = torch.cat([x['labels'] for x in outputs]).cpu().numpy()
        if len(self.model.classes) == 2:
            scores = torch.cat([x['scores'] for x in outputs]).cpu().numpy()
            scores = scores[:, 1]
            auc = roc_auc_score(
                labels,
                scores,
                average='macro')
            logging.info('auc: {}'.format(auc))
            precisions, recalls, thresholds = precision_recall_curve(
                labels,
                scores,
                pos_label=1
            )
            logging.info('precisions: {}'.format(precisions))
            logging.info('recalls: {}'.format(recalls))
            logging.info('thresholds: {}'.format(thresholds))
            plt.ioff()
            fig, ax = plt.subplots()
            line_kwargs = {"drawstyle": "steps-post", 'label': 'prf'}
            line_ = ax.plot(recalls, precisions, **line_kwargs)
            ax.set(xlabel="Recall", ylabel="Precision")
            plt.savefig(os.path.join(self.model.model_dir, 'test_prf.png'))
        report = classification_report(
            labels,
            preds,
            labels=list(range(len(self.model.classes))),
            target_names=self.model.classes,
            output_dict=True,
            zero_division=0)
        for key, value in report.items():
            if type(value) != dict:
                logging.info(f'{key}: {value:.2f}')
                continue
            p = value['precision']
            r = value['recall']
            f = value['f1-score']
            s = value['support']
            logging.info(f'p: {p:.2f}, r: {r:.2f}, f: {f:.2f}, s: {s} - {key}')

        conf = confusion_matrix(
            labels,
            preds,
            normalize='true')
        logging.info('confusion matrix\n{}'.format(conf))
        accuracy = accuracy_score(labels, preds)
        logging.info(f'loss: {avg_loss}, accuracy: {accuracy}, macro avg: {report["macro avg"]}')
        return {
            'accuracy': accuracy,
            'test_loss': avg_loss,
            'log': report['macro avg'],
        }


class AbstractsDataset(Dataset):
    """Reads strings and values from a CSV"""
    def __init__(
            self,
            csv_path: str,
            instance_name_field: str,
            instance_text_field: str,
            instance_cls_field: str,
            tokenizer,
            classes: List[str],
            max_length: int,
            seq: bool,
            limit_classes: bool):
        super(AbstractsDataset, self).__init__()
        df = pd.read_csv(csv_path).fillna(value="MISSING!")
        df = df[df[instance_cls_field] != "MISSING!"]
        if classes is None:
            classes = set(filter(lambda kls: ',' not in kls, df[instance_cls_field]))
        self.classes = list(classes)
        self.intern_class = dict(((x, i) for (i, x) in enumerate(classes)))
        AbstractsDataset.tokenizer = tokenizer
        self.max_length = max_length
        self.instances = []
        for row in df[[instance_name_field, instance_text_field, instance_cls_field]].itertuples(index=False):
            self.instances.extend(self._elem_to_training_instance(row))
        if seq:
            merged_instances = []
            prev_s2id = None
            found_ids = set()
            merged_tokens_one_instance = []
            merged_labels_one_instance = None
            for instance in self.instances:
                s2id, tokens, labels = instance
                if s2id != prev_s2id or len(merged_tokens_one_instance) + len(tokens) > self.max_length:
                    if prev_s2id is not None:
                        assert len(merged_labels_one_instance) == merged_tokens_one_instance.count(self.tokenizer.sep_token_id)
                        merged_instances.append((s2id, merged_tokens_one_instance, merged_labels_one_instance))

                    merged_tokens_one_instance = []
                    merged_labels_one_instance = []
                    prev_s2id = s2id
                    if s2id in found_ids:
                        logging.error(f'repeated s2id: {s2id}')
                    found_ids.add(s2id)
                if len(merged_tokens_one_instance) > 0:
                    tokens = tokens[1:]  # drop the leading <s>
                merged_tokens_one_instance.extend(tokens)
                merged_labels_one_instance.extend(labels)
            merged_instances.append((s2id, merged_tokens_one_instance, merged_labels_one_instance))
            self.instances = merged_instances

        if limit_classes:
            for instance in self.instances:
                labels = None
                new_labels = []
                for label in instance[2]:
                    if label == self.intern_class['BACKGROUND'] or label == self.intern_class['GOAL']:
                        new_labels.append(1)
                    elif label == self.intern_class['EFFECT']:
                        new_labels.append(2)
                    else:
                        new_labels.append(0)
                instance[2].clear()
                instance[2].extend(new_labels)
            self.classes = ['ETC', 'BACKGROUND', 'EFFECT']
            self.intern_class = {'ETC': 0, 'BACKGROUND': 1, 'EFFECT': 2}

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]

    def _elem_to_training_instance(self, elem):
        name, text, kls = elem
        # in initial annotation of the abstract sentence classes, some contain a
        # mix of information so instead of making a decision we punted and gave
        # it two classes.
        # as that provides unclear signal, we omit these instances
        if ',' in kls:
            classes = kls.split(',')
            return []
        else:
            classes = [kls]
        ret = []
        # auto trim instances
        text = self.tokenizer(text, truncation=True, max_length=self.max_length)['input_ids']
        for kls in classes:
            kls = self.intern_class[kls]
            ret.append((name, text, [kls]))
        return ret

    @staticmethod
    def collate_fn(instances):
        pad_token_id = AbstractsDataset.tokenizer.pad_token_id
        (names, texts, kls) = zip(*instances)
        input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(t) for t in texts], batch_first=True, padding_value=pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in kls], batch_first=True, padding_value=-100)

        return names, input_ids, labels


def main():
    parser = argparse.ArgumentParser(description='train and evaluate an abstract (or text) classifier from a CSV input')
    parser.add_argument('--model_dir', required=True, help='Training dir')
    parser.add_argument('--train', required=True, help='Training dataset')
    parser.add_argument('--test', required=True, help='Testing dataset')
    parser.add_argument('--name_field', required=True, help='Some field to grab an id')
    parser.add_argument('--text_field', required=True, help='Some field to grab text for classification')
    parser.add_argument('--label_field', required=True, help='Some field to grab the label')
    parser.add_argument('--model', default='roberta-large', help='BERT model?')
    parser.add_argument("--seed", type=int, default=1234, help="Seed")
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--grad_accum', default=1, type=int, help='gradient accumulation')
    parser.add_argument('--epochs', default=5, type=int, help='epochs')
    parser.add_argument('--save_file', required=False, help='Where to save an output file')
    parser.add_argument('--seq', action='store_true', help='Sequence labeling')
    parser.add_argument('--limit_classes', action='store_true', help='Background, effect, etc')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    classes = None
    max_length = 512

    logger = TensorBoardLogger(
        save_dir=os.path.join(args.model_dir, 'logs')
    )
    logging.info('Loading data')
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    def flatten_list_of_list(nested_list):
        return [item for sublist in nested_list for item in sublist]

    train_data = AbstractsDataset(args.train, args.name_field, args.text_field, args.label_field, tokenizer, classes, max_length, args.seq, args.limit_classes)
    classes = train_data.classes
    training_distribution = Counter(flatten_list_of_list([(train_data.classes[cls_id] for cls_id in inst[-1]) for inst in train_data.instances]))
    logging.info('Training distribution {}'.format(training_distribution))

    if args.limit_classes:
        classes = None
    test_data = AbstractsDataset(args.test, args.name_field, args.text_field, args.label_field, tokenizer, classes, max_length, args.seq, args.limit_classes)
    if args.limit_classes:
        classes = train_data.classes
        assert train_data.classes == test_data.classes

    testing_distribution = Counter(flatten_list_of_list([(train_data.classes[cls_id] for cls_id in inst[-1]) for inst in test_data.instances]))
    logging.info('Testing distribution {}'.format(testing_distribution))

    logging.info('Loading model')
    model = LightningAbstractTagger(args, args.model, classes, tokenizer, args.model_dir)

    logging.info('Loaded {} training examples, {} test examples'.format(len(train_data), len(test_data)))
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=AbstractsDataset.collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=AbstractsDataset.collate_fn)
    logging.info('Creating trainer')
    trainer = pl.Trainer(
        distributed_backend=None,
        replace_sampler_ddp=False,
        default_root_dir=args.model_dir,
        max_epochs=args.epochs,
        gpus=1,
        logger=logger,
        show_progress_bar=True,
        log_save_interval=1,
        row_log_interval=1,
        precision=16, amp_level='O2',
        accumulate_grad_batches=args.grad_accum,
        checkpoint_callback=None,
    )
    # TODO resume from checkpoint!
    logging.info('Training!')
    trainer.fit(model=model, train_dataloader=train_dataloader, val_dataloaders=test_dataloader)  # super cheating
    trainer.test(model=model, test_dataloaders=test_dataloader)
    if args.save_file:
        torch.save(model.model, args.save_file)

    sample_abstract = 'From three trials in more severe OAG, there is some evidence that medication was associated with more progressive visual field loss and 3 to 8 mmHg less IOP lowering than surgery. In the longer-term (two trials) the risk of failure of the randomised treatment was greater with medication than trabeculectomy (OR 3.90, 95% CI 1.60 to 9.53; hazard ratio (HR) 7.27, 95% CI 2.23 to 25.71). Medications and surgery have evolved since these trials were undertaken. Evidence from one trial suggests that, beyond five years, the risk of needing cataract surgery did not differ according to initial treatment policy (OR 0.63, 95% CI 0.15 to 2.62). Methodological weaknesses were identified in all the trials. AUTHORS CONCLUSIONS\nPrimary surgery lowers IOP more than primary medication but is associated with more eye discomfort. One trial suggests that visual field restriction at five years is not significantly different whether initial treatment is medication or trabeculectomy. There is some evidence from two small trials in more severe OAG, that initial medication (pilocarpine, now rarely used as first line medication) is associated with more glaucoma progression than surgery. Beyond five years, there is no evidence of a difference in the need for cataract surgery according to initial treatment. Further RCTs of current medical treatments compared with surgery are required, particularly for people with severe glaucoma and in black ethnic groups. Economic evaluations are required to inform treatment policy.'
    sentences = sample_abstract.split('. ')
    sentences = [s + '.' for s in sentences]  # pyt the period back
    model = model.eval()
    for p in model.parameters():
        p.requires_grad = False
    labels, _ = model.decode(sentences, max_length)
    for label_idx, sentence in zip(labels, sentences):
        label = classes[label_idx]
        logging.info(f'{label} - {sentence}')


if __name__ == '__main__':
    main()
