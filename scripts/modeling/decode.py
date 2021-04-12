import argparse
import json
import logging
import torch

#from ms2.data.review_datasets import ReviewDataset
from ms2.data.review_datasets import ReviewDataset, ToUnflattenedModelInputsFunction
from ms2.models.transformer_summarizer import LightningBartSummarizer

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description='Summarize a document using a saved checkpoint')
    parser.add_argument('--input', required=True, help='Dataset for decoding')
    parser.add_argument('--output', required=True, help='Output file')
    parser.add_argument('--checkpoint', required=True, help='Saved checkpoint')
    LightningBartSummarizer.add_args(parser)
    args = parser.parse_args()
    model = LightningBartSummarizer(args)
    model.load_state_dict(torch.load(args.checkpoint)['state_dict'])
    model.cuda()
    collate_fn = model.summarizer.collate_fn
    dataset = ReviewDataset.from_file(args.input, format_function=ToUnflattenedModelInputsFunction(model.config.pad_token_id))
    logging.info(f'Output file: {args.output}')
    with open(args.output, 'w') as output:
        assert output is not None
        for instance in dataset:
            inputs, preambles, targets = collate_fn([instance])
            # defaults to try: https://github.com/huggingface/transformers/blob/v2.10.0/examples/summarization/bart/evaluate_cnn.py#L26-L40
            outputs = model.summarizer.generate_summary(
                inputs=inputs.cuda(),
                preambles=preambles.cuda(),
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                max_length=args.max_length,
                min_length=args.min_length,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                early_stopping=True,
                decoder_start_token_id=model.config.bos_token_id,
                repetition_penalty=args.repetition_penalty,
                temperature=args.temperature,
            )
            generated = model.tokenizer.decode(outputs[0].cpu(), skip_special_tokens=False)
            target = model.tokenizer.decode(targets.data[0], skip_special_tokens=False)
            logging.info('Generated: {}'.format(generated))
            logging.info('Target:    {}'.format(target))
            output.write(json.dumps({
                'preamble': model.tokenizer.decode(preambles.squeeze()),
                'generated': generated,
                'target': target,
            }))
            output.write('\n')
            output.flush()

if __name__ == '__main__':
    main()