import argparse
import json

from sklearn.metrics import classification_report, confusion_matrix

from scimon.utils import (
    EXTRA_TOKENS,
    SEP_TOKEN,
)

def main():
    parser = argparse.ArgumentParser(description='Perform F1 scoring over textual versions of the output')
    parser.add_argument('--input', required=True, help='jsonl file of {generated: ..., target:...}')
    parser.add_argument('--output', required=False, help='output file')
    args = parser.parse_args()

    def fix_str(s):
        for t in EXTRA_TOKENS + [SEP_TOKEN, '<s>', '</s>']:
            s = s.replace(t, '')
            s = s.replace('  ', ' ')
        return s
    
    generations = []
    targets = []

    with open(args.input, 'r') as inf:
        for line in inf:
            content = json.loads(line)
            generations.append(fix_str(content['generated']))
            targets.append(fix_str(content['target']))
    
    str_map = {
        'significantly decreased': 0,
        'no significant difference': 1,
        'significantly increased': 2,
        'broken generation': 3,
    }
    
    for x in generations:
        if x not in str_map:
            print(x)
    generations = [str_map.get(x, str_map['broken generation']) for x in generations]
    targets = [str_map[x] for x in targets]
    
    scores = classification_report(targets, generations, digits=3, output_dict=True, target_names=list(str_map.keys())[:3])
    print(scores)
    confusions = confusion_matrix(targets, generations, normalize='true')
    print('confusion matrix')
    print(confusions)
    if args.output is not None:
        assert args.output != args.input
        with open(args.output, 'w') as of:
            of.write(json.dumps(scores))
            of.write('\n')
            of.write(json.dumps(confusions.tolist()))

if __name__ == '__main__':
    main()
    