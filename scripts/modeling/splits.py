import argparse
import json
import os

def main():
    parser = argparse.argumentparser(description='select a subset of a reviews file by s2id')
    parser.add_argument('--input_reviews', required=True, help='input reviews file, jsonl formatted')
    parser.add_argument('--output_dir', required=True, help='output reviews file')
    parser.add_argument('--test_ids', required=True, help='s2ids file')
    parser.add_argument('--train_ids', required=True, help='s2ids file')
    parser.add_argument('--val_ids', required=True, help='s2ids file')
    args = parser.parse_args()

    def read_ids(ids_file):
        with open(ids_file, 'r') as idf:
            ids = set(map(str.strip, idf))
            return ids
    train_ids = read_ids(args.train_ids)
    val_ids = read_ids(args.val_ids)
    test_ids = read_ids(args.test_ids)
    
    train_file = os.path.join(args.output_dir, 'train.jsonl')
    val_file = os.path.join(args.output_dir, 'val.jsonl')
    test_file = os.path.join(args.output_dir, 'test.jsonl')
    
    shared_ids = train_ids & val_ids & test_ids
    assert len(shared_ids) == 0

    with open(train_file, 'w') as train_f, \
        open(val_file, 'w') as val_f, \
        open(test_file, 'w') as test_f:
        for input_review in args.input_reviews.split(','):
            with open(input_review, 'r') as inf:
                for line in inf:
                    content = json.loads(line)
                    eyeD = str(content['s2id'])
                    if eyeD in train_ids:
                        train_f.write(line)
                    elif eyeD in val_ids:
                        val_f.write(line)
                    elif eyeD in test_ids:
                        test_f.write(line)
                    else:
                        print(f'Unknown id {eyeD}')

if __name__ == '__main__':
    main()
    