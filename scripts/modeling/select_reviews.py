import argparse
import json

def main():
    parser = argparse.ArgumentParser(description='Select a subset of a reviews file by s2id')
    parser.add_argument('--input_reviews', required=True, help='Input reviews file, jsonl formatted')
    parser.add_argument('--output_reviews', required=True, help='Output reviews file')
    parser.add_argument('--ids', required=True, help='s2ids file')
    args = parser.parse_args()

    with open(args.ids, 'r') as ids_file:
        ids = set(map(str.strip, ids_file))

    already_written = set()
    skipped = 0
    with open(args.output_reviews, 'w') as of:
        for input_review in args.input_reviews.split(','):
            with open(input_review, 'r') as inf:
                for line in inf:
                    content = json.loads(line)
                    title = content['title'].lower()
                    if 'redact' in title or 'withdraw' in title:
                        skipped += 1
                        continue
                    eyeD = str(content['s2id'])
                    if eyeD in ids:
                        if eyeD not in already_written:
                            of.write(line)
                            already_written.add(eyeD)
    print(f'skipped {skipped} reviews as redacted or withdrawn, wrote {len(already_written)} reviews')

if __name__ == '__main__':
    main()
    