# Readers for labeled text formats

import re

from collections import OrderedDict


SPLIT_KEEP_WHITESPACE_RE = re.compile(r'(\s+)')


def parse_fasttext_line(l, label_string='__label__'):
    split, idx = SPLIT_KEEP_WHITESPACE_RE.split(l), 0

    # Skip initial whitespace or empties
    while idx < len(split) and split[idx].isspace() or not split[idx]:
        idx += 1
    
    # Collect and skip labels
    labels = []
    while idx < len(split) and split[idx].startswith(label_string):
        labels.append(split[idx][len(label_string):])
        idx += 1
        # Skip whitespace and empties after label
        while idx < len(split) and split[idx].isspace() or not split[idx]:
            idx += 1

    # The rest is the text
    text = ''.join(split[idx:])
    return text, labels


def parse_tsv_line(l):
    fields = l.split('\t', 1)
    labels, text = fields
    labels = labels.upper().split()
    #labels = [l for l in labels.split() if l.isupper()]
    return text, labels


def read_fasttext(f, fn, label_string='__label__'):
    for ln, l in enumerate(f, start=1):
        l = l.rstrip('\n')
        try:
            text, labels = parse_fasttext_line(l, label_string)
        except Exception as e:
            raise ValueError(f'failed to parse {fn} line {ln}: {e}: {l}')
        yield text, labels


def read_tsv(f, fn):
    for ln, l in enumerate(f, start=1):
        l = l.rstrip('\n')
        try:
            if len(l) == 0:
                print("empty line")
            text, labels = parse_tsv_line(l)
        except Exception as e:
            print(f'failed to parse {fn} line {ln}: {e}: {l}')
            print(len(l))
            #continue
            raise ValueError(f'failed to parse {fn} line {ln}: {e}: {l}')
        yield text, labels


READERS = OrderedDict([
    ('tsv', read_tsv),
    ('fasttext', read_fasttext),
])


def get_reader(format_name):
    try:
        return READERS[format_name]
    except KeyError:
        raise ValueError(f'unknown format {format_name}')
