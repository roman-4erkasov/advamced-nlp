import io
import datetime
import numpy as np


def log(msg:str, headers=None):
    dttm = (
        datetime.datetime.now()
        .strftime("%Y-%m-%d %H:%M:%S.%f")
    )
    if headers:
        prefix = "".join(
            f"[{x}]" for x in 
            [dttm] + headers
        )
    else:
        prefix = f"[{dttm}]"
    print(f"{prefix} {msg}")


def get_sentence(filename):
    sentences = []

    with io.open(filename, 'r', encoding='utf-8') as fd:
        # Skip header
        next(fd)

        sentence = []
        for line in fd:
            # Sentences are separated with `\n`
            if len(line.strip()) == 0:
                # Add only valid sentences
                if len(sentence) == 0:
                    continue
                sentences.append(sentence)
                sentence = []
                continue

            # Add word and POS-tag for train data
            word = line.strip().split('\t')[2]
            pos = line.strip().split('\t')[3].split('#')[0]
            sentence.append({'word': word, 'pos': pos})

        # Add last sentence
        if len(sentence) != 0:
            sentences.append(sentence)

    return sentences


def get_tag_mapping(sentences):
    # Special tags
    pos_to_idx = {
        '<PAD>': 0,
        '<UNK>': 1
    }
    idx_to_pos = {
        0: '<PAD>',
        1: '<UNK>'
    }

    for sentence in sentences:
        for pair in sentence:
            if pair['pos'] not in pos_to_idx:
                pos_to_idx[pair['pos']] = len(pos_to_idx)
                idx_to_pos[len(idx_to_pos)] = pair['pos']

    return pos_to_idx, idx_to_pos


def get_embeddings(sentences, path):
    # Special tags
    word_to_idx = {
        '<PAD>': 0,
        '<UNK>': 1
    }
    idx_to_word = {
        0: '<PAD>',
        1: '<UNK>'
    }
    word_to_emb = {
        '<PAD>': np.random.uniform(-0.25, 0.25, 300),
        '<UNK>': np.zeros(300)
    }

    with open(path, 'r') as fd:
        # Skip header
        next(fd)

        for line in fd:
            word = line.split('_')[0].lower()
            weight = np.array(list(map(float, line.split(' ')[1:])))
            word_to_emb[word] = weight

    embeddings = [word_to_emb['<PAD>'], word_to_emb['<UNK>']]

    for sentence in sentences:
        for pair in sentence:
            if pair['word'].lower() not in word_to_idx:
                word_to_idx[pair['word'].lower()] = len(word_to_idx)
                idx_to_word[len(idx_to_word)] = pair['word'].lower()
                embeddings.append(word_to_emb.get(pair['word'].lower(), np.random.uniform(-0.25, 0.25, 300)))

    return word_to_idx, idx_to_word, np.array(embeddings)


def iterate_batches(sentences, word_to_idx, pos_to_idx, batch_size=16):
    sorted_sentences = sorted(sentences, key=lambda item: -len(item))

    # Group sentences with same length in one batch (not always the same, but close)
    for start in range(0, len(sorted_sentences), batch_size):
        end = min(start + batch_size, len(sorted_sentences))
        subsample = sorted_sentences[start: end]
        max_sentence_length = len(subsample[0])

        x = np.zeros((max_sentence_length, len(subsample)))
        y = np.zeros((max_sentence_length, len(subsample)))

        for batch_ind, sample in enumerate(subsample):
            for idx, pair in enumerate(sample):
                x[idx, batch_ind] = word_to_idx.get(pair['word'], word_to_idx['<UNK>'])
                y[idx, batch_ind] = pos_to_idx.get(pair['pos'], pos_to_idx['<UNK>'])

        yield x, y
