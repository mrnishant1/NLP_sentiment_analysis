import random
def mask_input_ids(ids,vocab_size, mask_prob=0.15,):
    """ids: 1D list/array of token ids (already has CLS/SEP/PAD).
    Returns (masked_ids, labels) where labels[i] = -100 for positions not
    used in the loss, or the ORIGINAL token id at positions selected for prediction."""
    PAD_ID = 0
    SEP_ID = 3
    MASK_ID = 4
    ids = list(ids)
    labels = [-100] * len(ids)

    for i in range(1, len(ids) - 1):          # skip CLS (pos 0) and final SEP/pad slot
        if ids[i] in (PAD_ID, SEP_ID):
            continue                           # never mask padding or SEP
        if random.random() < mask_prob:
            labels[i] = ids[i]
            r = random.random()
            if r < 0.8:
                ids[i] = MASK_ID
            elif r < 0.9:
                ids[i] = random.randint(5, vocab_size - 1)   # skip special-token ids 0-4
            # else: leave unchanged (10%) - label still set above
    return ids, labels