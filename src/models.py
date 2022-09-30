from torch import nn


def CBOW_transform(raw, window_len, add_padding=False):
    span = list()
    interval = (window_len, len(raw) - window_len)

    for center_idx in range(interval[0], interval[1]):
        context = [raw[word_idx] for word_idx in range(center_idx - window_len, center_idx)]
        context += [raw[word_idx] for word_idx in range(center_idx + 1, center_idx + window_len)]
        target = raw[center_idx]

        span.append((context, target))

    return span


class CBOW(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x