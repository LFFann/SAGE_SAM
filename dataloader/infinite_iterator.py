"""Small infinite iterator utility for unequal labeled/unlabeled loaders."""

from __future__ import annotations


def infinite_iterator(loader):
    while True:
        for batch in loader:
            yield batch
