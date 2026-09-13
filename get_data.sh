#!/bin/bash
# Fetch the public-domain corpus used in the paper (Project Gutenberg).
set -e
mkdir -p books
curl -sL https://www.gutenberg.org/cache/epub/11/pg11.txt   -o alice.txt          # Alice's Adventures in Wonderland (training)
for id in 12 1342 84 2701 98 1661 2600 1400 174 76 345 4300 135 2554 43 1952 120; do   # the last 12 are only needed to retrain engine_big.json (n8)                                                  # 12 = Through the Looking-Glass (TEST, never trained on)
  curl -sL "https://www.gutenberg.org/cache/epub/$id/pg$id.txt" -o "books/pg$id.txt"  # 1342 Pride & Prejudice, 84 Frankenstein, 2701 Moby Dick, 98 Tale of Two Cities
done
ls -la alice.txt books/
