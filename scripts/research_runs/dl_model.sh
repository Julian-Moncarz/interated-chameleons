#!/bin/bash
set -e
D=~/models/gemma-2-9b-it-abliterated
mkdir -p "$D"; cd "$D"
BASE=https://huggingface.co/IlyaGusev/gemma-2-9b-it-abliterated/resolve/main
# small files (sequential, fast)
for f in config.json generation_config.json model.safetensors.index.json special_tokens_map.json tokenizer.json tokenizer.model tokenizer_config.json; do
  curl -sL -o "$f" "$BASE/$f" && echo "got $f"
done
# 4 shards in parallel
for n in 1 2 3 4; do
  f=model-0000${n}-of-00004.safetensors
  curl -sL -o "$f" "$BASE/$f" -w "done $f %{size_download}B %{speed_download}B/s\n" &
done
wait
echo "ALL_DONE"
ls -lh "$D"
