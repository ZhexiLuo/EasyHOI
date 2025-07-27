#!/bin/bash

# 用法: ./copy_gpt.sh 源目录 目标目录
src_dir="$1"
dst_dir="$2"

if [[ -z "$src_dir" || -z "$dst_dir" ]]; then
    echo "用法: $0 <源目录> <目标目录>"
    exit 1
fi

if [ ! -d "$dst_dir" ]; then
    mkdir -p "$dst_dir"
fi

count=1

# 查找并拷贝
find "$src_dir" -type f -name "gpt.png" | while read -r file; do
    cp "$file" "$dst_dir/gpt_${count}.png"
    echo "复制: $file --> $dst_dir/gpt_${count}.png"
    count=$((count + 1))
done
