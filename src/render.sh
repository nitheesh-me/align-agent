#!/bin/sh
# render.sh name -> name.png (snap typst can't read /tmp, so feed it stdin)
cd "$(dirname "$0")" && cat common.typ "$1.typ" | typst compile --format png --ppi 200 - - > "$1.png" && echo "$1.png ok"
