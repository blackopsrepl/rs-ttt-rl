#!/usr/bin/env bash
## build all code directories in the repostitory using cargo clippy.

for DIR in src; do
    DIRNAME=$(basename "$DIR")
    echo "==> $DIRNAME <=="
    (cd $DIR && cargo build --release)
done

echo "Build complete."