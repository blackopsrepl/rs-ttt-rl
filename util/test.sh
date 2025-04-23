#!/usr/bin/env bash
## Test all code in the repostitory using cargo test.

for DIR in src; do
    DIRNAME=$(basename "$DIR")
    echo "==> $DIRNAME <=="
    (cd $DIR && cargo test)
done

echo "Test complete."
