python train.py \
    --domain metaworld \
    --task sweep-into-v2 \
    --episode-len 1000 \
    --discount 0.99 \
    --improve-step 5 \
    --pref-num 30 \
    --pref-max-iters 15 \
    --select-num 100 \
    --threshold 1.1 \
    --weight-decay 1e-3