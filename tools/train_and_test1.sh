wordbank=(d_vit32_clip_temporal g_vit32_clip_temporal)
for word in ${wordbank[@]}
do
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/fewshot/matchingnet/vit32_main/${word}.py --validate --seed 42 --deterministic  # train
    sleep 0.1s
    CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/fewshot/matchingnet/vit32_main/${word}.py work_dirs/${word}/best_accuracy_mean.pth --seed 0 --deterministic  # test top-1 acc
    sleep 0.1s
    CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/fewshot/matchingnet/vit32_main/${word}_5.py work_dirs/${word}/best_accuracy_mean.pth --seed 0 --deterministic  # test top-5 acc
    sleep 0.1s
done
