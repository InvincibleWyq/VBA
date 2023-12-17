wordbank=(k_vit32_clip_freeze u_vit32_clip_freeze s_vit32_clip_temporal h_vit32_clip_temporal f_vit32_clip_temporal)
for word in ${wordbank[@]}
do
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/fewshot/matchingnet/vit32_main/${word}_5.py --validate --seed 42 --deterministic  # train
    sleep 0.1s
    CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/fewshot/matchingnet/vit32_main/${word}.py work_dirs/${word}_5/best_accuracy_mean.pth --seed 0 --deterministic  # test top-1 acc
    sleep 0.1s
    CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/fewshot/matchingnet/vit32_main/${word}_5.py work_dirs/${word}_5/best_accuracy_mean.pth --seed 0 --deterministic  # test top-5 acc
    sleep 0.1s
done
