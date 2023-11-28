# Video Bypass Aggregator (VBA)

Undergraduate Thesis

Title: Understanding Few-shot Video with Pretrained Image-Text Models

Abstract: 
Most existing few-shot video classification methods do not consider the transfer of pretrained knowledge in their design. The accuracy of these methods is sub-optimal when directly introducing pretrained knowledge, and the transfer efficiency is relatively poor. On the other hand, existing video classification methods for full-sample supervised learning always encode each sample independently, making it difficult to jointly encode all samples in the support and query sets in few-shot tasks.

To effectively transfer the scene modeling ability of pretrained models to the video domain and achieve fine accuracy in few-shot video classification tasks, this paper proposes the Video Bypass Aggregator (VBA) structure. Through cross-layer information aggregation, the method effectively transfers the multi-granularity scene modeling ability of pretrained features. Through cross-frame information aggregation, the method fully learns the temporal modeling ability for videos. Through cross-video information aggregation, the method uses all sample information in the support and query sets to obtain more robust feature representations for each sample.

The proposed method achieves higher or comparable accuracy compared to existing methods with the same settings on 6 mainstream video datasets and 7 few-shot splits, demonstrating the overall effectiveness of the method. The ablation experiments demonstrate the role of each module. The paper also demonstrates the training efficiency of the method, shows its cross-domain understanding ability, and provides qualitative visualizations of attention maps.
