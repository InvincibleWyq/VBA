_base_ = ['u_vit32_clip_freeze.py']
data = dict(
    train=dict(num_shots=5),
    test=dict(num_shots=5, meta_test_cfg=dict(support=dict(batch_size=5 * 5))))
