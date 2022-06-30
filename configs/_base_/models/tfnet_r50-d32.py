norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='TFNet',
    backbone=dict(
        type='ResNetV1cMoreFeature',
        depth=50,
        num_stages=4,
        out_channels=(3, 64, 256, 512, 1024, 2048),
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnet50_v1c')),
    feat_sizes=(32, 16, 8),
    patch_sizes=(4, 2, 1),
    tf_out_channels=(256, 256, 256),
    decoder_channels=(256, 256),
    low_feat_index=-4,
    low_feat_out_channels=48,
    first_up=8,
    second_up=4,
    msa_heads=(4, 4, 4),
    mca_heads=(4, 4, 4),
    position_embed='condition',
    classes=2,
    activation=None,
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=4,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    loss=dict(
        type='TFLoss',
        kl_weight=0.1,
        bce_weight=1.0),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
