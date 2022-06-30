_base_ = [
    './_base_/models/tfnet_r50-d32.py',
    './_base_/datasets/ddti_256.py',
    './_base_/default_runtime.py',
    './_base_/schedules/schedule_40k.py'
]
model = dict(test_cfg=dict(mode='whole'))
evaluation = dict(interval=2000)