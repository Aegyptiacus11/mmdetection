
# the new config inherits the base configs to highlight the necessary modification
_base_ = 'swin/retinanet_swin-t-p4-w7_fpn_1x_coco.py'

# 1. dataset settings
dataset_type = 'CocoDataset'
classes = ('meteor','nonmeteor')
data_root='/kaggle/input/datameteors/datamet'
backend_args = None

train_pipeline = [  # Training data processing pipeline
    dict(type='LoadImageFromFile', backend_args=backend_args),  # First pipeline to load images from file path
    dict(
        type='LoadAnnotations',  # Second pipeline to load annotations for current image
        with_bbox=True,  # Whether to use bounding box, True for detection
        with_mask=False,  # Whether to use instance mask, True for instance segmentation
        poly2mask=False),  # Whether to convert the polygon mask to instance mask, set False for acceleration and to save memory
    dict(type='PackDetInputs')  # Pipeline that formats the annotation data and decides which keys in the data should be packed into data_samples
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='train.json',
        data_prefix=dict(img='train'),
        pipeline=train_pipeline)
    )

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='validation.json',
        data_prefix=dict(img='valid')
        )
    )

val_evaluator = dict(  # Validation evaluator config
    type='CocoMetric',  # The coco metric used to evaluate AR, AP, and mAP for detection and instance segmentation
    ann_file=data_root + '/validation.json',  # Annotation file path
    metric=['bbox'],  # Metrics to be evaluated, `bbox` for detection
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator  # Testing evaluator config
'''
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='test/annotation_data',
        data_prefix=dict(img='test/image_data')
        )
    )
'''
# 2. model settings

# explicitly over-write all the `num_classes` field from default 80 to 5.
model = dict(
    bbox_head=dict(
        type='RetinaHead',
            num_classes=2))

max_epochs = 12
train_cfg = dict(max_epochs=max_epochs)
