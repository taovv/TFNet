import os.path as osp

from mmseg.datasets.custom import CustomDataset
from mmseg.datasets.builder import DATASETS


@DATASETS.register_module()
class BreastDataset(CustomDataset):
    """DRIVE datasets.
    In segmentation map annotation for DRIVE, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '_manual1.png'.
    """

    CLASSES = ('background', 'tumor')

    PALETTE = [[120, 120, 120], [6, 230, 230]]

    def __init__(self, **kwargs):
        super(BreastDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)