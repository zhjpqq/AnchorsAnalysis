from torch.nn.modules.module import Module
from torch.nn.functional import avg_pool2d, max_pool2d
from ..functions.roi_align import RoIAlignFunction

# todo ??? 输出输入形状
# https://github.com/jwyang/faster-rcnn.pytorch/tree/master/lib/model
# https://github.com/jwyang/faster-rcnn.pytorch/blob/master/lib/model/roi_align/modules/roi_align.py


class RoIAlign(Module):
    """
    e.g.
    - inputs:  featurese: 1x256x256x256, rois: 32x5,  align_w_h:7x7
    - outputs: 32*256*7*7
    """
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlign, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return RoIAlignFunction(self.aligned_height, self.aligned_width,
                                self.spatial_scale)(features, rois)


class RoIAlignAvg(Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlignAvg, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        x = RoIAlignFunction(self.aligned_height + 1, self.aligned_width + 1,
                             self.spatial_scale)(features, rois)
        return avg_pool2d(x, kernel_size=2, stride=1)


class RoIAlignMax(Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlignMax, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        x = RoIAlignFunction(self.aligned_height + 1, self.aligned_width + 1,
                             self.spatial_scale)(features, rois)
        return max_pool2d(x, kernel_size=2, stride=1)
