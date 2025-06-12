
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform

class TransformNothing(BasicTransform):
    def __init__(self, transform: BasicTransform):
        super().__init__()
        self.transform = transform

    def apply(self, data_dict: dict, **params) -> dict:
        return self.transform(**data_dict)
    
    def __repr__(self):
        ret_str = f"{type(self).__name__}(transform={self.transform})"
        return ret_str
