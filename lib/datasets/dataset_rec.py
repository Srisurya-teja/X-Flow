"""
MXNet RecordIO dataset with FROM's on-the-fly occlusion generation.
Combines partialfc's RecordIO loading with FROM's occlude_img() pipeline.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import lib.core.utils as utils

Occluders = 'data/datasets/occluder/'
Occluders_List = 'data/datasets/occluder/occluder.txt'


class MXFaceDataset_Occ(Dataset):
    """
    RecordIO dataset with on-the-fly occlusion for FROM training.

    Args:
        root_dir: path to directory containing train.rec, train.idx, property
        mode: 'Clean', 'Occ', or 'Mask'
        img_size: (H, W) tuple, e.g. (112, 112)
        pattern: grid pattern for mask quantization
        ratio: 1-in-ratio samples stay clean in Mask/Occ mode
        transform: torchvision transform to apply
    """
    def __init__(self, root_dir, mode='Clean', img_size=(112, 112), pattern=5,
                 ratio=4, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.img_size = img_size
        self.ratio = ratio
        self.transform = transform

        # Load RecordIO
        import mxnet as mx
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

        # Read all indices
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            # InsightFace format: header contains range info
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

        # Occlusion setup
        self.grids = utils.get_grids(*img_size, pattern)

        # Load occluder list
        if os.path.exists(Occluders_List):
            self.occList = utils.occlist_reader(Occluders_List)
            self.occRoot = Occluders
            self.has_occluders = True
        else:
            self.has_occluders = False
            if mode in ['Mask', 'Occ']:
                print(f"WARNING: Occluder images not found at {Occluders_List}. "
                      f"Mode is '{mode}' but occlusion will be skipped!")

        print(f'MXFaceDataset_Occ: {len(self.imgidx)} images, mode={mode}, '
              f'pattern={pattern}, num_grids={len(self.grids)}, ratio={ratio}')

    def __len__(self):
        return len(self.imgidx)

    def PIL_reader(self, path):
        try:
            with open(path, 'rb') as f:
                return Image.open(f).convert('RGB')
        except IOError:
            print(f'Cannot load image {path}')
            return None

    def occlude_img(self, img):
        """Apply random occlusion and compute grid label."""
        if not self.has_occluders:
            return img, 0, None

        occPath = random.choice(self.occList)
        occ = self.PIL_reader(os.path.join(self.occRoot, occPath))
        if occ is None:
            return img, 0, None

        factor = random.choice(np.linspace(1, 5, 9, endpoint=True))
        img_occ, mask, _ = utils.occluded_image_ratio(img.copy(), occ, factor)

        mask_label = utils.cal_similarity_label(self.grids, mask)
        return img_occ, mask_label, mask

    def get_data(self, img):
        """Apply occlusion based on training mode."""
        if self.mode == 'Clean':
            return img, 0

        elif self.mode in ['Mask', 'Occ']:
            if random.choice(range(self.ratio)) == 0:
                return img, 0
            else:
                img_occ, mask_label, mask = self.occlude_img(img)
                return img_occ, mask_label
        else:
            raise ValueError(f'Unknown mode: {self.mode}')

    def __getitem__(self, index):
        import mxnet as mx

        idx = self.imgidx[index]

        # Read and decode from RecordIO
        for retry in range(5):
            try:
                s = self.imgrec.read_idx(idx)
                header, img_bytes = mx.recordio.unpack(s)
                img = mx.image.imdecode(img_bytes).asnumpy()  # BGR numpy array
                break
            except Exception as e:
                if retry == 4:
                    # Return a random other sample
                    return self.__getitem__(random.randint(0, len(self) - 1))
                continue

        # Convert BGR -> RGB -> PIL
        img = Image.fromarray(img[:, :, ::-1])

        # Resize to target size if needed
        if img.size != (self.img_size[1], self.img_size[0]):
            img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)

        # Get label
        label = int(header.label) if not isinstance(header.label, np.ndarray) else int(header.label[0])

        # Random horizontal flip
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Apply occlusion
        img, mask_label = self.get_data(img)

        # Transform
        if self.transform is not None:
            img = self.transform(img)

        return img, label, mask_label, str(idx)
