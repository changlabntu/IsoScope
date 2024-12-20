import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from utils.data_utils import imagesc
import os
import tifffile as tiff
import numpy as np


class PairedCubes(data.Dataset):
    def __init__(self, root, path, opt=None, mode='train', index=None, filenames=None, labels=None, crop=None):
        self.directions = path.split('_')
        self.labels = labels
        self.root = root
        self.crop = opt.cropsize

        # Get the list of file names from the first folder
        folder_a = os.path.join(root, self.directions[0])
        self.file_names = sorted([f for f in os.listdir(folder_a) if f.endswith('.tif')])

        # Verify that matching files exist in other folders
        for direction in self.directions[1:]:
            folder = os.path.join(root, direction)
            for file_name in self.file_names:
                if not os.path.exists(os.path.join(folder, file_name)):
                    raise FileNotFoundError(f"File {file_name} not found in {folder}")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file_name = self.file_names[index]
        image_list = []

        for i, direction in enumerate(self.directions):
            img_path = os.path.join(self.root, direction, file_name)
            image = tiff.imread(img_path)

            if self.crop > 0:
                dx = np.random.randint(0, (image.shape[1] - self.crop) // 2)
                dx2 = image.shape[1] - self.crop - dx
                dy = np.random.randint(0, (image.shape[1] - self.crop) // 2)
                dy2 = image.shape[2] - self.crop - dy
                image = image[:, dx:-dx2, dy:-dy2]

            # permute (Z, X, Y) > (C, X, Y, Z)
            image = torch.from_numpy(image).permute(1, 2, 0).unsqueeze(0).float()

            image_list.append(image)

        # Stack images into a single tensor
        paired_images = image_list

        if self.labels:
            return {'img': paired_images, 'labels': self.labels[index]}
        else:
            return {'img': paired_images}


if __name__ == '__main__':
    #train_set = PairedDataTif(root=os.environ.get('DATASET') + args.dataset + '/train/',
    #                         path='a_b', labels=None, crop=256)

    train_set = PairedDataTif(root='/media/ExtHDD01/Dataset/paired_images/DPM4X/train/',
                              path='oripatch_ft0patch', opt=None, mode='train', index=None, filenames=True)

    x = train_set.__getitem__(0)['img']