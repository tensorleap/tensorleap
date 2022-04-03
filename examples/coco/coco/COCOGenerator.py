from os.path import join
from pycocotools.coco import COCO
from skimage.io import imread
from skimage.color import gray2rgb
import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
from os.path import basename, join


class COCOGenerator(Sequence):

    def __init__(self, coco_path: str, categories: list, image_size: int = 128,
                 batch_size: int = 10, max_size: int = 1000):
        """
        A generator to train the coco model
        :param coco_path: a path to the coco_annotation file
        :param categories: a list of categories. Only images containing ALL these categories would be included
        :param image_size: which size of images required
        :param batch_size: what batch size to use for training
        :param max_size: what is the maximial size of the dataset
        """
        self.image_size = image_size
        self.path = coco_path
        self.cat = categories
        self.coco_file = COCO(coco_path)
        self.paths = self.load_set(self.coco_file)
        self.catids = self.coco_file.getCatIds(catNms=self.cat)
        self.paths = self.paths[:max_size]
        self.order = np.random.permutation(len(self.paths))
        self.batch_size = batch_size
        coco_name = basename(coco_path)
        if coco_name == "instances_train2014.json":
            self.subdir = "train2014"
        elif coco_name == "instances_val2014.json":
            self.subdir = "val2014"
        else:
            raise NotImplementedError

    def load_set(self, cocofile):
        # get all images containing given categories, select one at random
        catIds = cocofile.getCatIds(self.cat)
        imgIds = cocofile.getImgIds(catIds=catIds)
        imgs = cocofile.loadImgs(imgIds)
        return imgs

    def load_image(self, idx):
        x = self.paths[idx]
        filepath = "coco_data/ms-coco/{folder}/{file}".format(folder=self.subdir, file=x['file_name'])
        img = imread(filepath)
        if len(img.shape) == 2:
            # grascale -> expand to rgb
            img = gray2rgb(img)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        # rescale
        img = img / 255
        return img.astype(np.float)

    def ground_truth_mask(self, idx):
        x = self.paths[idx]
        annIds = self.coco_file.getAnnIds(imgIds=x['id'], catIds=self.catids, iscrowd=None)
        anns = self.coco_file.loadAnns(annIds)
        mask = np.zeros([x['height'], x['width']])
        for ann in anns:
            _mask = self.coco_file.annToMask(ann)
            mask[_mask > 0] = _mask[_mask > 0] * (self.catids.index(ann['category_id']) + 1)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        return mask.astype(np.float)

    def __len__(self):
        return len(self.paths)//self.batch_size

    def __getitem__(self, index):
        chosen_samples = self.order[index*self.batch_size:(index+1)*self.batch_size]
        masks = np.zeros((self.batch_size, self.image_size, self.image_size))
        images = np.zeros((self.batch_size, self.image_size, self.image_size, 3))
        for i, sample in enumerate(chosen_samples):
            masks[i, ...] = self.ground_truth_mask(sample)
            images[i, ...] = self.load_image(sample)
        return images, masks

    def on_epoch_end(self):
        self.order = np.random.permutation(len(self.paths))

# a = COCOGenerator("/home/tomtensor/Work/Projects/examples/tensorleap/examples/coco/coco/coco_data/ms-coco/annotations/instances_train2014.json",
#                   categories=['person', 'bicycle', 'car'])
#
# print(1)