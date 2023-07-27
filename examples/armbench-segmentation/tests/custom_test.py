from armbench_segmentation.metrics import instance_seg_loss, od_loss

from tensorleap import subset_images, input_image, get_bbs, get_masks
import os
from keras.models import load_model
import numpy as np
import tensorflow as tf

def check():
    train, val = subset_images()
    image = input_image(0, train)
    gt_bbox = get_bbs(0, train)
    mask = get_masks(0, train)
    mask = np.transpose(mask, (2, 1, 0))

    #------------export model----------------------------
    path = "/Users/chenrothschild/repo/tensorleap/examples/armbench-segmentation/tests"
    os.chdir(path)
    model = os.path.join(path, 'exported-model.h5')

    model = load_model(model)

    image_batch = np.expand_dims(image, axis=0)
    #moel output------------------------
    y_pred = model.predict(image_batch)
    ls = od_loss(bb_gt=tf.convert_to_tensor(gt_bbox), detection_pred=y_pred[0])
    a=1

if __name__=='__main__':
    check()