from tensorflow.keras import losses
from coco.COCOGenerator import COCOGenerator
from infer_model import unet_model
from os.path import join
from tensorflow.keras.metrics import MeanIoU
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
  def __init__(self,
               y_true=None,
               y_pred=None,
               num_classes=None,
               name=None,
               dtype=None):
    super(UpdatedMeanIoU, self).__init__(num_classes = num_classes,name=name, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_pred = tf.math.argmax(y_pred, axis=-1)
    return super().update_state(y_true, y_pred, sample_weight)


def save_without_compile():

    model_string = "vehicle_supercategory_04_0.26.h5"
    load = tf.keras.models.load_model(model_string, compile=False)
    load.save("lowest_val_loss_sc.h5")
    model_string = "vehicle_supercategory_08_0.28.h5"
    load = tf.keras.models.load_model(model_string, compile=False)
    load.save("highest_val_meaniou_sc.h5")


def infer_model():
    CATEGORIES = ['person', 'car']
    output_channels = len(CATEGORIES) + 1
    base_path = "/home/tomtensor/Work/Projects/examples/tensorleap/examples/coco/coco/coco_data/ms-coco/annotations/"
    # load = tf.keras.models.load_model("car_person_train_iou.h5", compile=False)
    # model = unet_model(output_channels)
    train_gen = COCOGenerator(join(base_path, "instances_train2014.json"),
                      categories=CATEGORIES, max_size=6000, append_vehicle_label=True)
    # val_gen = COCOGenerator(join(base_path, "instances_val2014.json"),categories=CATEGORIES, max_size=6000)
    a = train_gen[0]
    print(1)


def train_model():
    #TODO try to train a model without background prediction (i.e. only person + car)
    CATEGORIES = ['person', 'car']
    TRAIN_SIZE = 47500
    TEST_SIZE = 2800
    BATCH = 10
    EPOCHS = 25
    # SAVE_EPOCH = 5
    # train_steps = TRAIN_SIZE // BATCH
    output_channels = len(CATEGORIES) + 1
    model = unet_model(output_channels)
    base_path = "/home/tomtensor/Work/Projects/examples/tensorleap/examples/coco/coco/coco_data/ms-coco/annotations/"
    train_gen = COCOGenerator(join(base_path, "instances_train2014.json"), batch_size=BATCH,
                      categories=CATEGORIES, max_size=TRAIN_SIZE, append_vehicle_label=True,
                              force_contain_all_classes=False)
    val_gen = COCOGenerator(join(base_path, "instances_val2014.json"), batch_size=BATCH,
                      categories=CATEGORIES, max_size=TEST_SIZE, append_vehicle_label=True)
    model = unet_model(output_channels)
    model.compile(optimizer='adam',
                  loss=losses.SparseCategoricalCrossentropy(),
                  metrics=[UpdatedMeanIoU(num_classes=output_channels)])
    # define the checkpoint
    filepath = "vehicle_supercategory_all_val_{epoch:02d}_{val_loss:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    history = model.fit(x=train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks_list)
    print(1)


if __name__ == "__main__":
    # train_model()
    train_model()