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
    ending = ["", "_05", "_10", "_15", "_20", "_25"]
    # model = unet_model(output_channels)
    for j in range(len(ending)):
        model_string = "all_checkpoints" + ending[j] + ".h5"
        saved_model_string = "epoch" + ending[j] + "_trained.h5"
        load = tf.keras.models.load_model(model_string, compile=False)
        load.save(saved_model_string)


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
    TRAIN_SIZE = 6000
    TEST_SIZE = 2800
    BATCH = 10
    EPOCHS = 25
    # SAVE_EPOCH = 5
    # train_steps = TRAIN_SIZE // BATCH
    output_channels = len(CATEGORIES) + 1
    model = unet_model(output_channels)
    base_path = "/home/tomtensor/Work/Projects/examples/tensorleap/examples/coco/coco/coco_data/ms-coco/annotations/"
    train_gen = COCOGenerator(join(base_path, "instances_train2014.json"), batch_size=BATCH,
                      categories=CATEGORIES, max_size=TRAIN_SIZE, append_vehicle_label=True)
    val_gen = COCOGenerator(join(base_path, "instances_val2014.json"), batch_size=BATCH,
                      categories=CATEGORIES, max_size=TEST_SIZE, append_vehicle_label=True)
    model = unet_model(output_channels)
    model.compile(optimizer='adam',
                  loss=losses.SparseCategoricalCrossentropy(),
                  metrics=[UpdatedMeanIoU(num_classes=output_channels)])
    # define the checkpoint
    filepath = "vehicle_supercategory_{epoch:02d}_{val_loss:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_updated_mean_io_u', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    model.fit(x=train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks_list)


if __name__ == "__main__":
    # train_model()
    train_model()