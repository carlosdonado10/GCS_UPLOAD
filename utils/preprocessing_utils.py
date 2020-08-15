import cv2
from google.cloud import storage
import tensorflow as tf
import logging
import os
from glob import glob
from uuid import uuid4
from tqdm import tqdm
from utils.detect_faces import detect_faces
from config import rotations
import matplotlib.pyplot as plt
from pathlib import Path


#TODO:  resize images y cortar cara

def extract_images(pathIn, pathOut, fps, label):
    labels = {
        0: 'rested',
        1: 'intermediate',
        2: 'drowsy'
    }
    bucket_dir = labels.get(label)

    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    bucket = 'tesis_lv_lu'
    fps = vidcap.get(cv2.CAP_PROP_FPS)  # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    try:
        for msec in tqdm(range(0, frame_count, int(fps))):
            vidcap.set(cv2.CAP_PROP_POS_MSEC, msec)    # added this line
            success, image = vidcap.read()

            rotation = rotations.get(int(Path(pathIn).as_posix().split('/')[1]))
            if rotation is not None:
                image = cv2.rotate(image, rotation)
            _, image, _ = detect_faces(image, draw_box=False)
            if image is not None and image != []:
                image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
                convert_to_tfrecord(image, label, count)
                upload_blob(bucket_name=bucket, source_file_name=os.path.join(pathOut, f'{count}.tfrecord'), destination_blob_name=f'{bucket_dir}/{str(uuid4())}.tfrecord')

    except Exception as e:
        raise e
    finally:
        for file in glob('./processed_images/*'):
            os.remove(file)


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client.from_service_account_json('credentials.json')
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)



def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecord(image, label, name):
    """
    Converts a record to a TFRecord
    :param image:
    :param label:
    :return:
    """
    tfrecord_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'processed_images', f'{name}.tfrecord')
    writer = tf.io.TFRecordWriter(tfrecord_path)
    logging.info('Writing TFRecord file')
    image_raw = image.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': _int64_feature(int(label)),
        'image_raw': _bytes_feature(image_raw),
    }))
    writer.write(example.SerializeToString())
    logging.info('DONE :)')


def sample_image(pathIn: str):
    vidcap = cv2.VideoCapture(pathIn)
    vidcap.set(cv2.CAP_PROP_POS_MSEC, 0)
    success, image = vidcap.read()
    plt.imshow(image)
    plt.title(pathIn)
    plt.show()
    _ = input(f'Viewing {pathIn} Press enter to continue \n')

