from functools import partial, wraps
from timeit import default_timer
import logging
from pathlib import Path
import tqdm
import numpy as np
import cv2


MODEL_NAME = 'center-loss_model'
PROTOTXT = 'models/center-loss_model/face_deploy.prototxt'
CAFFE_MODEL = 'models/center-loss_model/face_model.caffemodel'
# DATA_ROOT = '../insightface-original/IJB_release/IJBC/affine'
DATA_ROOT = '../insightface-original/megaface_testpack_v1.0/data/megaface_images'
# DATA_ROOT = '../insightface-original/megaface_testpack_v1.0/data/facescrub_images/'
# FEATURE_OUT = 'features_output/FaceScrub/amsoftmax'
FEATURE_OUT = 'features_output/MegaFace/amsoftmax'
# FEATURE_OUT = 'features_output/IJBC/amsoftmax'
LOG_PATH = 'log/amsoftmax-megaface.log'
N_THREAD = 15


def _time_and_log(orig_func, logpath):
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        logging.basicConfig(filename=logpath, level=logging.INFO)

        start = default_timer()
        result = orig_func(*args, **kwargs)
        end = default_timer()

        logging.info(f"Time elapsed: {end - start} seconds")
        return result

    return wrapper


time_and_log = partial(_time_and_log, logpath=LOG_PATH)


@time_and_log
def extract_and_save_features(net_, data_root_, feature_out_, use_norm=False):
    """
    Extract and deep features while reserving the original filepath hierarchy of datasets
    """
    print("[INFO] Loading image data ... ")
    for n in tqdm.tqdm([i for i in (Path(data_root_).rglob(e) for e in ('*.png', '*.jpg')) for i in i]):

        # Inference network and collect feature extracted
        img = cv2.imread(str(n))
        blob = cv2.dnn.blobFromImage(img, 1, (112, 96), (104, 117, 123))
        net_.setInput(blob)
        feat = net_.forward()

        if use_norm:
            feat = feat / np.sqrt(np.sum(feat ** 2, -1, keepdims=True))

        # Setup directory for saving feature vector
        saved_path_parent = Path(f'{feature_out_}/{n.relative_to(data_root_).parent}')
        saved_path_parent.mkdir(parents=True, exist_ok=True)

        # Save feature vector to specific location
        saved_path = saved_path_parent / f"{n.stem}.npy"
        np.save(saved_path, feat)


if __name__ == "__main__":

    net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFE_MODEL)
    data_root = DATA_ROOT
    feature_out = FEATURE_OUT

    print(f"MODEL: {CAFFE_MODEL} \nDATA: {DATA_ROOT} \nLOG:{LOG_PATH}")

    extract_and_save_features(net, data_root, feature_out, use_norm=USE_NORM)
