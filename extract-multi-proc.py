from functools import partial, wraps
from timeit import default_timer
import logging
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import cv2

MODEL_NAME = 'LMSoftmax'
PROTOTXT = f'models/{MODEL_NAME}_model/lm_model_deploy_test2.prototxt'
CAFFE_MODEL = f'models/{MODEL_NAME}_model/lm-result_iter_30000.caffemodel'

# MODEL_NAME = 'sphereface'
# PROTOTXT = f'models/{MODEL_NAME}_model/sphereface_deploy.prototxt'
# CAFFE_MODEL = f'models/{MODEL_NAME}_model/sphereface_model.caffemodel'

DATA_ROOT = '../insightface-original/IJB_release/IJBC/affine'
# DATA_ROOT = '../insightface-original/megaface_testpack_v1.0/data/megaface_images'
# DATA_ROOT = '../insightface-original/megaface_testpack_v1.0/data/facescrub_images/'
# FEATURE_OUT = 'features_output/FaceScrub/amsoftmax'
# FEATURE_OUT = 'features_output/MegaFace/{MODEL_NAME}'
FEATURE_OUT = f'features_output/IJBC/{MODEL_NAME}-test'
# LOG_PATH = 'log/Normface-megaface.log'
LOG_PATH = f'log/{MODEL_NAME}-ijbc-test.log'
N_THREAD = 15


net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFE_MODEL)

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
def extract_and_save_features(feature_out_, img_path_, use_norm=False):
    """
    Extract and deep features while reserving the original filepath hierarchy of datasets
    """
    print(F"{MODEL_NAME} {img_path_}")

    # Inference network and collect feature extracted
    img = cv2.imread(str(img_path_))
    blob = cv2.dnn.blobFromImage(img, 1, (112, 96), (104, 117, 123))
    net.setInput(blob)
    feat = net.forward()

    if use_norm:
        feat = feat / np.sqrt(np.sum(feat ** 2, -1, keepdims=True))

    # Setup directory for saving feature vector
    saved_path_parent = Path(f'{feature_out_}/{img_path_.relative_to(DATA_ROOT).parent}')
    saved_path_parent.mkdir(parents=True, exist_ok=True)

    # Save feature vector to specific location
    saved_path = saved_path_parent / f"{img_path_.stem}.npy"
    np.save(saved_path, feat)


if __name__ == "__main__":

    data_root = DATA_ROOT
    feature_out = FEATURE_OUT
    print(f"MODEL: {CAFFE_MODEL} \nDATA: {DATA_ROOT} \nLOG:{LOG_PATH}")

    print("Loading images ... ")
    img_list = [n for n in [i for i in (Path(data_root).rglob(e) for e in ('*.png', '*.jpg')) for i in i]]

    func = partial(extract_and_save_features, feature_out)

    pool = Pool(N_THREAD)
    pool.map(func, img_list)
    pool.close()
    pool.join()
