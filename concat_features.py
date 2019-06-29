from pathlib import Path
import numpy as np
import tqdm


data_root = 'features_output/IJBC/sphereface'
img_feats = []

img_list = [str(x) for x in Path(data_root).iterdir()]
img_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

for n in tqdm.tqdm(img_list):
    img_feats.append(np.load(n).flatten())
    
img_feats_arr = np.array(img_feats).astype(np.float32)
np.save('features_output/IJBC/sphereface-concat.npy', img_feats_arr)
