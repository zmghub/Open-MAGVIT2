import bisect
import io
import numpy as np
import albumentations
import os
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as T
from obs import ObsClient

#  --- spider-cct数据
# host = 'obs.cn-north-4.myhuaweicloud.com/'
host = "obs.cn-north-9.myhuaweicloud.com"
# bucket = 'spider-cct'
bucket = "spider-cct-wlcb"
ak = "0OY8DYV9MNYU2HW2RPUW"
sk = "jLPaP4vvy554nl1FjkwrQVufgJVTdVTdohpJ8ozG"

class ConcatDatasetWithIndex(ConcatDataset):
    """Modified from original pytorch code to return dataset idx"""
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx

class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None):
        self.size = size
        self.random_crop = random_crop
        self.svd = os.getenv("SVD_FLAG", "false").lower() == "true"
        print(f"[INFO] SVD_FLAG: {self.svd}, for different image transform")

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self.client = None
        if 'myhuaweicloud.com/' in paths[0]:
            self.client = ObsClient(access_key_id=ak,
                   secret_access_key=sk,
                   server=host.strip("/"),
                   timeout=1200)
            print(f"[INFO] obs client init success with ak:{ak}, sk:{sk}, host: {host}.")
        self._length = len(paths)

        if self.size is not None and self.size > 0:
            self.rescaler = T.Resize(self.size)
            if not self.random_crop:
                self.cropper = T.CenterCrop((self.size, self.size))
            else:
                self.cropper = T.RandomCrop((self.size, self.size))
            self.preprocessor = T.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path, obs=False):
        if obs:
            if self.client is None:
                self.client = ObsClient(access_key_id=ak,
                       secret_access_key=sk,
                       server=host.strip("/"),
                       timeout=1200)
            resp = self.client.getObject(bucket, image_path)
            if resp.status < 300:
                ctx = resp.body.response.read()
                img_str = io.BytesIO(ctx)
            else:
                raise ValueError(f"load image error `http://` or `https://`, and {image_path} is not a valid path")
        else:
            img_str = image_path
        image = Image.open(img_str)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = self.preprocessor(image)
        image = np.array(image)
        if not self.svd:
            image = (image/127.5 - 1.0).astype(np.float32)
        else:
            image = (image/255.0).astype(np.float32)
            image -= np.array([[[0.485, 0.456, 0.406]]])
            image /= np.array([[[0.229, 0.224, 0.225]]])
        return image
    
    def __getitem__(self, i):
        example = dict()
        flag = False
        while not flag:
            try:
                image_path = self.labels["file_path_"][i]
                image_path = image_path.replace("%2F", "/")
                find_idx = image_path.rfind('myhuaweicloud.com/')
                image_path = image_path[find_idx + 18:]
                obs_flag = find_idx != -1
                example["image"] = self.preprocess_image(image_path, obs=obs_flag)
                flag = True
            except Exception as e:
                print(f"[WARN] err read obs 4: {image_path}")
                i = np.random.randint(0, self.__len__())
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example

class NumpyPaths(ImagePaths):
    def preprocess_image(self, image_path):
        image = np.load(image_path).squeeze(0)  # 3 x 1024 x 1024
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(image, mode="RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
