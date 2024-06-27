import os
import random
from omegaconf import OmegaConf
import pandas as pd
from torch.utils.data import Dataset

from taming.data.base import ImagePaths
from taming.util import retrieve
from joblib import delayed, Parallel


def find_files(dir, format="csv"):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if str(fname).startswith("."):
                continue
            if str(fname).endswith(format) and os.path.isfile(os.path.join(root, fname)):
                path = os.path.join(root, fname)
                images.append(path)
    return images


class ImageNetTrain(Dataset):
    sub_dir: str = "train"
    def __init__(self, config=None, root_dir=None):
        self.config = config or OmegaConf.create()
        if not type(self.config)==dict:
            self.config = OmegaConf.to_container(self.config)
        root_dir = root_dir or os.environ.get("IMAGENET_ROOT", "data/imagenet")
        if not root_dir.endswith(self.sub_dir):
            root_dir = os.path.join(root_dir, self.sub_dir)
        self.root_dir = root_dir
        self._load()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def _prepare(self):
        raise NotImplementedError()

    def _load(self):
        img_file_list = [x for x in sorted(find_files(self.root_dir))]
        self.image_url_key = "cloud_url"
        valid_fieldnames = ["aspect", "width", "height", self.image_url_key]
        print("[info]valid_fieldnames: ", valid_fieldnames)
        results = Parallel(n_jobs=16)(delayed(self.read_csv_data)([file],
                                                                  valid_fieldnames,
                                                                  # exclude_content=None,
                                                                  url_key=self.image_url_key,
                                                                  limit_clip_duration=None,
                                                                  is_video_flag=False,
                                                                  ) for file in img_file_list)
        valid_fieldnames += ["is_video_flag"]
        self.fn2ind_img = {k: v for v, k in enumerate(valid_fieldnames)}

        self.data = []
        for data in results:
            self.data += data

        random.shuffle(self.data)

        print(f"[info]{self.sub_dir} total data: {len(self.data)}")

        self.img_urls = [item[self.fn2ind_img[self.image_url_key]] for item in self.data]

        self.data = ImagePaths(self.img_urls,
                               size=retrieve(self.config, "size", default=0),
                               random_crop=True)
        

    def read_csv_data(self, file_list, overall_fieldnames, exclude_content="cn-north-9", url_key="clip_cloud_url",
                      limit_clip_duration=None, is_video_flag=True):
        final_data = []
        for file in file_list:
            with open(file, "r") as f:
                line = f.readline()
            fieldnames = line.strip("\n").split("\t")
            data = pd.read_csv(file, usecols=fieldnames,
                                dtype='str', engine='python',
                                sep='\t', quotechar='"', encoding='utf-8').to_dict('records')
            remap_data = []
            for doc in data:
                if str(doc[url_key]) == "nan":
                    continue
                if exclude_content is not None and exclude_content in str(doc[url_key]):
                    continue
                if limit_clip_duration is not None:
                    clip_duration = float(doc["clip_duration"])
                    if clip_duration < limit_clip_duration[0]\
                                    or clip_duration > limit_clip_duration[1]:
                        continue
                cur_data = [doc.get(field_k, -100) for field_k in overall_fieldnames]
                cur_data += [is_video_flag]

                remap_data.append(cur_data)

        print(f"load data {file}, total: {len(remap_data)}")
        final_data += remap_data
        return final_data
    

class ImageNetValidation(ImageNetTrain):
    sub_dir: str = "val"
