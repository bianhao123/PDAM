#!/usr/bin/env python3

import argparse
import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
import pycococreatortools
import numpy as np
from PIL import Image
from skimage import io
from os.path import join as opj
import glob
import natsort
import os
import tqdm


parser = argparse.ArgumentParser(description="coco dataset")
parser.add_argument(
    "--dataset",
    default="tnbc",
    help="dataset name",
    type=str,
)
parser.add_argument(
    "--stage",
    default="train",
    help="stage name",
    type=str,
)
args = parser.parse_args()
dataset = args.dataset
stage = args.stage

ROOT_DIR = f'/data111/bianhao/code/zhangye/PDAM/ZY_CVPR/{dataset}/'
OUT_DIR = opj(ROOT_DIR, 'coco', stage)
os.makedirs(OUT_DIR, exist_ok=True)
SEMGT_DIR = opj(OUT_DIR, 'semgt')
SOURCE_DIR = opj(OUT_DIR, 'source_images')
TARGET_DIR = opj(OUT_DIR, 'target_images')
os.makedirs(SEMGT_DIR, exist_ok=True)
os.makedirs(SOURCE_DIR, exist_ok=True)
os.makedirs(TARGET_DIR, exist_ok=True)
IMAGE_DIR = opj(ROOT_DIR, stage)
# IMAGE_DIR = os.path.join(ROOT_DIR, "shapes_train2018")
# ANNOTATION_DIR = os.path.join(ROOT_DIR, "annotations")

INFO = {
    "description": f"{dataset.upper()} Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2022,
    "contributor": "haobian",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'nuclei',
        'supercategory': 'cell',
    },

]


def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]

    return files


def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(
        os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(
        file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files


def main():

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1

    # filter for jpeg images
    for image_filename in tqdm.tqdm(natsort.natsorted(glob.glob(f'{IMAGE_DIR}/*[0-9].png'))):
        # image_files = filter_for_jpeg(root, files)

        # go through each image
        # for image_filename in image_files:
        image = Image.open(image_filename)
        image_info = pycococreatortools.create_image_info(
            image_id, os.path.basename(image_filename), image.size)
        coco_output["images"].append(image_info)

        # associated png annotations
        # image_filename = 'train/3.png'
        image_dirname = os.path.dirname(image_filename)
        image_basename = os.path.basename(image_filename)
        image_file_id = image_basename.split('.')[0]
        ins = np.load(opj(image_dirname, image_file_id + '_instance.npy'))
        seg = io.imread(opj(image_dirname, image_file_id + '_semantic.png'))

        image.save(opj(SOURCE_DIR, image_basename))
        image.save(opj(TARGET_DIR, image_basename))
        io.imsave(opj(SEMGT_DIR, image_basename), np.uint8(seg * 255))

        # filter for associated png annotations
        ins_list = range(1, int(ins.max()) + 1)
        for inst_id in ins_list:
            # annotation_files = filter_for_annotations(
            # root, files, image_filename)

            # print(annotation_filename)
            class_id = 1  # 只有一类 cell

            category_info = {'id': class_id,
                             'is_crowd': 'crowd' in image_filename}

            inst_map = np.array(ins == inst_id, np.uint8)

            binary_mask = inst_map
            # inst_box = get_bounding_box(inst_map)

            annotation_info = pycococreatortools.create_annotation_info(
                segmentation_id, image_id, category_info, binary_mask,
                image.size, tolerance=2)

            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)

            segmentation_id = segmentation_id + 1

        image_id = image_id + 1

    with open(f'{OUT_DIR}/{dataset}_instance_{stage}.json'.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)


if __name__ == "__main__":
    main()
