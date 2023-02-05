

import cv2
from maskrcnn_benchmark.utils.miscellaneous import mkdir
import tifffile as tiff
from inference.cell_predictor import CellDemo
from inference.metrics import mask2out, removeoverlap
from maskrcnn_benchmark.config import cfg
import os
import numpy as np
from maskrcnn_benchmark.modeling.detector import build_detection_model


def infer_fluo2tnbc(wts_root, out_pred_root, setting):

    config_file = f"../configs/uda_nuclei_seg/e2e_mask_rcnn_R_101_FPN_1x_gn_{setting}.yaml"

    cfg.merge_from_file(config_file)

    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
    model = build_detection_model(cfg)

    cell_demo = CellDemo(
        cfg,
        min_image_size=256,
        confidence_threshold=0.5,
        weight=wts_root,
        model=model
    )

    # with testing images
    test_root_name = '/data111/bianhao/code/zhangye/PDAM/datasets/consep/test/source_images'

    mkdir(out_pred_root)

    test_imgs = os.listdir(test_root_name)
    for img_name in test_imgs:

        if img_name.endswith(".png"):
            image = cv2.imread(os.path.join(test_root_name, img_name))

            # compute predictions
            predictions, mask_list = cell_demo.run_on_opencv_image(image)

            if mask_list.shape[-1]:
                masks_no_overlap, bi_map, num_mask = removeoverlap(mask_list)
                pred_ins = mask2out(masks_no_overlap, num_mask)

                cv2.imwrite(os.path.join(out_pred_root, img_name), predictions)
                cv2.imwrite(os.path.join(out_pred_root, 'bi_mask_' +
                            img_name), (bi_map * 255).astype(np.uint8))

                pred_ins_name = os.path.join(
                    out_pred_root, img_name.split('.')[0] + '.tif')
                tiff.imsave(pred_ins_name, pred_ins)

            else:
                cv2.imwrite(os.path.join(out_pred_root, img_name), predictions)
                cv2.imwrite(os.path.join(out_pred_root, 'bi_mask_' +
                            img_name), np.zeros(image.shape[:2], dtype=np.uint8))
                pred_ins_name = os.path.join(
                    out_pred_root, img_name.split('.')[0] + '.tif')
                tiff.imsave(pred_ins_name, np.zeros(
                    image.shape[:2], dtype=np.uint16))


if __name__ == "__main__":
    setting = 'tnbc2consep'
    wts_root = f'/data111/bianhao/code/zhangye/PDAM/work_dir/{setting}-models/pdam/model_epoch_010.pth'
    out_pred_root = setting

    infer_fluo2tnbc(wts_root, out_pred_root, setting)
