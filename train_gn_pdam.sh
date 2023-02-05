#!/usr/bin/env bash

# consep2tnbc
# CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
#    --config-file "configs/uda_nuclei_seg/e2e_mask_rcnn_R_101_FPN_1x_gn_consep2tnbc.yaml" \
#    OUTPUT_DIR ./work_dir/consep2tnbc-models/pdam




# # pannuke2tnbc
# CUDA_VISIBLE_DEVICES=1 python tools/train_net.py \
#    --config-file "configs/uda_nuclei_seg/e2e_mask_rcnn_R_101_FPN_1x_gn_pannuke2tnbc.yaml" \
#    OUTPUT_DIR ./work_dir/pannuke2tnbc-models/pdam

# pannuke2cryonuseg
# CUDA_VISIBLE_DEVICES=1 python tools/train_net.py \
#    --config-file "configs/uda_nuclei_seg/e2e_mask_rcnn_R_101_FPN_1x_gn_pannuke2cryonuseg.yaml" \
#    OUTPUT_DIR ./work_dir/pannuke2cryonuseg-models/pdam


# tnbc2consep
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
   --config-file "configs/uda_nuclei_seg/e2e_mask_rcnn_R_101_FPN_1x_gn_tnbc2consep.yaml" \
   OUTPUT_DIR ./work_dir/tnbc2consep-models/pdam
