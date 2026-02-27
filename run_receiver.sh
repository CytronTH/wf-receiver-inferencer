#!/bin/bash
DISPLAY=:0 python3 ai_receiver_gui.py \
  --hef-surface models/patchcore_bank_surface_masked_grayscale_0.5.hef \
  --hef-crop0 models/patchcore_bank_mask_0_grayscale_0.5.hef \
  --hef-crop2 models/patchcore_bank_mask_2_grayscale_0.5.hef \
  --hef-crop3 models/patchcore_bank_mask_3_grayscale_0.5.hef \
  --hef-crop4 models/patchcore_bank_mask_4_grayscale_0.5.hef \
  --hef-crop5 models/patchcore_bank_mask_5_grayscale_0.5.hef \
  --size-surface 320 \
  --size-crop 320 \
  --threshold 0.1838
