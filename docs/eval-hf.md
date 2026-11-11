Backbone `timm/wide_resnet50_2.tv_in1k`, describer `Qwen/Qwen3-VL-2B-Instruct` (hf)

| surface | train | test_good | test_defect | image_auroc | pixel_auroc | flagged_defect | flagged_good | top_region_hits | fit_s | s_per_image | label_accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| metal | 8 | 6 | 12 | 1.0 | 0.993 | 12/12 | 0/6 | 12/12 | 3.1 | 2.11 | 12/12 |
| fabric | 8 | 6 | 12 | 1.0 | 0.993 | 12/12 | 0/6 | 12/12 | 1.9 | 1.85 | 8/12 |
| stone | 8 | 6 | 12 | 0.958 | 0.957 | 11/12 | 1/6 | 10/11 | 2.1 | 2.35 | 9/10 |
