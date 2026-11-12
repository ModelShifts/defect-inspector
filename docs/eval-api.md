Backbone `timm/wide_resnet50_2.tv_in1k`, describer `qwen3vl-2b-instruct-q4_k_m` (api)

| surface | train | test_good | test_defect | image_auroc | pixel_auroc | flagged_defect | flagged_good | top_region_hits | fit_s | s_per_image | label_accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| metal | 8 | 6 | 12 | 1.0 | 0.993 | 12/12 | 0/6 | 12/12 | 4.5 | 2.21 | 12/12 |
| fabric | 8 | 6 | 12 | 1.0 | 0.993 | 12/12 | 0/6 | 12/12 | 2.2 | 4.4 | 9/12 |
| stone | 8 | 6 | 12 | 0.958 | 0.957 | 11/12 | 1/6 | 10/11 | 267.9 | 10.61 | 9/10 |
