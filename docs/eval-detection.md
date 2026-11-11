Backbone `timm/wide_resnet50_2.tv_in1k`

| surface | train | test_good | test_defect | image_auroc | pixel_auroc | flagged_defect | flagged_good | top_region_hits | fit_s | s_per_image |
|---|---|---|---|---|---|---|---|---|---|---|
| metal | 8 | 6 | 12 | 1.0 | 0.993 | 12/12 | 0/6 | 12/12 | 4.6 | 0.04 |
| fabric | 8 | 6 | 12 | 1.0 | 0.993 | 12/12 | 0/6 | 12/12 | 2.0 | 0.05 |
| stone | 8 | 6 | 12 | 0.958 | 0.957 | 11/12 | 1/6 | 10/11 | 2.3 | 0.04 |
