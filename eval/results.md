# Generation FID results any to RGB:

- 5000 images from test directory
- num_steps = [8]
- temps = [0.01]
- top_ps = [0.2]
- top_ks = [0.0]

| Modality | FID score |
|---|---|
| depth | 35.15 |
| scene_desc | 113.33 |
| canny | 51.35 |
| normal | 34.26 |

# Comparaison of FID score by chaining with canny to RGB (modality->canny-rgb):

- 5000 images from test directory
- num_steps = [8, 8]
- temps = [0.01, 0.01]
- top_ps = [0.2, 0.2]
- top_ks = [0.0, 0.0]

| Modality | FID score |
|---|---|
| depth | 35.15 |
| depth->canny | 39.55 |

| Modality | FID score |
|---|---|
| scene_desc | 113.33 |
| scene_desc->canny | 111.37 |

| Modality | FID score |
|---|---|
| normal | 34.26 |
| normal->canny | 39.65 |