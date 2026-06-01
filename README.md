# nano4M extension: Canny Edges

<h2 align="center">
  <a href="https://theomaillard.github.io/nano4M-extension/">Website</a>
</h2>

## Environment Setup

The simplest way to get started is via the provided setup script, which creates a `nanofm` conda environment, registers a Jupyter kernel, and installs all dependencies:

```bash
bash setup_env.sh
```

> All notebooks should be run on **a single GPU**.

## Training on SCITAS

Training is submitted to the SCITAS cluster via the provided submission script:

```bash
python submit_job.sh cfgs\nano4M\multiclevr_d6-6w512.yaml <wandb_api_key> 2
```
- The config file (`cfgs\nano4M\multiclevr_d6-6w512.yaml`) controls the model architecture, dataset, training duration, etc...
- The tokenized Canny dataset is located in `/scratch/tmaillar/canny/`
- The checkpoint for the trained model is located at `/scratch/tmaillar/outputs/nano4M/multiclevr_d6-6w512/checkpoint-final.safetensors`

---

## File Hierarchy

```
├───dataset/ # notebooks for dataset transformation & visualization and script to transform RGB dataset to Canny edges
│       
├───docs/ # website
│           
├───eval/ # notebooks for inference and evaluation
│       
└───nano4M
    │   pyproject.toml
    │   run_training.py
    │   setup_env.sh
    │   submit_job.sh
    │   
    ├───cfgs
    │   └───nano4M
    │           multiclevr_d6-6w512.yaml
    │           
    └───nanofm
        ├───data
        │   │   utils.py
        │   │   __init__.py
        │   │   
        │   └───multimodal
        │           masking.py
        │           simple_multimodal_dataset.py
        │           utils.py
        │           __init__.py
        │           
        ├───modeling
        │       transformer_layers.py
        │       __init__.py
        │       
        ├───models
        │       fourm.py
        │       __init__.py
        │       
        └───utils/
```