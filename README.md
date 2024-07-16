# Splatfacto-W: A Nerfstudio Implementation of Gaussian Splatting for In the Wild Captures
An official implementation for [Splatfacto-W](https://kevinxu02.github.io/splatfactow/).

## Important
The code part for appearance changing hasn't been merged into nerfstudio main branch yet. You might need to manually clone this PR nerfstudio-project/nerfstudio#3293.

## Registering with Nerfstudio
Ensure that nerfstudio has been installed according to the [instructions](https://docs.nerf.studio/en/latest/quickstart/installation.html). Clone or fork this repository and run the commands:

```
conda activate nerfstudio
cd splatfacto-w/
pip install -e .
ns-install-cli
```

## Downloading data
You can download the phototourism dataset from running.
```
ns-download-data phototourism --capture-name <capture_name>
```

## Running Splafacto-w
To train with it, download the train/test tsv file from the bottom of [nerf-w](https://nerf-w.github.io/) and put it under the data folder (or copy them from `./splatfacto-w/dataset_split`). For instance, for Brandenburg Gate the path would be `your-data-folder/brandenburg_gate/brandenburg.tsv`. You should have the following structure in your data folder:
```
|---brandenburg_gate
|   |---dense
|   |   |---images
|   |   |---sparse
|   |   |---stereo
|   |---brandenburg.tsv
```
Then, run the command:
```
ns-train splatfacto-w --data [PATH]
```

If you want to train datasets without nerf-w's train/test split or your own datasets, we provided a light-weight version of the method for general cases. To train with it, you can run the following command
```bash
ns-train splatfacto-w-light [OPTIONS] --data [PATH] [dataparser]
```
You can add these options to enhance the training process:

1. `--pipeline.model.enable_bg_model True`
   - Enables background modeling
   - Useful for scenes with distinct foreground and background

2. `--pipeline.model.enable_alpha_loss True`
   - Enables alpha loss for punishing gaussians from occupying sky areas.
   - Should be used with background modeling

3. `--pipeline.model.enable_robust_mask True`
   - Improves handling of transient objects (things that appear/disappear between views)

For phototourism, the `dataparser` should be `colmap` and you need to change the colmap path through the CLI because phototourism dataparser does not load 3D points.
