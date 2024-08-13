# Splatfacto-W: A Nerfstudio Implementation of Gaussian Splatting for In the Wild Captures
An official implementation for [Splatfacto-W](https://kevinxu02.github.io/splatfactow/).

## Important
Please update your nerfstudio to the latest development version.
```
git clone https://github.com/nerfstudio-project/nerfstudio.git
cd nerfstudio
pip install --upgrade pip setuptools
pip install -e .
```

## TODO List
- [x] Support rendering videos for selected camera index
- [x] Support exporting PLY files for selected index
- [ ] (Probably) Merge splatfacto-w-light into Nerfstudio main branch

## Registering with Nerfstudio
Ensure that nerfstudio has been installed according to the [instructions](https://docs.nerf.studio/en/latest/quickstart/installation.html). Clone or fork this repository and run the commands:

```
conda activate nerfstudio
cd splatfacto-w/
pip install -e .
ns-install-cli
```

## Downloading Data
You can download the phototourism dataset from running.
```
ns-download-data phototourism --capture-name <capture_name>
```

## Running Splafacto-W
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
ns-train splatfacto-w --data [PATH] nerf-w-data-parser-config --data_name [trevi,sacre,brandenburg]
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

## Exporting PLY Files

We provide a temporary script `export_script.py` for exporting PLY files. 

The basic command structure is as follows:

```
python export_script.py --load_config [config_file] --output_dir [output_directory] --camera_idx [camera_index]
```

##  Rendering Videos For Selected Camera Index

You can see the current camera index in the CLI as you click each image in Viser.

To render videos for selected camera index, add the `--camera_idx [camera_index]` parameter to the end of your `ns-render` command.

