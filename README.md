# Splatfacto-W: A Nerfstudio Implementation of Gaussian Splatting for In the Wild Captures
An official implementation for [Splatfacto-W](https://kevinxu02.github.io/gsw.github.io/)

## Registering with Nerfstudio
Ensure that nerfstudio has been installed according to the [instructions](https://docs.nerf.studio/en/latest/quickstart/installation.html). Clone or fork this repository and run the commands:

```
conda activate nerfstudio
cd splatfacto-w/
pip install -e .
ns-install-cli
```

## Running the new method
To train with it, download the train/test tsv file from the bottom of [nerf-w](https://nerf-w.github.io/) and put it under the data folder (or copy them from `.\splatfacto-w\dataset_split`). For instance, for Brandenburg Gate the path would be `splatfacto-w\data\brandenburg_gate\brandenburg.tsv`.
Then, run the command:
```
ns-train splatfacto-w --data [PATH]
```

If you want to train datasets without nerf-w's train/test split:
```
ns-train splatfacto-w --data [PATH] phototourism-data --load-3d-points=True
```
