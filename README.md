# Splatfacto-W: A Nerfstudio Implementation of Gaussian Splatting for In the Wild Captures
A official implementation for [Splatfacto-W](https://kevinxu02.github.io/gsw.github.io/)

## Registering with Nerfstudio
Ensure that nerfstudio has been installed according to the [instructions](https://docs.nerf.studio/en/latest/quickstart/installation.html). Clone or fork this repository and run the commands:

```
conda activate nerfstudio
cd splatfacto-w/
pip install -e .
ns-install-cli
```

## Running the new method
This repository creates a new Nerfstudio method named "method-template". To train with it, download the train/test tsv file from the bottom of [nerf-w](https://nerf-w.github.io/) and put it under the data folder. For instance, for brandenburg Gate the path would be `splatfacto-w\data\brandenburg_gate\brandenburg.tsv`.
Then, run the command:
```
ns-train splatfacto-w --data [PATH]
```
