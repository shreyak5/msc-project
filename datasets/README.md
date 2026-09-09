# Datasets used in this work
All of the datasets used in this project for training and evaluation are listed below and linked to their sources. These are all publicly available, although some of them require contacting the author for access.


## Sign Language Datasets
Save under `datasets/sign_datasets/`
- [How2Sign](https://how2sign.github.io/#download) 
- [CSL-Daily](https://ustc-slr.github.io/datasets/2021_csl_daily/)
- [PHOENIX2014T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)

## Face Datasets
Save under `datasets/face_datasets/`
- [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [FFHQ](https://github.com/NVlabs/ffhq-dataset)
- [BUPT-CBFace](https://buptzyb.github.io/CBFace/?reload=true#download)
- [MEAD](https://github.com/uniBruce/Mead)
- [AFEW-VA](https://ibug.doc.ic.ac.uk/resources/afew-va-database/)
- [FaMoS](https://tempeh.is.tue.mpg.de/)
- [Headspace](https://www-users.york.ac.uk/~np7/research/Headspace/)
- [DAD-3DHeads](https://github.com/PinataFarms/DAD-3DHeads)
- [CoMA](https://coma.is.tue.mpg.de/)
- [VOCASET](https://voca.is.tue.mpg.de/)

## Data preparation

### Step 1: Point script to the dataset storage directory
Point `dataset_processing/paths.py` to the directory where the face_datasets are stored.
```
FACE_DATASETS_ROOT = Path("/path/to/datasets/face_datasets")
```
`SIGN_DATASETS_ROOT` is derived automatically as the sibling directory `/path/to/datasets/sign_datasets/`.


### Step 2: Run the indexer scripts and setup the manifest files

Each downloaded dataset is structured differently. To unify the access pattern, each dataset has its own indexer under `dataset_processing/indexers/`, which scans the raw layout above and writes a manifest to `dataset_processing/manifests/<name>.jsonl`. These dataset-specific manifests store the paths to each datasample and are what the dataloaders actually read from.

Run the indexer for each dataset as follows:

```
python -m dataset_processing.indexers.index_how2sign
python -m dataset_processing.indexers.index_csl_daily
python -m dataset_processing.indexers.index_phoenix2014t

python -m dataset_processing.indexers.index_celeba
python -m dataset_processing.indexers.index_ffhq
python -m dataset_processing.indexers.index_bupt_cbface12
python -m dataset_processing.indexers.index_mead
python -m dataset_processing.indexers.index_afew_va
python -m dataset_processing.indexers.index_famos
python -m dataset_processing.indexers.index_headspace
python -m dataset_processing.indexers.index_dad_3dheads
python -m dataset_processing.indexers.index_coma
python -m dataset_processing.indexers.index_vocaset
```

### Step 3: Preprocessing for training

During training, caches are built to store reused per-sample data such as crops, detected landmarks, mica predictions and face masks. However, it is recommended to prewarm these caches beforehand, to avoid cache-building contention mid-run. There's one prewarm script per cache type:

```
python scripts/prewarm_crop_cache.py --dataloader_config dataset_processing/config/dataloader.yaml --dataset all --split train

python scripts/prewarm_landmark_cache.py --dataloader_config dataset_processing/config/dataloader.yaml --dataset all --split train

python scripts/prewarm_mica_cache.py --dataloader_config dataset_processing/config/dataloader.yaml --dataset all --split train

python scripts/prewarm_face_parsing_cache.py --dataloader_config dataset_processing/config/dataloader.yaml --dataset all --split train
```