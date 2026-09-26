# Datasets

Image files are not distributed with this repository. Obtain each dataset separately and arrange images in the class folders listed in the README.

| Dataset | Composition used | Role |
|---|---|---|
| miniImageNet | 100 classes, 600 images/class | Source training, validation and in-domain test |
| CUB-200-2011 | 200 classes | Fine-grained target |
| EuroSAT | 10 classes, 27,000 images | Satellite target |
| ISIC 2019 | 8 classes, 25,331 images | Dermoscopic target |

Target copies used in the study were obtained through Kaggle. The repository does not specify exact listing identifiers or distribute those copies. An exact CUB image total is not asserted.

## Source partition

`python -m scripts.prepare_miniimagenet` reads flat `.jpg` files whose first nine filename characters are the WordNet synset ID. It requires 100 classes, sorts their IDs, and copies the first 64 to train, next 16 to validation, and last 20 to test. With 600 images/class this gives 38,400 / 9,600 / 12,000 images.

Do not replace this with a shuffled or conventional miniImageNet class assignment when reproducing the reported protocol. The generic splitter in `experimental/` is for alternative-source experiments and follows a different, shuffled class split.

## Episodes and transforms

Each episode samples five classes and distinct support/query image indices within each class. There are 1 or 5 supports and 15 queries per class. Target episodes are drawn from class folders; there is no conventional target training split and no target parameter update in the main study.

Images are converted to RGB. Evaluation resizes to 256 then centre-crops to 224 x 224. Training uses random resized crops, horizontal flips and colour jitter (brightness, contrast and saturation 0.4). Both use ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`.

The loader reads files directly inside each class folder, with extensions `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif` or `.tiff`. It does not recursively flatten nested folders. Support order is class-major; multi-query training/evaluation shuffles queries before grouping.

## Optional targets

CropDisease and ChestX paths are available for extensions but are not part of the reported three-target study. `python -m scripts.check_dataset_layout` checks the primary datasets; add `--include-optional` to check these two as well.

The pretrained encoder may already have seen classes related to the source or target classes. Episodically held-out classes should not be described as guaranteed unseen during encoder pretraining.
