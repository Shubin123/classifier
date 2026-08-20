# classifier — Apple Quality Classifier

Grades apples as good or bad from a photograph, entirely in the browser. A
Vision Transformer fine-tuned for binary classification runs client-side through
TensorFlow.js, with an optional YOLO11m segmentation model for locating the
fruit in the frame first.

Part of the [webgl](https://github.com/Shubin123/webgl) demo collection.

## What it does

Upload one or more images and press **Predict**. Each image is:

1. Read into a tensor with `tf.browser.fromPixels`.
2. Resized to **224×224**, scaled to `[0, 1]`, and given a batch dimension.
3. Run through the fine-tuned classifier, whose logits are passed through
   `tf.softmax` to produce confidence scores, with `tf.argMax` selecting the
   predicted class.

Ticking the secondary-model checkbox additionally loads **YOLO11m-seg**, which
runs at **640×640** and produces boxes, class scores and segmentation feature
maps. Detections are filtered with `tf.image.nonMaxSuppressionAsync`, and
objectness is combined with class scores before thresholding. Tensors are
explicitly released with `tf.dispose` after each pass.

## Models

| Directory | Size | Shards | Used | What it is |
| --- | --- | --- | --- | --- |
| `binary_classification_model/` | 328 MB | 82 | yes | The apple classifier — a fine-tuned ViT, converted to a TF.js graph model |
| `yolo11m-seg_web_model/` | 86 MB | 22 | optional | YOLO11m segmentation, loaded when the checkbox is ticked |
| `object_detection_model/` | 22 MB | 6 | **no** | Not referenced anywhere in `index.html` |

Total repository size is roughly **437 MB**. No individual file exceeds 4 MB —
TensorFlow.js splits weights into 4 MB shards — so everything is committed as
ordinary git objects with no Git LFS involved. Cloning transfers the full
437 MB.

### Weight caching

Because the classifier alone is 328 MB across 82 shards, the page implements its
own **IndexedDB** cache: shards are stored as blobs after first download and
served from the database on subsequent visits, with a counter driving the
loading display.

## Training

Both scripts that produced the classifier are included. They expect a dataset
directory `./resized_apples/` laid out for `ImageFolder` / `flow_from_directory`
— one subdirectory per class (bad and good apples), images already resized to
224×224. **That dataset is not in this repository**; supply your own.

### `finetune.py` — the approach that shipped

Fine-tunes [`google/vit-base-patch16-224-in21k`](https://huggingface.co/google/vit-base-patch16-224-in21k)
with PyTorch and Hugging Face Transformers for 2-class output. It loads local
pre-trained weights from `pytorch_model.bin`, deleting `classifier.weight` and
`classifier.bias` from the state dict first — those have the wrong shape once
`num_labels` changes — and loading the rest with `strict=False`. Images are
normalised with the standard ImageNet statistics and optimised with Adam.

### `trainfromscratch.py` — the baseline

Trains a small Keras CNN from scratch on the same data for comparison, with
aggressive augmentation (50° rotation, 0.4 shift and zoom, 0.3 shear, horizontal
flips), a 20% validation split, and early stopping.

### Converting to TensorFlow.js

The models here are TF.js **graph** models. To regenerate them after training:

```sh
pip install tensorflowjs
tensorflowjs_converter \
  --input_format=tf_saved_model \
  --output_format=tfjs_graph_model \
  ./saved_model ./binary_classification_model
```

The converter writes `model.json` plus the `group1-shard{i}of{n}.bin` files. If
the shard count changes, update `totalFiles` in `loadModel` in `index.html`.

## Running it

TensorFlow.js 4.22.0 loads from jsDelivr; there is nothing to build. Serve over
HTTP so the model manifests and weight shards resolve:

```sh
python -m http.server 8000
# then open http://localhost:8000/
```

Expect a slow first load — 328 MB of weights before the first prediction, or
414 MB with the YOLO model enabled. Subsequent loads are served from IndexedDB.

## Known limitations

- **The shard cache loop requests filenames that do not exist.** `loadModel`
  builds paths with `String(i).padStart(2, '0')`, producing
  `group1-shard01of82.bin`, but TensorFlow.js emits **unpadded** names —
  `group1-shard1of82.bin`. Shards 1 through 9 therefore 404 on every run and are
  never cached. The demo still works, because `tf.loadGraphModel` has already
  fetched the weights itself via `model.json` before this loop runs; the loop is
  a supplementary caching and progress layer. Removing the `padStart` fixes it.
- **`object_detection_model/` is dead weight** — 22 MB that no code path loads.
- **The training dataset is absent.** Both scripts hard-code `./resized_apples/`,
  and `finetune.py` also expects a local `pytorch_model.bin`. Neither is in the
  repository, so the scripts are a record of method rather than something you can
  run unmodified.
- **437 MB is a lot for a git repository.** GitHub accepts it — no single file is
  near the 100 MB cap — but it is well above the size where clones stay
  comfortable. Quantising the classifier, or converting it to a TF.js *layers*
  model with 16-bit weights, would cut this substantially.
- A large block of alternative segmentation post-processing is left commented out
  at the bottom of `index.html`.
- `ViTFeatureExtractor` is instantiated in `finetune.py` but the transforms are
  then done manually with `torchvision`, so the extractor is unused. It is also
  deprecated in current Transformers releases in favour of `ViTImageProcessor`.
