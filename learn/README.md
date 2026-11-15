# Learning track: *Through the Thicket to Inference*

Nine notebooks that go with the article. Each runs on a laptop CPU or on free Google Colab (click the badge at the
top of a notebook). They download one shard of the training data (29 MB) and, from notebook 07 on, the trained model
(157 MB plus a 470 MB text encoder).

| notebook | case file | what you do | CPU time |
|---|---|---|---|
| `01_meet_the_sprites` | 1 · The Evidence Room | open the dataset, look at pixels, RGB and alpha, count colours | ~1 min |
| `02_fingerprints` | 2 · Fingerprints | measure pieces, holes, symmetry, compactness; compare with random shapes | ~1 min |
| `03_bits` | 3 · The Profile Theory | bits and entropy; describe shapes by rows; a Markov chain that draws (badly) | ~1 min |
| `04_tiny_transformer` | 4 · The First Witness | write a transformer from scratch, train it on silhouettes, guide it with words | ~7 min |
| `05_palette` | 5 · The Colour Alibi | OKLab, k-means palettes, mean vs worst error | ~1 min |
| `06_mini_canvas` | 6 · The Whole Canvas | train a small version of the real model (shape + colour) and sample from it | ~9 min |
| `07_guidance_lab` | 7 · Lying Clues | the real model: temperature, guidance, split guidance; catch guidance erasing poses | ~3 min |
| `08_interrogate` | 9 · The Interrogation | bits on unseen sprites, named-colour test, ROC AUC, attribute pairs | ~3 min |
| `09_inference_from_scratch` | Epilogue | write your own sampler, match the library exactly, measure the KV-cache speed-up | ~2 min |

Times were measured on an 8-thread desktop CPU; a Colab GPU is faster for 04 and 06.

To run locally:

```bash
pip install "pixelsprite[learn] @ git+https://github.com/umanets/pixelsprite" jupyter
jupyter lab learn/
```

The notebooks are built from the plain-Python sources in `learn/src/` (cells marked with `# %%`).
