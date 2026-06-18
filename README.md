# Text Guidance and Distance Gating Improve Detection and Segmentation of Focal Cortical Dysplasia

**Accepted at MICCAI 2026**

German Mikhelson, Annalena Lange, Theodor Rüber, MELD Working Group, Thomas Schultz
b-it and Institute of Computer Science, University of Bonn · Contact: s17gmikh@uni-bonn.de

## Overview

Focal Cortical Dysplasia (FCD) is a leading cause of drug-resistant focal epilepsy, yet it is notoriously difficult to detect on routine MRI. This repository contains the code for our MICCAI 2026 paper, which proposes two complementary improvements over the MELD graph-based FCD detection baseline:

1. **Text-guided segmentation** — radiology-style text descriptions of lesion location (hemisphere and lobe) are encoded with RadBERT and injected into the GNN decoder via cross-attention (GuideDecoder). At test time the model can run without text, falling back to full-brain mode.
2. **Distance-based gating** — a lightweight MLP estimates the distance of each detected cluster to the nearest sulcal fundus. Clusters with large predicted distances (unlikely FCD) are suppressed, improving specificity without retraining the main model.

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{mikhelson2026fcd,
  title     = {Text Guidance and Distance Gating Improve Detection and Segmentation of Focal Cortical Dysplasia},
  author    = {Mikhelson, German and Lange, Annalena and R{\"u}ber, Theodor and {MELD Working Group} and Schultz, Thomas},
  booktitle = {Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year      = {2026}
}
```

## Acknowledgements

This work builds on the [MELD project](https://github.com/MELDProject/meld_graph). We thank the MELD Working Group for the multi-centre dataset and the open-source framework.
