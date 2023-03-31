# Block-NeRF
> Matthew Tancik, Vincent Casser, Xinchen Yan, Sabeek Pradhan, Ben Mildenhall, Pratul P. Srinivasan, Jonathan T. Barron, Ren Ng, Henrik Kretzschmar<br>
> *UC Berkeley, Waymo, Google Research*<br>
> [arXiv](https://arxiv.org/abs/2202.05263)<br>
> [Project page](https://nvlabs.github.io/instant-ngp)<br>

## Abstract
> We present Block-NeRF, a variant of Neural Radiance Fields that can represent large-scale environments. Specifically, we demonstrate that when scaling NeRF to render city-scale scenes spanning multiple blocks, it is vital to decompose the scene into individually trained NeRFs. This decomposition decouples rendering time from scene size, enables rendering to scale to arbitrarily large environments, and allows per-block updates of the environment. We adopt several architectural changes to make NeRF robust to data captured over months under different environmental conditions. We add appearance embeddings, learned pose refinement, and controllable exposure to each individual NeRF, and introduce a procedure for aligning appearance between adjacent NeRFs so that they can be seamlessly combined. We build a grid of Block-NeRFs from 2.8 million images to create the largest neural scene representation to date, capable of rendering an entire neighborhood of San Francisco.

## Introduction
- Earlier NeRF methods address object-centric or small-scale scenes
  - Do not naively scale to city-scale scenes
- Reconstructing large-scale scenes has many applications
  - For autonomous driving, novel view synthesis is useful for alternative trajectories and environments (exposure, time of day, weather)
- Large-scale reconstruction has many challenges
  - Memory, compute, capacity, transient objects
  - Generally cannot capture entire scene at once, leading to variance
- Block-NeRF extends NeRF
  - Adds appearance embeddings, learned pose refinement, and controllable exposure
  - Scaling achieved by decomposing scene into blocks, each with its own Block-NeRF

## Method
- Split environment into a set of independently trained Block-NeRFs
  - At inference time, dynamically select blocks, optimize appearance codes to match lighting, and composite with interpolation
- Blocks placed at intersections, covering 75% of each street until the next intersection
  - Block size is therefore variable
- Uses Generative Latent Optimization to optimize per-image appearance embedding vectors as in NeRF-W
  - Allows for varying the lighting and weather of the scene at inference time
- Feed in positionally-encoded camera exposure as input to help compensate for visual differences
- Mask out transient objects (cars, pedestrians) by using a semantic segmentation model to identify them
- Use a small visibility network to predict how visible points are from the camera
- When rendering only consider blocks within a certain radius of the camera and whose mean visibility is above a threshold
- Compositing is done in image space, using inverse distance between the camera and block centers as weights
- Appearance codes between adjacent blocks are optimized to be close at inference time