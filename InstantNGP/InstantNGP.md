# Instant Neural Graphics Primitives with a Multiresolution Hash Encoding
> Thomas Müller, Alex Evans, Christoph Schied, Alexander Keller<br>
> NVIDIA<br>
> SIGGRAPH 2022
> 
> [Project page](https://nvlabs.github.io/instant-ngp)<br>
> [My presentation](https://www.icloud.com/keynote/051disEtf0MTAf5oNq3nZguvg#InstantNGP) (best viewed in Keynote, rather than in the browser)

## Motivation
- [Neural Radiance Fields](https://arxiv.org/abs/2003.08934) (NeRF) produce incredible representations of scenes
  - But it's *really \*\*\*\*\*\* slow*
  - **How can we speed it up?**


## Background
> Question: How can we help the network understand the spatial input better?
> 
> Answer: By **"encoding the inputs of a machine learning model into a higher-dimensional space"**


### Positional encoding (from language models and earlier)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - a multiresolution sequence of $L \in N$ sine and cosine functions
  - $\text{enc}(\mathbf{x}) = \left(\sin(2^0\mathbf{x}), \cos(2^0\mathbf{x}), \sin(2^1\mathbf{x}), \cos(2^1\mathbf{x}), \ldots, \sin(2^{L-1}\mathbf{x}), \cos(2^{L-1}\mathbf{x})\right)$
  - NeRF's use of this positional encoding allows it to capture fine detail
- *Frequency encodings* (this paper's nomenclature) may be a more apt name
  - Mapping from spatial to frequency domain
  - [Demo](https://github.com/dylanhu7/positional-encoding-demo)
- [SIREN](https://www.vincentsitzmann.com/siren/) uses $\sin$ activations instead of ReLU
  - First SIREN layer mathematically equivalent to a frequency encoding

### Parametric encodings
> "The idea is to arrange additional trainable parameters (beyond weights and biases) in an auxiliary data structure, such as a grid or a tree, and to look-up and (optionally) interpolate these parameters
depending on the input vector $\mathbf{x} \in \mathbb{R}^d$."

- **More memory, less computation**
  - For each gradient, only a small subset of the parameters need to be updated
  - Enables a smaller MLP

#### Sparse parametric encodings
- For most scenes, we only care about parts where there is *stuff*
  - We don't want to waste memory on empty space
- We could cull away parts of our grid that are empty
  - Requires knowledge of the scene beforehand
  - Increased control flow
  - Harder to implement


## Method
### Multiresolution hash encoding
> "Given a fully connected neural network $m(\mathbf{y}; \Phi)$, we are interested in an *encoding* of its inputs $\mathbf{y} = \text{enc}(\mathbf{x};\theta)$ that improves the approximation quality and training speed."

Given an input $\mathbf{x} \in \mathbb{R}^d$:
- Store $L$ "voxel" grids of varying resolution
  - Each "voxel" vertex is $d$-dimensional
- For each grid, determine which voxel $\mathbf{x}$ falls into using $\lfloor\mathbf{x}\rfloor$ and $\lceil\mathbf{x}\rceil$.
  - For each vertex of the voxel, hash the coordinates to an integer hash table index
    - $h: \mathbb{Z}^d \rightarrow \mathbb{Z}_T$
  - Look up the $F$-dimensional feature vectors corresponding to the indices in the hash table
  - Linearly interpolate between the feature vectors based on the corresponding vertex distance to the input $\mathbf{x}$ and reduce to a single $F$-dimensional feature vector
    - Not sure how they are reduced
- Concatenate the $L$ interpolated feature vectors, along with auxiliary inputs $\xi \in \mathbb{R}^E$
  - Final result $y \in R^{LF + E}$
- Feed $y$ into a fully connected neural network $m(\mathbf{y}; \Phi)$
- Gradients are backpropagated through the MLP, concatenation, interpolation, and then accumulated in the feature vectors

### Why does it work?
- Nearby inputs will hash to the same feature vectors
  - Linear interpolation within voxels handles this
- Separate regions may hash to the same features
  - Statistically unlikely to occur *simultaneously* across all layers
  - Induces *adaptivity*
- **Adaptivity**
  - Separate regions are likely not equal in importance
  - Colliding gradients average, larger (more important) gradients dominate
    - *Automatically* prioritizes regions with finer detail

## Other advantages
- Online adaptivity
  - Automatically handles changes in the training data distribution
  - Seen in neural radiance caching application
- Continuity
  - Linear interpolation induces continuity within voxels
  - Shared voxel vertices avoids discontinuities between voxels

# Results
> Visit [project page](https://nvlabs.github.io/instant-ngp) for result videos
## Gigapixel image approximation
ACORN (2021): 36.9 hours

InstantNGP: 2.5 minutes

## Signed distance fields
- Comparable performance to NGLOD with >0.999 IoU

- Significantly outperforms frequency encoding

- Authors note presence of surface roughness and artifacts as a result of unhandled hash collisions, but I found this imperceptible in the SDF application

## Neural radiance caching
- Leverages online adaptivity
- Didn't really look into this

## Neural radiance fields
- Coherent image after just 1 second on a single GPU
- Achieves PSNR comparable to NeRF after just 5 seconds
- Performs particularly well on scenes with a lot of high-frequency detail

# Discussion
- Accelerates NeRF training by ~1000x
  - How much can be attributed to the hash encoding vs just great engineering?
    - Authors claim "20-60x improvement" from hash encoding alone by using frequency encoding with their optimized MLPs and volume renderer
- Has applications to many tasks, particularly "low-dimensional tasks that require accurate, high-frequency fits"