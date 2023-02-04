# Instant Neural Graphics Primitives with a Multiresolution Hash Encoding
> Authors: Thomas Müller, Alex Evans, Christoph Schied, Alexander Keller
> 
> https://nvlabs.github.io/instant-ngp

## Motivation
- [Neural Radiance Fields](https://arxiv.org/abs/2003.08934) (NeRF) produce incredible representations of scenes
  - But it's *really \*\*\*\*\*\* slow*
  - **How can we speed it up?**


## Background
> Question: How can we help the network understand the spatial input better?**
> 
> Answer: By **"encoding the inputs of a machine learning model into a higher-dimensional space"**


### Positional encoding (from language models)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - a multiresolution sequence of $L \in N$ sine and cosine functions
  - $\text{enc}(\mathbf{x}) = \left(\sin(2^0\mathbf{x}), \cos(2^0\mathbf{x}), \sin(2^1\mathbf{x}), \cos(2^1\mathbf{x}), \ldots, \sin(2^{L-1}\mathbf{x}), \cos(2^{L-1}\mathbf{x})\right)$
  - NeRF's use of this positional encoding allows it to capture fine detail
- *Frequency encodings* (this paper's nomenclature) may be a more apt name
  - Mapping from spatial to frequency domain
  - [Demo](https://github.com/dylanhu7/positional-encoding-demo)

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
- For each grid, determine which voxel $\mathbf{x}$ falls into
  - For each vertex of the voxel, hash the integer coordinates to an integer index
    - $h: \mathbb{Z}^d \rightarrow \mathbb{Z}_T$
  - Look up the $F$-dimensional feature vectors corresponding to the indices in a hash table
  - Linearly interpolate between the feature vectors based on the corresponding vertex distance to the input $\mathbf{x}$ and concatenate
- Concatenate the $L$ interpolated feature vectors, along with auxiliary inputs $\xi \in \mathbb{R}^E$
  - Final result $y \in R^{LF + E}$