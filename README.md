# MultiScale Deformable Attention V2 for Ascend Device

## __Description__
An improved version of MultiScale Deformable Attention, designed for enhanced performance on Ascend platforms, featuring multi-core tiling, on-chip buffer optimization, fine-grained event-based synchronization for efficient parallel computation, etc. 

## __Device__
- version tested: ___Ascend 910B3___ 
- verified in the following files:
  - `msda_v2/CMakePresets.json` 
  - `msda_v2/op_host/*.cpp`

## __Installation__

### Clone the source
```bash
git clone https://github.com/UrWright/msda_v2.git
```

### Build the source

```bash
cd msda_v2 

# Build to ./build_out/
# Use vendor_name 'xxxxxx' from 'CMakePresets.json'
bash build.sh
# Will generate ./build_out/custom_opp_euleros_aarch64.run
```

### Install the run file
```bash
# Execute run file
./build_out/custom_opp_euleros_aarch64.run
# Extracted to toolkit's vendors path (e.g., ascend-toolkit/latest/opp/vendors/xxxxxx)
```

### Try the example

An example is provided, which simulates one-layer training by combining forward computation, a simple MSE loss calculation, gradient computation, and parameter updates.

```bash
# Build the example
# Embed runtime search path of vendor 'xxxxxx' 
cd examples
bash build.sh 

# Execute the example
./build_out/simu_one_layer
```

## __Functionality__

Forward computation and Gradient computation are supported by four major functions.

### Forward Computation

#### `aclnnMultiScaleDeformableAttnFuncV2GetWorkspaceSize`

Retrieves the required workspace size and executor handle for running the forward computation.

##### ___Parameters___

| Parameter        | Type             | Direction | Shape   |Description  |
|------------------|-----------------|-----------|-------------|-----------------|
| `value`          | aclTensor        | input     | (bs, num_keys, num_heads, embed_dims)              | Input feature map tensor. Supports FLOAT/FLOAT16/BFLOAT16, non-contiguous, ND format. |
| `spatialShape`   | aclTensor        | input     | (num_levels, 2)                                    | Tensor storing height and width of each feature map level. Supports INT32/INT64, non-contiguous, ND format. |
| `levelStartIndex`| aclTensor        | input     | (num_levels,)                                      | Tensor with start indices of each feature map. Supports INT32/INT64, non-contiguous, ND format. |
| `location`       | aclTensor        | input     | (bs, num_queries, num_heads, num_levels, num_points, 2) | Sampling location tensor. Supports FLOAT/FLOAT16/BFLOAT16, non-contiguous, ND format. |
| `attnWeight`     | aclTensor        | input     | (bs, num_queries, num_heads, num_levels, num_points) | Sampling weight tensor. Supports FLOAT/FLOAT16/BFLOAT16, non-contiguous, ND format. |
| `output`         | aclTensor        | output    | (bs, num_queries, num_heads * embed_dims)         | Operator output tensor. Supports FLOAT/FLOAT16/BFLOAT16, non-contiguous, ND format. |
| `workspaceSize`  | uint64_t*        | output    | —                                                   | Size of workspace to allocate on device (in bytes).                          |
| `executor`       | aclOpExecutor**  | output    | —                                                   | Operator executor for forward computation.                                   |

##### ___Return___

| Value | Description |
|--------|-------------|
| `aclnnStatus` | Status code (e.g., `ACLNN_SUCCESS`). Check [documentation](https://gitee.com/ascend/cann-ops-adv/blob/v0.4-8.0.RC3.alpha003/docs/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81.md) for more details.|


#### `aclnnMultiScaleDeformableAttnFuncV2`

Executes the forward computation using the allocated workspace, executor, and specified stream.

##### ___Parameters___

| Parameter        | Type             | Direction | Description  |
|------------------|-----------------|-----------|-------------|
| `workspace` | void* | input | Device-side memory buffer allocated by users for temporary storage during operator computation. |
| `workspaceSize` | uint64_t | input | Size of the allocated workspace in bytes, obtained from `aclnnMultiScaleDeformableAttnFuncV2GetWorkspaceSize`. |
| `executor` | aclOpExecutor* | input | Operator executor containing the computation plan for forward operator. |
| `stream` | aclrtStream | input | AscendCL stream on which the operator executes, allowing asynchronous execution. |

##### ___Return___

| Value | Description |
|--------|-------------|
| `aclnnStatus` | Status code (e.g., `ACLNN_SUCCESS`). Check [documentation](https://gitee.com/ascend/cann-ops-adv/blob/v0.4-8.0.RC3.alpha003/docs/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81.md) for more details.|


### Gradient Computation

#### `aclnnMultiScaleDeformableAttnGradV2GetWorkspaceSize`

Retrieves the required workspace size and executor handle for the gradient computation.

##### ___Parameters___
Reuses the forward compuation parameters `value`, `spatialShape`, `levelStartIndex`, `location` and `attnWeight`, and adds the following parameters:

| Parameter        | Type             | Direction | Shape   |Description  |
|------------------|-----------------|-----------|-------------|-----------------|
| `gradOutput`          | aclTensor        | input     | (bs, num_queries, num_heads * embed_dims)                  | Gradient of the loss related to forward output.      |
| `gradValueOut`        | aclTensor        | output    | (bs, num_keys, num_heads, embed_dims)                      | Gradient related to input feature map `value`.       |
| `gradSamplingLocOut`  | aclTensor        | output    | (bs, num_queries, num_heads, num_levels, num_points, 2)    | Gradient related to sampling locations `location`.   |
| `gradAttnWeightOut`   | aclTensor        | output    | (bs, num_queries, num_heads, num_levels, num_points)       | Gradient related to attention weights `attnWeight`.  |
| `workspaceSize`       | uint64_t*        | output    | —                                                          | Workspace size to allocate on device.            |
| `executor`            | aclOpExecutor**  | output    | —                                                          | Operator executor for gradient computation.      |

##### ___Return___

| Value | Description |
|--------|-------------|
| `aclnnStatus` | Status code (e.g., `ACLNN_SUCCESS`). Check [documentation](https://gitee.com/ascend/cann-ops-adv/blob/v0.4-8.0.RC3.alpha003/docs/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81.md) for more details.|

#### `aclnnMultiScaleDeformableAttnGradV2`

Executes the gradient computation using allocated workspace and executor.

##### ___Parameters___

| Parameter        | Type             | Direction | Description  |
|------------------|-----------------|-----------|-------------|
| `workspace` | void* | input | Device-side memory buffer allocated by users for temporary storage during gradient computation. Must be at least `workspaceSize`. |
| `workspaceSize` | uint64_t | input | Size of the allocated workspace in bytes, obtained from `aclnnMultiScaleDeformableAttnGradV2GetWorkspaceSize`.|
| `executor` | aclOpExecutor* | input | Operator executor obtained from `aclnnMultiScaleDeformableAttnGradV2GetWorkspaceSize`. Encapsulates the computation plan for backward operator. |
| `stream` | aclrtStream | input | AscendCL stream for executing the backward operator asynchronously. |

##### ___Return___

| Value | Description |
|--------|-------------|
| `aclnnStatus` | Status code (e.g., `ACLNN_SUCCESS`). Check [documentation](https://gitee.com/ascend/cann-ops-adv/blob/v0.4-8.0.RC3.alpha003/docs/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81.md) for more details.|

## __Constraints__

Below are important constraints to ensure proper memory alignment, hardware vectorization, and efficient resource utilization on Ascend AI Cores.

| Parameter        | Constraint             | Notes   |
|------------------|------------------------|---------|
| embed_dims       | `embed_dims % 8 == 0` and `embed_dims <= 256` | Alignment and AICore vectorization requirement  |
| num_queries      | `32 <= num_queries < 500000`              | Total queries processed by the operator             |
| num_levels       | `num_levels <= 16`                         | Number of feature map levels                       |
| num_heads        | `num_heads <= 16`                          | Number of attention heads                          |
| num_points       | `num_points <= 16`                         | Number of sampling points per query per level      |
| batch_size       | implicit, typically small (<1024)         | Affects memory allocation in GM and UB              |
| map_height       | >=1, aligned implicitly by UB allocation  | Determines `numKeys = mapHeight * mapWidth`         |
| map_width        | >=1, aligned implicitly by UB allocation  | Determines `numKeys = mapHeight * mapWidth`         |
| num_keys         | `numKeys = mapHeight * mapWidth`          | Total key/value positions                           |

## __Two-stage Interface__

A single-operator API is generally defined as a two-stage interface. Check [documentation](https://gitee.com/ascend/cann-ops-adv/blob/v0.4-8.0.RC3.alpha003/docs/common/%E4%B8%A4%E6%AE%B5%E5%BC%8F%E6%8E%A5%E5%8F%A3.md) for more details.
### Forward computation

1. Call `aclnnMultiScaleDeformableAttnFuncV2GetWorkspaceSize` to obtain the required `workspaceSize` _(for allocating the `workspace` later)_ and `executor`
2. Call `aclnnMultiScaleDeformableAttnFuncV2` to perform the computation using the allocated `workspace` and `executor`, and obtain the result in the specified `stream`

### Gradient computation

- Follow a similar procedure by calling `aclnnMultiScaleDeformableAttnGradV2GetWorkspaceSize` and `aclnnMultiScaleDeformableAttnGradV2` respectively.


