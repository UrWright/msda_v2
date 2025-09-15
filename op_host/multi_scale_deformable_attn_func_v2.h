#ifndef MULIT_SCALE_DEFOEMABLE_ATTN_FUNC_V2_TILING_H
#define MULIT_SCALE_DEFOEMABLE_ATTN_FUNC_V2_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
    BEGIN_TILING_DATA_DEF(MultiScaleDeformableAttnFuncV2TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize)
    TILING_DATA_FIELD_DEF(uint32_t, numKeys)
    TILING_DATA_FIELD_DEF(uint32_t, numHeads)
    TILING_DATA_FIELD_DEF(uint32_t, embedDims)
    TILING_DATA_FIELD_DEF(uint32_t, numLevels)
    TILING_DATA_FIELD_DEF(uint32_t, numQueries)
    TILING_DATA_FIELD_DEF(uint32_t, numPoints)
    TILING_DATA_FIELD_DEF(uint32_t, coreNum)

    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(MultiScaleDeformableAttnFuncV2, MultiScaleDeformableAttnFuncV2TilingData)
} // namespace optiling
#endif // MULIT_SCALE_DEFOEMABLE_ATTN_FUNC_V2_TILING_H
