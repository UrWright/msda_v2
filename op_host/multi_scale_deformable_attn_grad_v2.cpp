#include "multi_scale_deformable_attn_grad_v2.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ge;
using namespace std;
using namespace AscendC;

namespace optiling {
    static ge::graphStatus TilingFuncForMultiScaleDeformableAttnGradV2(gert::TilingContext *context) {
        MultiScaleDeformableAttnGradV2TilingData tiling;

        auto valueShape = context->GetInputTensor(0)->GetStorageShape();
        auto samplingLocationsShape = context->GetInputTensor(3)->GetStorageShape();

        auto platformInfoptr = context->GetPlatformInfo();
        if (platformInfoptr == nullptr) {
            return ge::GRAPH_FAILED;
        }
        auto ascendplatformInfo = platform_ascendc::PlatformAscendC(platformInfoptr);
        uint32_t coreNum = ascendplatformInfo.GetCoreNumAiv();
        context->SetBlockDim(coreNum);

        tiling.set_batchSize(valueShape.GetDim(0));
        tiling.set_numKeys(valueShape.GetDim(2));
        tiling.set_numHeads(valueShape.GetDim(1));
        tiling.set_embedDims(valueShape.GetDim(3));
        tiling.set_numLevels(samplingLocationsShape.GetDim(3));
        tiling.set_numQueries(samplingLocationsShape.GetDim(1));
        tiling.set_numPoints(samplingLocationsShape.GetDim(5));
        tiling.set_coreNum(coreNum);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 16 * 1024 * 1024;
        return ge::GRAPH_SUCCESS;
    }
}

namespace ge {
    static ge::graphStatus InferShapeForMultiScaleDeformableAttnGradV2(gert::InferShapeContext *context) {
        const gert::Shape *value_shape = context->GetInputShape(0);
        if (value_shape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const gert::Shape *sampling_locations_shape = context->GetInputShape(3);
        if (sampling_locations_shape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        gert::Shape *grad_value_shape = context->GetOutputShape(0);
        gert::Shape *grad_sample_loc_shape = context->GetOutputShape(1);
        gert::Shape *grad_attn_weight_shape = context->GetOutputShape(2);
        if ((grad_value_shape == nullptr) || (grad_sample_loc_shape == nullptr) || (grad_attn_weight_shape == nullptr)) {
            return ge::GRAPH_FAILED;
        }
        grad_value_shape->AppendDim(value_shape->GetDim(0));
        grad_value_shape->AppendDim(value_shape->GetDim(1));
        grad_value_shape->AppendDim(value_shape->GetDim(2));
        grad_value_shape->AppendDim(value_shape->GetDim(3));
        grad_sample_loc_shape->AppendDim(sampling_locations_shape->GetDim(0));
        grad_sample_loc_shape->AppendDim(sampling_locations_shape->GetDim(1));
        grad_sample_loc_shape->AppendDim(sampling_locations_shape->GetDim(2));
        grad_sample_loc_shape->AppendDim(sampling_locations_shape->GetDim(3));
        grad_sample_loc_shape->AppendDim(sampling_locations_shape->GetDim(4));
        grad_sample_loc_shape->AppendDim(sampling_locations_shape->GetDim(5));
        grad_attn_weight_shape->AppendDim(sampling_locations_shape->GetDim(0));
        grad_attn_weight_shape->AppendDim(sampling_locations_shape->GetDim(1));
        grad_attn_weight_shape->AppendDim(sampling_locations_shape->GetDim(2));
        grad_attn_weight_shape->AppendDim(sampling_locations_shape->GetDim(3));
        grad_attn_weight_shape->AppendDim(sampling_locations_shape->GetDim(5));
        return GRAPH_SUCCESS;
    }
}

namespace ops {
    class MultiScaleDeformableAttnGradV2 : public OpDef {
    public:
        explicit MultiScaleDeformableAttnGradV2(const char *name) : OpDef(name) {
            this->Input("value")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND})
		.AutoContiguous();
            this->Input("spatial_shapes")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND})
		.AutoContiguous();
            this->Input("level_start_index")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND})
		.AutoContiguous();
            this->Input("sampling_loc")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND})
		.AutoContiguous();
            this->Input("attn_weight")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND})
		.AutoContiguous();
            this->Input("grad_output")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND})
		.AutoContiguous();
            this->Output("grad_value")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Output("grad_sampling_loc")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Output("grad_attn_weight")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->SetInferShape(ge::InferShapeForMultiScaleDeformableAttnGradV2);

            this->AICore()
                .SetTiling(optiling::TilingFuncForMultiScaleDeformableAttnGradV2);

            this->AICore().AddConfig("ascend910b");
        }
    };

    OP_ADD(MultiScaleDeformableAttnGradV2);
}
