#include <iostream>
#include <vector>
#include <random>
#include <acl/acl.h>
#include "aclnn_multi_scale_deformable_attn_func_v2.h"
#include "aclnn_multi_scale_deformable_attn_grad_v2.h"

#define CHECK_RET(cond, return_expr) do { if (!(cond)) { return_expr; } } while (0)
#define LOG_PRINT(msg, ...) do { printf(msg, ##__VA_ARGS__); fflush(stdout); } while (0)

class MultiScaleDeformableAttnV2Simu {
public:
    MultiScaleDeformableAttnV2Simu(int bS, int nH, int mH, int mW, int eD, int nQ, int nL, int nP)
                    : batchSize(bS), numHeads(nH), mapHeight(mH), mapWidth(mW),
                    embedDims(eD), numQueries(nQ), numLevels(nL), numPoints(nP) {
        numKeys = mapHeight * mapWidth;
        valueShape       = {batchSize, numHeads, numKeys, embedDims};
        spatialShapeShape = {1,2};
        locationShape    = {batchSize, numQueries, numLevels, 1, numPoints, 2};
        attnWeightShape  = {batchSize, numQueries, numLevels, 1, numPoints};
        outputShape      = {batchSize, numQueries, embedDims};
        levelStartIndexShape = {1};

        CHECK_RET(aclInit(nullptr) == ACL_SUCCESS, throw std::runtime_error("ACL init failed\n"); );
        CHECK_RET(aclrtSetDevice(GetDeviceZero()) == ACL_SUCCESS, throw std::runtime_error("SetDevice failed\n"); );
        CHECK_RET(aclrtCreateStream(&stream) == ACL_SUCCESS, throw std::runtime_error("CreateStream failed\n"); );
    }

    ~MultiScaleDeformableAttnV2Simu(){
        ReleaseDeviceMemory();
        aclrtDestroyStream(stream);
        aclrtResetDevice(0);
        aclFinalize();
    }

    int GetDeviceZero() {
        uint32_t deviceCount = 0;
        aclError ret = aclrtGetDeviceCount(&deviceCount);
        if(ret != ACL_SUCCESS){
            LOG_PRINT("Get device count failed, error: %d\n", ret);
            return -1;
        }
        LOG_PRINT("- Total NPU devices: %d\n", deviceCount);

        for (uint32_t i = 0; i < deviceCount; ++i) {
            LOG_PRINT("%4cAvailable Device ID: %d\n", ' ', i);
        }
        return 0;
    }

    size_t GetShapeSize(const std::vector<int64_t> &shape) {
        size_t size=1;
        for(auto d : shape) size *= d;
        return size;
    }

    void InitializeData() {
        spatialShapeHost = {mapHeight, mapWidth};
        levelStartIndexHost = {0};

        outputHost.resize(GetShapeSize(outputShape), 0.0f);
        valueHost.resize(GetShapeSize(valueShape));
        attnWeightHost.resize(GetShapeSize(attnWeightShape));
        locationHost.resize(GetShapeSize(locationShape));
        gradOutputHost.resize(GetShapeSize(outputShape), 1.0f);
        gradValueHost.resize(GetShapeSize(valueShape), 0.0f);
        gradLocationHost.resize(GetShapeSize(locationShape), 0.0f);
        gradAttnHost.resize(GetShapeSize(attnWeightShape), 0.0f);
        targetHost.resize(GetShapeSize(outputShape));
        
        std::mt19937 gen(102); 
        std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);
        std::uniform_real_distribution<float> attn_dist(0.1f, 1.0f);
        std::uniform_real_distribution<float> loc_dist(0.0f, 1.0f); 

        for(auto &v : valueHost) v = val_dist(gen);
        for(auto &v : attnWeightHost) v = attn_dist(gen);

        size_t idx = 0;
        for (int b = 0; b < batchSize; ++b)
            for (int q = 0; q < numQueries; ++q)
                for (int l = 0; l < numLevels; ++l)
                    for (int p = 0; p < numPoints; ++p){
                        locationHost[idx++] = loc_dist(gen); // y
                        locationHost[idx++] = loc_dist(gen); // x
                    }

        for(auto &t : targetHost) t = val_dist(gen);
    }

    int ForwardComputation() {
        CHECK_RET(CreateAclTensor("value", valueHost, valueShape, &valueDevice, ACL_FLOAT, &value)==ACL_SUCCESS, return -1);
        CHECK_RET(CreateAclTensor("spatialShapes", spatialShapeHost, spatialShapeShape, &spatialDevice, ACL_INT32, &spatial)==ACL_SUCCESS, return -1);
        CHECK_RET(CreateAclTensor("levelStartIndex", levelStartIndexHost, levelStartIndexShape, &levelStartDevice, ACL_INT32, &levelStart)==ACL_SUCCESS, return -1);
        CHECK_RET(CreateAclTensor("samplingLocations", locationHost, locationShape, &locationDevice, ACL_FLOAT, &location)==ACL_SUCCESS, return -1);
        CHECK_RET(CreateAclTensor("attentionWeights", attnWeightHost, attnWeightShape, &attnDevice, ACL_FLOAT, &attn)==ACL_SUCCESS, return -1);
        CHECK_RET(CreateAclTensor("output", outputHost, outputShape, &outputDevice, ACL_FLOAT, &output)==ACL_SUCCESS, return -1);

        auto ret = aclnnMultiScaleDeformableAttnFuncV2GetWorkspaceSize(value, spatial, levelStart, location, attn, output, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Forward GetWorkspaceSize failed\n"); return -1);

        if(workspaceSize>0){
            aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Forward workspace malloc failed\n"); return -1);
        }

        ret = aclnnMultiScaleDeformableAttnFuncV2(workspace, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Forward failed\n"); return -1);

        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Forward sync failed\n"); return -1);

        ret = aclrtMemcpy(outputHost.data(), GetShapeSize(outputShape)*sizeof(float), outputDevice, GetShapeSize(outputShape)*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Memcpy forward output failed\n"); return -1);

        LOG_PRINT("* Forward Computation Done. \n");
        PrintTensor("  Forward Output ", outputHost);
        return 0;
    }

    int GradientComputation() {
        // MSE Loss Gradient
        float loss = 0.0f;
        for (size_t i=0; i<outputHost.size(); ++i) {
            float diff = outputHost[i] - targetHost[i];
            loss += diff * diff;
        }
        loss /= outputHost.size();
        LOG_PRINT("* MSE Loss: %f\n", loss);
        for(size_t i=0;i<outputHost.size();++i)
            gradOutputHost[i] = 2.0f*(outputHost[i]-targetHost[i])/outputHost.size();

        CHECK_RET(CreateAclTensor("gradOutput", gradOutputHost, outputShape, &gradOutputDevice, ACL_FLOAT, &gradOutput)==ACL_SUCCESS, return -1);
        CHECK_RET(CreateAclTensor("gradValue", gradValueHost, valueShape, &gradValueDevice, ACL_FLOAT, &gradValue)==ACL_SUCCESS, return -1);
        CHECK_RET(CreateAclTensor("gradLocation", gradLocationHost, locationShape, &gradLocationDevice, ACL_FLOAT, &gradLocation)==ACL_SUCCESS, return -1);
        CHECK_RET(CreateAclTensor("gradAttn", gradAttnHost, attnWeightShape, &gradAttnDevice, ACL_FLOAT, &gradAttn)==ACL_SUCCESS, return -1);

        auto ret = aclrtMemcpy(gradOutputDevice, GetShapeSize(outputShape)*sizeof(float), gradOutputHost.data(), GetShapeSize(outputShape)*sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Memcpy gradOutput failed\n"); return -1);

        ret = aclnnMultiScaleDeformableAttnGradV2GetWorkspaceSize( value, spatial, levelStart, location, attn, gradOutput, gradValue, gradLocation, gradAttn, &gradWorkspaceSize, &gradExecutor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Grad GetWorkspaceSize failed\n"); return -1);

        if(gradWorkspaceSize>0) {
            ret = aclrtMalloc(&gradWorkspace, gradWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Grad workspace malloc failed\n"); return -1);
        }

        ret = aclnnMultiScaleDeformableAttnGradV2(gradWorkspace, gradWorkspaceSize, gradExecutor, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Gradient failed\n"); return -1);

        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Gradient sync failed\n"); return -1);

        ret = aclrtMemcpy(gradValueHost.data(), GetShapeSize(valueShape)*sizeof(float), gradValueDevice, GetShapeSize(valueShape)*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

        ret |= aclrtMemcpy(gradLocationHost.data(), GetShapeSize(locationShape)*sizeof(float), gradLocationDevice, GetShapeSize(locationShape)*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

        ret |= aclrtMemcpy(gradAttnHost.data(), GetShapeSize(attnWeightShape)*sizeof(float), gradAttnDevice, GetShapeSize(attnWeightShape)*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Memcpy backward output failed\n"); return -1);

        LOG_PRINT("* Gradient Computation Done. \n");
        PrintTensor("  Grad Value", gradValueHost);
        PrintTensor("  Grad Location", gradLocationHost);
        PrintTensor("  Grad Attention", gradAttnHost);
        return 0;
    }

    void UpdateParameter(float lr=0.01f){
        for(size_t i=0;i<valueHost.size();++i) valueHost[i] -= lr*gradValueHost[i];
        for(size_t i=0;i<locationHost.size();++i) locationHost[i] -= lr*gradLocationHost[i];
        for(size_t i=0;i<attnWeightHost.size();++i) attnWeightHost[i] -= lr*gradAttnHost[i];
    }

    template <typename T>
    void PrintTensor(const std::string &name, const std::vector<T>& data, size_t limit=16) {
        std::cout << name << ": [";
        size_t printCount = std::min(limit, data.size());
        for (size_t i=0; i<printCount; ++i) {
            std::cout << data[i];
            if (i != printCount-1) std::cout <<","; 
        }
        if (data.size() > limit) std::cout <<", ...";
        std::cout <<"]" <<std::endl;
    }

private:
    template<typename T>
    aclError CreateAclTensor(const std::string &name, std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr, aclDataType dataType, aclTensor **tensor){
        size_t size = GetShapeSize(shape)*sizeof(T);
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s malloc failed\n", name.c_str()); return ret);

        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s memcpy failed\n", name.c_str()); return ret);

    #if 1
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; --i) strides[i] = shape[i + 1] * strides[i + 1];

        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    #else
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, ACL_FORMAT_NCHW, shape.data(), shape.size(), *deviceAddr);
    #endif
        CHECK_RET(*tensor!=nullptr, LOG_PRINT("%s create tensor failed\n", name.c_str()); return ACL_ERROR);
        return ACL_SUCCESS;
    }

    void ReleaseDeviceMemory(){
        aclDestroyTensor(value); 
        aclDestroyTensor(spatial); 
        aclDestroyTensor(levelStart);
        aclDestroyTensor(location); 
        aclDestroyTensor(attn); 
        aclDestroyTensor(output);
        aclDestroyTensor(gradOutput); 
        aclDestroyTensor(gradValue); 
        aclDestroyTensor(gradLocation); 
        aclDestroyTensor(gradAttn);
        aclrtFree(valueDevice); 
        aclrtFree(spatialDevice); 
        aclrtFree(levelStartDevice);
        aclrtFree(locationDevice); 
        aclrtFree(attnDevice); 
        aclrtFree(outputDevice);
        aclrtFree(gradOutputDevice); 
        aclrtFree(gradValueDevice); 
        aclrtFree(gradLocationDevice); 
        aclrtFree(gradAttnDevice);
        if(workspace) aclrtFree(workspace); 
        if(gradWorkspace) aclrtFree(gradWorkspace);
    }

private:
    int batchSize, numHeads, mapHeight, mapWidth, numKeys, embedDims, numQueries, numLevels, numPoints;
    std::vector<int64_t> valueShape, spatialShapeShape, locationShape, attnWeightShape, outputShape, levelStartIndexShape;
    std::vector<int32_t> spatialShapeHost, levelStartIndexHost;
    std::vector<float> valueHost, attnWeightHost, locationHost, outputHost;
    std::vector<float> gradOutputHost, gradValueHost, gradLocationHost, gradAttnHost, targetHost;

    void *valueDevice=nullptr, *spatialDevice=nullptr, *levelStartDevice=nullptr;
    void *locationDevice=nullptr, *attnDevice=nullptr, *outputDevice=nullptr, *workspace=nullptr;

    aclTensor *value=nullptr, *spatial=nullptr, *levelStart=nullptr, *location=nullptr, *attn=nullptr, *output=nullptr;
    uint64_t workspaceSize=0; 
    aclOpExecutor* executor=nullptr; 

    void *gradOutputDevice=nullptr, *gradValueDevice=nullptr, *gradLocationDevice=nullptr; 
    void *gradAttnDevice=nullptr, *gradWorkspace=nullptr;
    aclTensor *gradOutput=nullptr, *gradValue=nullptr, *gradLocation=nullptr, *gradAttn=nullptr;
    uint64_t gradWorkspaceSize=0; 
    aclOpExecutor* gradExecutor=nullptr; 

    aclrtStream stream=nullptr;
};

int main(){
    try {
        MultiScaleDeformableAttnV2Simu msda_v2_simu(1,1,8,8,8,32,1,4);
        int epochs = 5; //100000;
        float lr = 0.01f;
        msda_v2_simu.InitializeData(); 
        for(int e=0;e<epochs;++e){
            LOG_PRINT("\n!!!! Epoch %d !!!!\n", e+1);
            msda_v2_simu.ForwardComputation();
            msda_v2_simu.GradientComputation();
            msda_v2_simu.UpdateParameter(lr);
        }
        return 0;
    } catch (const std::exception &e) {
        LOG_PRINT("try Layer failed: %s\n", e.what());
        return -1;
    }
}