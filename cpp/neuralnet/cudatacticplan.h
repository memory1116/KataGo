#ifndef NEURALNET_CUDATACTICPLAN_H_
#define NEURALNET_CUDATACTICPLAN_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../core/config_parser.h"
#include "../core/logger.h"

namespace CudaTacticPlan {

struct DeviceRequirements {
  std::string name;
  int computeCapabilityMajor;
  int computeCapabilityMinor;
  int multiProcessorCount;
  int maxThreadsPerBlock;
  int maxThreadsPerMultiprocessor;
  int regsPerMultiprocessor;
  int sharedMemPerBlockOptin;
  int sharedMemPerMultiprocessor;
  int l2CacheSize;
  int memoryBusWidth;
  int asyncEngineCount;
  bool concurrentKernels;
  uint64_t totalGlobalMem;
};

struct Plan {
  std::string path;
  std::string fileSha256;
  std::string planId;
  std::string planSha256;
  std::string architecture;
  std::string gpuClass;
  std::string modelSha256;
  int batchSize;
  int streamsPerDevice;
  DeviceRequirements device;
  std::map<std::string,std::string> tacticOverrides;
};

// Loads and applies cudaTacticPlanFile when configured. The selected plan is
// used as an optimization preset. Runtime model, batch, lane count, and device
// selection remain user-controlled. Unsupported model shapes retain the
// official CUDA path, and device ordinals remain receiver-local.
std::unique_ptr<Plan> loadAndApply(
  ConfigParser& cfg,
  Logger& logger,
  int nnXLen,
  int nnYLen,
  bool& requireExactNNLen
);

// Ensures only that every receiver belongs to the CUDA architecture family for
// which the preset was compiled. Product name and performance characteristics
// are advisory rather than runtime authorization checks.
void validateDevices(
  const Plan& plan,
  const std::vector<int>& gpuIdxByServerThread
);

}

#endif  // NEURALNET_CUDATACTICPLAN_H_
