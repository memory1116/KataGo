#include "../neuralnet/cudatacticplan.h"

#include "../core/fileutils.h"
#include "../external/nlohmann_json/json.hpp"

#if defined(USE_CUDA_BACKEND)
#include "../neuralnet/cudaincludes.h"
#include "../neuralnet/cudaerrorcheck.h"
#endif

#include <set>

using namespace std;
using json = nlohmann::json;

namespace CudaTacticPlan {

namespace {

int requiredInt(const json& obj, const string& key, const string& context) {
  if(!obj.is_object() || !obj.contains(key) || !obj[key].is_number_integer())
    throw StringError("CUDA tactic plan " + context + " requires integer " + key);
  return obj[key].get<int>();
}

string requiredString(const json& obj, const string& key, const string& context) {
  if(!obj.is_object() || !obj.contains(key) || !obj[key].is_string())
    throw StringError("CUDA tactic plan " + context + " requires string " + key);
  return obj[key].get<string>();
}

bool requiredBool(const json& obj, const string& key, const string& context) {
  if(!obj.is_object() || !obj.contains(key) || !obj[key].is_boolean())
    throw StringError("CUDA tactic plan " + context + " requires boolean " + key);
  return obj[key].get<bool>();
}

uint64_t requiredUint64(const json& obj, const string& key, const string& context) {
  if(!obj.is_object() || !obj.contains(key) || !obj[key].is_number_unsigned())
    throw StringError("CUDA tactic plan " + context + " requires unsigned integer " + key);
  return obj[key].get<uint64_t>();
}

DeviceRequirements parseDeviceRequirements(const json& target) {
  if(!target.contains("cuda_device_capabilities_at_scan") ||
     !target["cuda_device_capabilities_at_scan"].is_array() ||
     target["cuda_device_capabilities_at_scan"].empty())
    throw StringError("CUDA tactic plan has no producer device capabilities");
  const json& device = target["cuda_device_capabilities_at_scan"][0];
  DeviceRequirements result;
  result.name = requiredString(device,"name","device target");
  result.computeCapabilityMajor = requiredInt(device,"computeCapabilityMajor","device target");
  result.computeCapabilityMinor = requiredInt(device,"computeCapabilityMinor","device target");
  result.multiProcessorCount = requiredInt(device,"multiProcessorCount","device target");
  result.maxThreadsPerBlock = requiredInt(device,"maxThreadsPerBlock","device target");
  result.maxThreadsPerMultiprocessor = requiredInt(device,"maxThreadsPerMultiProcessor","device target");
  result.regsPerMultiprocessor = requiredInt(device,"regsPerMultiprocessor","device target");
  result.sharedMemPerBlockOptin = requiredInt(device,"sharedMemPerBlockOptin","device target");
  result.sharedMemPerMultiprocessor = requiredInt(device,"sharedMemPerMultiprocessor","device target");
  result.l2CacheSize = requiredInt(device,"l2CacheSize","device target");
  result.memoryBusWidth = requiredInt(device,"memoryBusWidth","device target");
  result.asyncEngineCount = requiredInt(device,"asyncEngineCount","device target");
  result.concurrentKernels = requiredBool(device,"concurrentKernels","device target");
  result.totalGlobalMem = requiredUint64(device,"totalGlobalMem","device target");
  return result;
}

int selectBatch(ConfigParser& cfg, const json& payload) {
  if(!payload.contains("batches") || !payload["batches"].is_array() || payload["batches"].empty())
    throw StringError("CUDA tactic plan has no certified batches");
  set<int> batches;
  for(const json& item : payload["batches"]) {
    if(!item.is_number_integer())
      throw StringError("CUDA tactic plan batches must be integers");
    batches.insert(item.get<int>());
  }

  int batch = -1;
  if(cfg.contains("cudaTacticPlanBatch"))
    batch = cfg.getInt("cudaTacticPlanBatch",1,65536);
  else if(batches.size() == 1)
    batch = *batches.begin();
  else
    throw StringError(
      "CUDA tactic plan contains multiple batches; set cudaTacticPlanBatch explicitly"
    );
  if(batches.find(batch) == batches.end())
    throw StringError(
      "CUDA tactic plan has no certified B" + Global::intToString(batch) + " selection"
    );
  return batch;
}

map<string,string> parseTacticOverrides(const json& payload, int batch) {
  const string batchKey = Global::intToString(batch);
  if(!payload.contains("apply") || !payload["apply"].is_object() ||
     !payload["apply"].contains("per_batch_tactic_overrides") ||
     !payload["apply"]["per_batch_tactic_overrides"].is_object() ||
     !payload["apply"]["per_batch_tactic_overrides"].contains(batchKey))
    throw StringError("CUDA tactic plan has no B" + batchKey + " apply mapping");
  const json& value = payload["apply"]["per_batch_tactic_overrides"][batchKey];
  if(!value.is_string())
    throw StringError("CUDA tactic plan B" + batchKey + " apply mapping is not a string");
  map<string,string> overrides = ConfigParser::parseCommaSeparated(value.get<string>());
  if(overrides.empty())
    throw StringError("CUDA tactic plan B" + batchKey + " apply mapping is empty");
  for(const auto& item : overrides) {
    if(item.first.compare(0,4,"cuda") != 0)
      throw StringError("CUDA tactic plan apply mapping contains non-CUDA key " + item.first);
    if(item.first.find("DeviceToUse") != string::npos ||
       item.first.find("GpuToUse") != string::npos)
      throw StringError("CUDA tactic plan apply mapping must not contain a producer device ordinal");
  }
  return overrides;
}

}

unique_ptr<Plan> loadAndApply(
  ConfigParser& cfg,
  Logger& logger,
  int nnXLen,
  int nnYLen,
  bool& requireExactNNLen
) {
  if(!cfg.contains("cudaTacticPlanFile"))
    return nullptr;
#if !defined(USE_CUDA_BACKEND)
  (void)logger;
  (void)nnXLen;
  (void)nnYLen;
  (void)requireExactNNLen;
  throw StringError("cudaTacticPlanFile requires a CUDA backend build");
#else
  (void)nnXLen;
  (void)nnYLen;
  (void)requireExactNNLen;
  const string path = cfg.getString("cudaTacticPlanFile");
  string contents;
  string fileSha256;
  FileUtils::loadFileIntoString(path,"",contents,&fileSha256);

  json payload;
  try {
    payload = json::parse(contents);
  }
  catch(const nlohmann::detail::exception& e) {
    throw StringError("Could not parse CUDA tactic plan " + path + ": " + e.what());
  }
  if(!payload.is_object() || requiredInt(payload,"schema","root") != 1 ||
     requiredString(payload,"kind","root") != "cuda-tactic-plan")
    throw StringError("Unsupported CUDA tactic plan schema in " + path);
  if(!payload.value("production_ready",false) ||
     payload.value("status",string()) != "complete_long_stable")
    logger.write("WARNING: loading an uncertified CUDA tactic preset from " + path);

  if(!payload.contains("target") || !payload["target"].is_object())
    throw StringError("CUDA tactic plan has no target");
  const json& target = payload["target"];
  const string architecture = requiredString(target,"architecture","target");
  if(architecture != "sm86" && architecture != "sm89" && architecture != "sm120")
    throw StringError("Unsupported CUDA tactic plan architecture " + architecture);
  const int batch = selectBatch(cfg,payload);
  const int streams = requiredInt(target,"streams","target");
  if(streams <= 0)
    throw StringError("CUDA tactic plan target stream count must be positive");
  const int evaluatorThreads = cfg.contains("numNNServerThreadsPerModel") ?
    cfg.getInt("numNNServerThreadsPerModel",1,65536) : streams;
  map<string,string> tacticOverrides = parseTacticOverrides(payload,batch);

  map<string,string> runtimeOverrides;
  // The preset supplies tactic choices only. Model, physical batch, lane count,
  // precision, layout, graph, and scheduler settings belong to the user. The
  // specialized backend already falls back to the official implementation
  // when a model does not support its optimized forward.
  for(const auto& item : tacticOverrides) {
    if(!cfg.contains(item.first))
      runtimeOverrides[item.first] = item.second;
  }
  if(!cfg.contains("cudaTacticPlanBatch"))
    runtimeOverrides["cudaTacticPlanBatch"] = Global::intToString(batch);
  if(architecture == "sm86" || architecture == "sm89") {
    if(!cfg.contains("cudaSm89Backend"))
      runtimeOverrides["cudaSm89Backend"] = "true";
    if(!cfg.contains("cudaSm89Forward"))
      runtimeOverrides["cudaSm89Forward"] = "true";
  }
  else if(!cfg.contains("cudaSm120Backend"))
    runtimeOverrides["cudaSm120Backend"] = "true";
  cfg.overrideKeys(runtimeOverrides);

  unique_ptr<Plan> plan = make_unique<Plan>();
  plan->path = path;
  plan->fileSha256 = fileSha256;
  plan->planId = requiredString(payload,"plan_id","root");
  plan->planSha256 = requiredString(payload,"plan_sha256","root");
  plan->architecture = architecture;
  plan->gpuClass = requiredString(target,"gpu_class","target");
  plan->modelSha256 = target.value("model_sha256",string());
  plan->batchSize = batch;
  plan->streamsPerDevice = streams;
  plan->device = parseDeviceRequirements(target);
  plan->tacticOverrides = tacticOverrides;

  logger.write(
    "Loaded CUDA tactic plan " + plan->planId + " from " + path +
    " fileSha256=" + fileSha256 + " B" + Global::intToString(batch) +
    " streamsPerDevice=" + Global::intToString(streams) +
    " runtimeBatch=" +
      (cfg.contains("nnMaxBatchSize") ? Global::intToString(cfg.getInt("nnMaxBatchSize")) : "auto") +
    " evaluatorThreads=" + Global::intToString(evaluatorThreads) +
    " (model/device/topology certification metadata is advisory)"
  );
  return plan;
#endif
}

void validateDevices(const Plan& plan, const vector<int>& gpuIdxByServerThread) {
#if !defined(USE_CUDA_BACKEND)
  (void)plan;
  (void)gpuIdxByServerThread;
  throw StringError("CUDA tactic plan device validation requires a CUDA backend build");
#else
  set<int> devices;
  for(int deviceIdx : gpuIdxByServerThread)
    devices.insert(deviceIdx < 0 ? 0 : deviceIdx);
  for(int deviceIdx : devices) {
    cudaDeviceProp prop;
    CUDA_ERR("CUDA tactic plan",cudaGetDeviceProperties(&prop,deviceIdx));
    const bool architectureMatches =
      (plan.architecture == "sm86" && prop.major == 8 && prop.minor == 6) ||
      (plan.architecture == "sm89" && prop.major == 8 && prop.minor == 9) ||
      (plan.architecture == "sm120" && prop.major == 12 && prop.minor == 0);
    if(!architectureMatches)
      throw StringError(
        "CUDA tactic preset " + plan.architecture + " cannot run on compute capability " +
        Global::intToString(prop.major) + "." + Global::intToString(prop.minor)
      );
  }
#endif
}

}
