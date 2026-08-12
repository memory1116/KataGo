# RTX 3080 Ti B8/S4 CUDA tactic plan

This directory contains the production-ready SM86 plan certified on an
NVIDIA GeForce RTX 3080 Ti. It is bound to exact 19x19 FP16/NHWC inference,
model SHA-256
`1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6`,
exact batch 8, and four inference streams on one device.

The complete 19-family search covered 123 candidates. Its final long gate
measured `1504.050501` physical nnEval/s and the final combined graph passed
an 8192-row full-FP32 replay. The certified runtime contract is explicit:

```text
cudaUseFP16=true
cudaUseGraphInference=false
cudaUseNHWC=true
cudaWarmupOnlyMaxBatchSize=true
nnBatchAwareDispatch=false
```

The loader validates and applies those values together with all tactic
overrides. Conflicting settings fail closed. The CUDA ordinal in the plan is
producer provenance only; receiver thread-to-device mapping remains in the
normal KataGo config.

For a single GPU, add the plan and four lanes to the normal config:

```cfg
cudaTacticPlanFile = /absolute/path/to/best-tactic-plan.json
cudaTacticPlanBatch = 8

numNNServerThreadsPerModel = 4
cudaDeviceToUseThread0 = 0
cudaDeviceToUseThread1 = 0
cudaDeviceToUseThread2 = 0
cudaDeviceToUseThread3 = 0
```

The plan ID is `sm86-rtx3080ti-abc187f1c89a74d4`; the file SHA-256 is
`933f50fb95fb0857a5f76191046e7b58997c98e235496d92d5a5e7a758ec6ff6`.
The strict B8/S4/T48 search benchmark and the full optimization history are
recorded under `optimization-history/sm86/rtx3080ti-e022/`.
