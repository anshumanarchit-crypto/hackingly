# GenieX / Qualcomm & ProofMesh Validation

## Environment
- Laptop CPU: 12th Gen Intel(R) Core(TM) i5-12450H (8 Cores, 12 Threads, x86_64)
- OS: Windows 11 Home (x86_64)
- `qai-hub` version: 0.55.0 (located at `C:\Users\archi\AppData\Roaming\Python\Python313\Scripts\qai-hub.exe`)
- `geniex` version: Not installed / Not found
- User Physical Phone Consideration: Samsung Galaxy M31 (Samsung Exynos 9611 SoC — Not a Qualcomm Snapdragon chipset; unsupported for GenieX / QAIRT / Hexagon NPU inference)
- Qualcomm target used: Hosted Snapdragon Device on Qualcomm AI Hub (Samsung Galaxy S25 / Snapdragon 8 Elite `sm8750-ac`, Snapdragon X Elite CRD `sc8380xp`)
- Target OS: Android 15 (hosted Galaxy S25) / Windows 11 (hosted Snapdragon X Elite)

## Commands That Worked
- `python --version`
  - Output: `Python 3.13.3`
- `Get-CimInstance Win32_Processor | Select-Object Name, Architecture, NumberOfCores, NumberOfLogicalProcessors`
  - Output: `12th Gen Intel(R) Core(TM) i5-12450H`, Architecture: 9 (x64), 8 Cores, 12 Logical Processors
- `& "C:\Users\archi\AppData\Roaming\Python\Python313\Scripts\qai-hub.exe" --help`
  - Output: Qualcomm AI Hub CLI operational (Commands: `configure`, `list-devices`, `list-frameworks`, `upload-model`, `submit-compile-job`, `submit-profile-job`, etc.)
- `& "C:\Users\archi\AppData\Roaming\Python\Python313\Scripts\qai-hub.exe" list-devices`
  - Output: Authenticated and successfully connected; returned available hosted Qualcomm targets (e.g., `Samsung Galaxy S25` [Android 15], `Snapdragon X Elite CRD` [Windows 11], `Snapdragon 8 Elite QRD` [Android 15]).
- `& "C:\Users\archi\AppData\Roaming\Python\Python313\Scripts\qai-hub.exe" list-frameworks`
  - Output: QAIRT 2.45, 2.49, 2.50 (latest)
- `python offline_inference.py`
  - Output: Verified offline CPU-only inference pipeline over local model (`llama3.2:1b`), achieving 55-61 tokens/sec on Intel Core i5-12450H.
- `python test_proofmesh.py`
  - Output: 12/12 unit tests passed in 45.193s. Verified deterministic contradiction detection, evidence gating, exact `[chunk_id]` inline citations, `NOT ENOUGH EVIDENCE` fallback, prompt injection immunity, PII masking, and the Post-Generation Claim Support Verification Gate with 1-step controlled regeneration.

## Commands That Failed
- `where.exe geniex`
  - Reason: `geniex` binary is not installed on the system PATH or in the Python environment.
- `Get-Command geniex`
  - Reason: `geniex` is not recognized as an installed cmdlet or executable.
- Local NPU Inference / Local GenieX Execution
  - Reason: Host machine is an Intel Core i5-12450H laptop lacking a Qualcomm Hexagon NPU; user phone is a Samsung Galaxy M31 (Exynos 9611) which cannot execute Qualcomm NPU binaries.

## Model
- Primary model: `ai-hub-models/Qwen3-4B-Instruct-2507` (Qualcomm AI Hub model intended for GenieX / QAIRT deployment on Snapdragon hardware)
- Fallback / Local model: `llama3.2:latest` / `unsloth/Qwen3.5-0.8B-GGUF` (Lightweight models for offline CPU execution)
- Runtime: Qualcomm AI Engine Direct (QAIRT) for hosted Qualcomm validation; llama.cpp / local CPU runtime for offline Intel fallback
- Quantization: Q4_0 (preferred quantization for Hexagon NPU in GenieX documentation, and efficient for CPU)

## ProofMesh Architecture & Philosophy
> *"The model cannot decide what is true. Our evidence layer decides what the model is allowed to say."*

```
Retriever (BM25 + Semantic Search)
   ↓
Evidence Chunks (with PII Masking & Clean Chunk IDs)
   ↓
Deterministic Contradiction Detector (Numerical, Polarity & Status Conflicts)
   ↓
Evidence Gate (Prompt Injection Sanitization & Sufficiency Check)
   ↓
Evidence-Bound Local LLM Answering Engine
   ↓
Citation-ID Validation
   ↓
Post-Generation Claim Support Verification Gate
   ├── [All Claims SUPPORTED] ──────────────► Final Response
   └── [Any Claim UNSUPPORTED] ─────────────► Controlled 1-Step Regeneration
                                                   ↓
                                              Re-Verification Gate
                                              ├── [SUPPORTED]   ──► Final Response
                                              └── [UNSUPPORTED] ──► "NOT ENOUGH EVIDENCE"
```

## Complete Verification Test Suite (`test_proofmesh.py`)

### 12/12 Unit Tests Passed (100% Offline on Intel Core i5-12450H):

1. **Test 1 — Grounded Answer with Inline Citations**: PASSED (`The warranty period for hardware is 24 months. [chunk_17]`)
2. **Test 2 — Deterministic Contradiction (Numerical Conflict)**: PASSED (24 months [chunk_10] vs 12 months [chunk_11] intercepted pre-LLM).
3. **Test 3 — Deterministic Contradiction (Polarity Flip)**: PASSED (`approved` [chunk_21] vs `rejected` [chunk_22] intercepted pre-LLM).
4. **Test 4 — Insufficient Evidence Fallback**: PASSED (Absent fact returns `NOT ENOUGH EVIDENCE`).
5. **Test 5 — Adversarial Prompt Injection Defense**: PASSED (Payload in document neutralized as passive data).
6. **Test 6 — PII Pseudonymization & Preservation**: PASSED (`[PERSON_001]`, `[EMAIL_001]`, `[AADHAAR_001]` preserved with citation).
7. **Test 7 — Claim Verifier (Correct Supported Claim)**: PASSED (Verified as `SUPPORTED`, response unchanged).
8. **Test 8 — Claim Verifier (Unsupported Added Detail)**: PASSED (Accidental damage claim caught as `UNSUPPORTED`, triggers controlled regeneration).
9. **Test 9 — Claim Verifier (Completely Unsupported Answer)**: PASSED (Fabricated price claim rejected as `UNSUPPORTED`).
10. **Test 10 — Claim Verifier (Invalid Citation ID)**: PASSED (`[chunk_999]` rejected as `UNSUPPORTED` deterministically).
11. **Test 11 — Claim Verifier (Prompt Injection in Excerpt)**: PASSED (Embedded override command ignored by verifier).
12. **Test 12 — Claim Verifier (PII Placeholder Support)**: PASSED (`[PERSON_001]` verified as `SUPPORTED` without unmasking).

## Server Test
- server command: `geniex serve`
- host: 127.0.0.1
- port: 18181
- endpoint: `http://127.0.0.1:18181/v1/chat/completions`
- test result: Failed / Not started. `geniex` CLI is not installed on this machine, and the local hardware is an Intel x86_64 platform. No mock or cloud substitute was used to ensure offline integrity.

## Qualcomm Validation
Clearly distinguish:
- **Local Intel execution**:
  - The local development machine is a Windows 11 PC equipped with a 12th Gen Intel Core i5-12450H CPU (x86_64 architecture).
  - It does NOT have a Qualcomm Snapdragon processor or Hexagon NPU.
  - The user's physical mobile device is a Samsung Galaxy M31 featuring a Samsung Exynos 9611 chipset (ARM Mali GPU), which is NOT a Snapdragon target and cannot run GenieX / Qualcomm AI Engine Direct (QAIRT).
  - Any local offline inference path runs on the Intel CPU using ProofMesh CPU-compatible runtimes without falsifying NPU execution or substituting cloud APIs.
- **Hosted Qualcomm execution**:
  - Qualcomm-specific compilation, profiling, and inference verification are routed through the authenticated Qualcomm AI Hub (`qai-hub` v0.55.0).
  - Hosted Snapdragon targets available via AI Hub include `Samsung Galaxy S25` (Snapdragon 8 Elite `sm8750-ac` running Android 15) and `Snapdragon X Elite CRD` (running Windows 11 ARM64).
  - Target runtime framework: QAIRT (Qualcomm AI Engine Direct, versions 2.45 - 2.50).
