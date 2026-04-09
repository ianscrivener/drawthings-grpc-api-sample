# Draw Things gRPC Protocol

## Service Definition

**Proto:** `draw-things-community/Libraries/GRPC/Models/Sources/imageService/imageService.proto`

```protobuf
service ImageGenerationService {
  rpc GenerateImage(ImageGenerationRequest) returns (stream ImageGenerationResponse);
  rpc FilesExist(FileListRequest) returns (FileExistenceResponse);
  rpc UploadFile(stream FileUploadRequest) returns (stream UploadResponse);
  rpc Echo(EchoRequest) returns (EchoReply);
  rpc Pubkey(PubkeyRequest) returns (PubkeyResponse);
  rpc Hours(HoursRequest) returns (HoursResponse);
}
```

## Methods

| Method | Type | Description |
|--------|------|-------------|
| `GenerateImage` | **Server Streaming** | Main image generation with progress updates |
| `FilesExist` | Unary | Check if model files exist on server |
| `UploadFile` | Bidirectional Streaming | Upload model files in chunks |
| `Echo` | Unary | Health check + available model discovery |
| `Pubkey` | Unary | Returns public key |
| `Hours` | Unary | Compute unit thresholds |

## Message Structures

### Echo (Unary)

**Request:**
```protobuf
message EchoRequest {
  string name = 1;
  optional string sharedSecret = 2;
}
```

**Response:**
```protobuf
message EchoReply {
  string message = 1;
  repeated string files = 2;           // Available .ckpt model files
  optional MetadataOverride override = 3;
  bool sharedSecretMissing = 4;
  optional ComputeUnitThreshold thresholds = 5;
  uint64 serverIdentifier = 6;        // Distinguishes remote vs local server
}
```

### FilesExist (Unary)

**Request:**
```protobuf
message FileListRequest {
  repeated string files = 1;
  repeated string filesWithHash = 2;   // Files to check AND compute SHA256
  optional string sharedSecret = 3;
}
```

**Response:**
```protobuf
message FileExistenceResponse {
  repeated string files = 1;
  repeated bool existences = 2;
  repeated bytes hashes = 3;          // SHA256 hashes if requested
}
```

### GenerateImage (Server Streaming)

**Request:**
```protobuf
message ImageGenerationRequest {
  optional bytes image = 1;            // Input image (img2img) as bytes with 68-byte header
  int32 scaleFactor = 2;
  optional bytes mask = 3;            // Mask data (inpaint)
  repeated HintProto hints = 4;       // ControlNet hints
  string prompt = 5;
  string negativePrompt = 6;
  bytes configuration = 7;            // FlatBuffer-encoded GenerationConfiguration
  MetadataOverride override = 8;      // Model overrides
  repeated string keywords = 9;
  string user = 10;
  DeviceType device = 11;             // PHONE, TABLET, LAPTOP
  repeated bytes contents = 12;        // Content-addressable storage (SHA256 -> data)
  optional string sharedSecret = 13;
  bool chunked = 14;                  // Client can accept chunked responses (>4MB)
}
```

**Response:**
```protobuf
message ImageGenerationResponse {
  repeated bytes generatedImages = 1;        // Final image data
  optional ImageGenerationSignpostProto currentSignpost = 2;
  repeated ImageGenerationSignpostProto signposts = 3;
  optional bytes previewImage = 4;          // Preview image (fpzip compressed)
  optional int32 scaleFactor = 5;
  repeated string tags = 6;
  optional int64 downloadSize = 7;          // Announced total size before sending
  ChunkState chunkState = 8;               // LAST_CHUNK or MORE_CHUNKS
  optional RemoteDownloadResponse remoteDownload = 9;
  repeated bytes generatedAudio = 10;
}
```

### UploadFile (Bidirectional Streaming)

**Request:**
```protobuf
message FileUploadRequest {
  oneof request {
    InitUploadRequest initRequest = 1;
    FileChunk chunk = 2;
  }
  optional string sharedSecret = 3;
}

message InitUploadRequest {
  string filename = 1;
  bytes sha256 = 2;
  int64 totalSize = 3;
}

message FileChunk {
  bytes content = 1;
  string filename = 2;
  int64 offset = 3;
}
```

**Response:**
```protobuf
message UploadResponse {
  bool chunkUploadSuccess = 1;
  int64 receivedOffset = 2;
  string message = 3;
  string filename = 4;
}
```

## FlatBuffers Configuration Schema

**Schema:** `draw-things-community/Libraries/DataModels/Sources/config.fbs`

### Core GenerationConfiguration Table

```flatbuffers
table GenerationConfiguration {
  id: long;
  start_width: ushort;
  start_height: ushort;
  seed: uint;
  steps: uint;
  guidance_scale: float;
  strength: float;                    // Denoising strength (img2img)
  model: string;                      // Model filename
  sampler: SamplerType = DPMPP2MKarras;
  batch_count: uint = 1;
  batch_size: uint = 1;
  hires_fix: bool = false;
  hires_fix_start_width: ushort;
  hires_fix_start_height: ushort;
  hires_fix_strength: float = 0.7;
  upscaler: string;
  controls: [Control];                // ControlNet entries
  loras: [LoRA];                     // LoRA adjustments
}
```

### SamplerType Enum

```flatbuffers
enum SamplerType: byte {
  DPMPP2MKarras, EulerA, DDIM, PLMS, DPMPPSDEKarras, UniPC, LCM,
  EulerASubstep, DPMPPSDESubstep, TCD, EulerATrailing, ...
}
```

### ControlInputType Enum

```flatbuffers
enum ControlInputType: byte {
  Unspecified, Custom, Depth, Canny, Scribble, Pose, Normalbae,
  Color, Lineart, Softedge, Seg, Inpaint, Ip2p, Shuffle, ...
}
```

### LoRAMode Enum

```flatbuffers
enum LoRAMode: byte { All, Base, Refiner }
```

## Image Data Format

Images use a custom 68-byte header + Float16 tensor (NHWC format):

```
Offset 0-3:    Magic (0)
Offset 4-7:    CCV_TENSOR_CPU_MEMORY
Offset 8-11:   CCV_TENSOR_FORMAT_NHWC
Offset 12-15:  CCV_16F (Float16)
Offset 16-19:  Flags (0)
Offset 20-23:  Batch (1)
Offset 24-27:  Height
Offset 28-31:  Width
Offset 32-35:  Channels
Offset 36-67:  Reserved/Padding
Offset 68+:    Float16 pixel data (NHWC layout)
```

## Streaming Behavior

### Progress Signposts

During generation, the server streams `ImageGenerationSignpostProto` messages:

```protobuf
oneof signpost {
  TextEncoded textEncoded = 1;
  ImageEncoded imageEncoded = 2;
  Sampling sampling = 3;              // { step: int32 }
  ImageDecoded imageDecoded = 4;
  SecondPassImageEncoded = 5;
  SecondPassSampling = 6;              // { step: int32 }
  SecondPassImageDecoded = 7;
  FaceRestored = 8;
  ImageUpscaled = 9;
}
```

Progress flow: `textEncoded` → `sampling` → `imageDecoded` → (optional `secondPass*`) → `imageEncoded`

### Chunking

- If `request.chunked = true` AND total size > 4MB
- Server sends image data in 4MB chunks
- Each chunk has `chunkState = MORE_CHUNKS` except the last (`LAST_CHUNK`)
- Client reassembles chunks sequentially

## Server CLI

```bash
gRPCServerCLI <models_path> [options]
  --port 7859              # Default port
  --sharedSecret <secret>  # Auth secret
  --noTLS                  # Disable TLS
  --noResponseCompression   # Disable fpzip compression
  --modelBrowser           # Enable model browsing via Echo
  --join                   # Join proxy server cluster
```

## Client Implementation Reference

See `draw-things-comfyui/src/draw_things.py` for Python async gRPC client implementation using `grpc.aio`.
