import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

with open("spear_dummy.trt", "rb") as f:
    engine_data = f.read()

runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(engine_data)

print("Engine loaded:", engine is not None)
print("Num I/O tensors:", engine.num_io_tensors)

for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    mode = engine.get_tensor_mode(name)
    print(f"Tensor {i}: name={name}, mode={mode}")
