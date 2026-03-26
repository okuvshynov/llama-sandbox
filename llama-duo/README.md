# llama duo - asyncronous/distributed speculative decoding for llama3. 

llama duo is an attempt to make simple linear speculative decoding work in parallel with the main model. It is mostly intended to work in situations when two compute devices are available (e.g. Mac Mini and laptop or GPU and good CPU on the same box) and we share the compute to use the second device to speed up.
Not every hardware/model combination would benefit from such setup. 

Example hardware/potential use-cases:
* Instance with good CPU, large system RAM, and good GPU with small VRAM
* Situations where main model doesn't fit into one device and is split - this was we can amortize the cost of running speculation on 'spare' compute.
* Apple's M1/M2/M3 devices where we'll utilize both CPU and GPU for speculation.

## Dependencies

1. [llama.cpp](https://github.com/ggerganov/llama.cpp)

## Example on CPU+CUDA on same machine.

Machine configuration:
* Intel Xeon Platinum 8358, 30 Cores (probably HT threads, not physical cores)
* ~205GB RAM
* GPU 0: NVIDIA A10 with 24GB VRAM

We'll setup and run duo and compare it to normal llama.cpp run with GPU/CPU offload. 

First, getting the models in gguf format. We use `llama3-70B-Instruct.Q8` as main model and `llama3-8B-Instruct.Q8` as draft model:

```
pip install -U "huggingface_hub[cli]"
mkdir ~/llms && cd ~/llms

# download llama3-70b-q8
huggingface-cli download QuantFactory/Meta-Llama-3-70B-Instruct-GGUF-v2 Meta-Llama-3-70B-Instruct-v2.Q8_0-00001-of-00003.gguf --local-dir .
huggingface-cli download QuantFactory/Meta-Llama-3-70B-Instruct-GGUF-v2 Meta-Llama-3-70B-Instruct-v2.Q8_0-00002-of-00003.gguf --local-dir .
huggingface-cli download QuantFactory/Meta-Llama-3-70B-Instruct-GGUF-v2 Meta-Llama-3-70B-Instruct-v2.Q8_0-00003-of-00003.gguf --local-dir .

# download llama3-8b-q8
huggingface-cli download QuantFactory/Meta-Llama-3-8B-Instruct-GGUF-v2 Meta-Llama-3-8B-Instruct-v2.Q8_0.gguf --local-dir .
```

Get and build llama-duo (this repo):
```
cd ~
git clone https://github.com/okuvshynov/llama-sandbox.git
cd llama-sandbox/llama-duo/
mkdir _build && cd _build
cmake -DLLAMA_RPC=ON -DGGML_CUDA=ON ..
make -j 16
```

Now get llama.cpp and build it with RPC and CUDA support:
```
cd ~
git clone https://github.com/ggerganov/llama.cpp.git
cd ~/llama.cpp
mkdir build-rpc-cuda
cd build-rpc-cuda
cmake .. -DGGML_CUDA=ON -DGGML_RPC=ON
make -j 16
```

Now we can try using llama.cpp. Note that we cannot load entire 70B model to GPU vram, so the model must be shared between main memory and GPU.

We'll use following prompt:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful, respectful and honest assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

Please illustrate the difference between concurrency and parallelism in python<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

which is located at ```test_prompt.txt```

Run on CPU entirely:
```
cd ~/llama.cpp/
time ./build-rpc-cuda/bin/llama-cli -m ../llms/Meta-Llama-3-70B-Instruct-v2.Q8_0-00001-of-00003.gguf -f ../llama-sandbox/llama-duo/test_prompt.txt -n 512 -ngl 0
...
llama_print_timings:        load time =    3966.50 ms
llama_print_timings:      sample time =      44.57 ms /   512 runs   (    0.09 ms per token, 11487.81 tokens per second)
llama_print_timings: prompt eval time =    6596.12 ms /    36 tokens (  183.23 ms per token,     5.46 tokens per second)
llama_print_timings:        eval time =  250417.02 ms /   511 runs   (  490.05 ms per token,     2.04 tokens per second)
llama_print_timings:       total time =  257221.29 ms /   547 tokens
Log end

real	4m22.915s
user	125m48.069s
sys	0m5.870s

```


We can offload part of the model to GPU, with 22 being the largest number of layers I could fit for this hardware/model/prompt combination

```
cd ~/llama.cpp/
time ./build-rpc-cuda/bin/llama-cli -m ../llms/Meta-Llama-3-70B-Instruct-v2.Q8_0-00001-of-00003.gguf -f ../llama-sandbox/llama-duo/test_prompt.txt -n 512 -ngl 22
...
llama_print_timings:        load time =    5650.47 ms
llama_print_timings:      sample time =      40.73 ms /   512 runs   (    0.08 ms per token, 12571.82 tokens per second)
llama_print_timings: prompt eval time =    4932.85 ms /    36 tokens (  137.02 ms per token,     7.30 tokens per second)
llama_print_timings:        eval time =  202207.16 ms /   511 runs   (  395.71 ms per token,     2.53 tokens per second)
llama_print_timings:       total time =  207313.90 ms /   547 tokens
Log end

real	3m34.382s
user	92m7.193s
sys	0m5.575s
```

We get some improvement, from 2.05 -> 2.53 tps on generation and ~50 seconds absolute time.

We can also monitor resource usage with cubestat: `cubestat -i 100`

<img width="834" alt="Screenshot 2024-07-10 at 12 21 55 PM" src="https://github.com/okuvshynov/llama_duo/assets/661042/48cdc3ab-cdae-4756-9b60-094c13dafbb6">

and see that GPU is heavily underutilizing the compute as it has to wait for CPU to complete its part.

Now let's do speculation. We'll do speculation on GPU and main model on CPU, which sounds counterintuitive, but:
* this way we get to quickly compute speculation model
* speculation model will fit entirely into GPU's VRAM
* CPU execution ports while not as wide as GPU, still benefit from batches of size > 1, implementation would use SIMD (AVX, etc.)

From llama.cpp build, start rpc server in a separate terminal window:
```
cd ~/llama.cpp/build-rpc-cuda
bin/rpc-server -p 20002
```

Now we can start duo with the same prompt. Main model runs on CPU and speculation - on GPU, see ngl anad ngld options. Note that we configure rpc server to be used for draft model only, not for main model [here](https://github.com/okuvshynov/llama-sandbox/blob/main/llama-duo/duo.cpp#L305-L307)
```
cd ~/llama-sandbox/llama-duo/
time ./_build/duo -m ../llms/Meta-Llama-3-70B-Instruct-v2.Q8_0-00001-of-00003.gguf -md ../llms/Meta-Llama-3-8B-Instruct-v2.Q8_0.gguf -f ./test_prompt.txt -n 512 --draft 4 -ngl 0 -ngld 99 --rpc localhost:20002
...

tokens: 512 tps: 3.78504

real	2m33.495s
user	67m34.029s
sys	0m8.442s
```

We got another minute off!

Now, we can check resource utilization as well.

<img width="840" alt="Screenshot 2024-07-10 at 12 30 46 PM" src="https://github.com/okuvshynov/llama_duo/assets/661042/ef80c83b-274a-4812-923a-5f5fec68cf45">

As we can see, gpu is still utilized not too well, and on top of that we have spare VRAM. So while we have speculation running on GPU we can also offload part of the main model to GPU (11-12 layers is likely the limit):

```
cd ~/llama-sandbox/llama-duo/
time ./_build/duo -m ../llms/Meta-Llama-3-70B-Instruct-v2.Q8_0-00001-of-00003.gguf -md ../llms/Meta-Llama-3-8B-Instruct-v2.Q8_0.gguf -f ./test_prompt.txt -n 512 --draft 4 -ngl 11 -ngld 99 --rpc localhost:20002
...
tokens: 512 tps: 4.47763

real	2m13.046s
user	51m40.820s
sys	0m8.915s
```

And 20 more seconds of overall time.

In this particular example both speculation and main model are running on the same host, we don't actually have to use rpc at all - we can just start duo like this and get similar results: 

```
cd ~/llama-sandbox/llama-duo/
time ./_build/duo -m ../llms/Meta-Llama-3-70B-Instruct-v2.Q8_0-00001-of-00003.gguf -md ../llms/Meta-Llama-3-8B-Instruct-v2.Q8_0.gguf -f ./test_prompt.txt -n 512 --draft 4 -ngl 11 -ngld 99
...
tokens: 514 tps: 4.43318

real	2m10.058s
user	52m53.796s
sys	0m5.889s
```

Some notes:
* there's nothing smart done about scheduling GPU work when both speculation and part of main model are running there.
* settings are very likely suboptimal - for example, it's possible we could use more aggresively quantized speculation model and keep more main model layers on GPU.
* sampling is just greedy and needs to be done right.

