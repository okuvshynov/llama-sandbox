# CPU run on prompt processing (batch size)
$HOME/projects/llama.cpp/build/bin/llama-bench -m \
    $HOME/projects/llms/gemma4/gemma-4-E2B-it-UD-Q2_K_XL.gguf,$HOME/projects/llms/gemma4/gemma-4-E2B-it-UD-Q3_K_XL.gguf,$HOME/projects/llms/gemma4/gemma-4-E2B-it-UD-Q4_K_XL.gguf,$HOME/projects/llms/gemma4/gemma-4-E2B-it-UD-Q5_K_XL.gguf,$HOME/projects/llms/gemma4/gemma-4-E2B-it-UD-Q6_K_XL.gguf,$HOME/projects/llms/gemma4/gemma-4-E2B-it-UD-Q8_K_XL.gguf,$HOME/projects/llms/gemma4/gemma-4-E2B-it-BF16.gguf \
-n 0 -d 0 -p 2,4,8,16,32,64,128,256,512,1024 -ngl 0 -o csv
