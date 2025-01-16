# Qwen2.5-Coder-32B-4bit-locally
run Qwen2.5-Coder-32B-Instruct-GPTQ-Int4 locally with 16gb vram


########################
cuda_11.8.0_522.06_windows

git clone https://github.com/PanQiWei/AutoGPTQ.git && cd AutoGPTQ
pip install .


pip install threadpoolctl==3.1.0
pip install auto-gptq[cuda]
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install optimum



#########################


Microsoft Windows [Version 10.0.19045.5371]
(c) Microsoft Corporation. All rights reserved.

C:\Users\m>cd C:\Users\m\Desktop\1 && a\Scripts\activate

(a) C:\Users\m\Desktop\1>cd C:\Users\m\Desktop\1\Qwen2.5-Coder-32B-Instruct-GPTQ-Int4

(a) C:\Users\m\Desktop\1\Qwen2.5-Coder-32B-Instruct-GPTQ-Int4>python
Python 3.11.0 (main, Oct 24 2022, 18:26:48) [MSC v.1933 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> import torch
>>> # مسار مجلد النموذج المحلي
>>> model_path = "C:/Users/m/Desktop/1/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4" # استبدل هذا بالمسار الفعلي لمجلد النموذج
>>>
>>> model = AutoModelForCausalLM.from_pretrained(
...     model_path,  # قم بتغيير model_name إلى model_path
...     torch_dtype="auto",
...     device_map="auto",
... )
CUDA extension not installed.
CUDA extension not installed.
`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
C:\Users\m\Desktop\1\a\Lib\site-packages\accelerate\utils\modeling.py:1593: UserWarning: Current model requires 5071298688 bytes of buffer for offloaded layers, which seems does not fit any GPU's remaining memory. If you are experiencing a OOM later, please consider using offload_buffers=True.
  warnings.warn(
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████| 5/5 [00:44<00:00,  8.83s/it]
Some parameters are on the meta device because they were offloaded to the cpu.
>>> tokenizer = AutoTokenizer.from_pretrained(
...     model_path,  # قم بتغيير model_name إلى model_path
... )
>>>
>>> prompt = "write a quick sort algorithm."
>>> messages = [
...     {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
...     {"role": "user", "content": prompt}
... ]
>>> text = tokenizer.apply_chat_template(
...     messages,
...     tokenize=False,
...     add_generation_prompt=True
... )
>>> model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
>>>
>>> generated_ids = model.generate(
...     **model_inputs,
...     max_new_tokens=50
... )
>>> generated_ids = [
...     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
... ]
>>>
>>> response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>>
>>>
>>> print(response)
Certainly! Quick Sort is a popular and efficient comparison-based, divide-and-conquer, in-place sort algorithm. Below is a Python implementation of the Quick Sort algorithm:

```python
def quick_sort(arr):
    # Base case: if the array is
>>>



############################

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "C:/Users/m/Desktop/1/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4"

model = AutoModelForCausalLM.from_pretrained(
    model_path,  
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,  
)

prompt = "write a quick sort algorithm."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


print(response)















