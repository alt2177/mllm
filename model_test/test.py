from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Check if a GPU is available and if not, use a CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

# Move the model to the GPU
model = model.to(device)

input_text = "translate this to French: This is not a pipe"

# Encode the input text
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Create the attention mask
attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

# Move the tensors to the GPU
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)

# Generate the output
output = model.generate(input_ids, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id, max_length=150, num_return_sequences=1)

# Decode the generated text
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print('INPUT --- ', input_text)
print('OUTPUT --- ', decoded_output)