from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")

# Load the model
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

# def ask(tasktext):


# input_text = "generate: Write a poem about Ireland"
input_text = "question: what is 1+1"
# input_text = "summarize: This is a long article about the history of Rome. Rome was founded in 753 BC by Romulus..."
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate the output
output = model.generate(input_ids, max_length=150, num_return_sequences=1)

# Decode the generated text
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print('INPUT --- ', input_text)
print('OUTPUT --- ', decoded_output)
