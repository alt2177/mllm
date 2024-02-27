from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")

# Load the model
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")

# def ask(tasktext):


# input_text = "generate: Write a poem about Ireland"
input_text = "question: what is 1+1"
# input_text = "summarize: This is a long article about the history of Rome. Rome was founded in 753 BC by Romulus and then Ceasar died and they had Aqueducts and they crossed Rubicon"
# input_text = "translate this to French: I want to learn english. What is the best way to do that?"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate the output
output = model.generate(input_ids, max_length=150, num_return_sequences=1)
print('RAW OUTPUT --- ', output)

# Decode the generated text
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print('INPUT --- ', input_text)
print('OUTPUT --- ', decoded_output)
