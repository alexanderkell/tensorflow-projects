import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM
from transformers import pipeline

def generate_review(model, prompt):
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
    return generator(prompt, max_length=100, num_return_sequences=1)[0]["generated_text"]



model = TFAutoModelForCausalLM.from_pretrained("alexkell/yelp-review-generator")

while True:
    prompt = input("Enter a prompt: ")

    result = generate_review(model, prompt)
    print(result)