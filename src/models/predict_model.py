import tensorflow as tf
from transformers import (
    AutoTokenizer,
    TFAutoModelForCausalLM,
    TFAutoModelForSequenceClassification,
)
from transformers import pipeline


def generate_review(model, prompt):
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
    return generator(prompt, max_length=100, num_return_sequences=1)[0][
        "generated_text"
    ]


def classify_review_sentiment(model, review):
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    classifier = pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer)
    return classifier(review)[0]["label"]


generator_model = TFAutoModelForCausalLM.from_pretrained(
    "alexkell/yelp-review-generator"
)
classifier_model = TFAutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

while True:
    prompt = input("Enter a prompt: ")

    result = generate_review(generator_model, prompt)
    print(result)
    sentiment = classify_review_sentiment(classifier_model, result)
    print(sentiment)
