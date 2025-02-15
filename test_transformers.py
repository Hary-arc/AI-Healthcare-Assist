from transformers import __version__, pipeline

print(f"Transformers version: {__version__}")
test_pipe = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
print(test_pipe("This movie is great!")) 