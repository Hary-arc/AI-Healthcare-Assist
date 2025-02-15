# File: train_healthcare.py
from transformers import Trainer, TrainingArguments
from configuration_deepseek import DeepseekV3Config
from datasets import load_dataset
import torch
import random
import json
import pickle
import numpy as np
import pandas as pd
import nltk
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from tf.keras.models import Sequential
from tf.keras.layers import Dense, Dropout
from tf.keras.optimizers import SGD


# 1. Data Preparation
def preprocess_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-1.3b")
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=4096,  # Use model's max context length
        padding="max_length",
    )

dataset = load_dataset(
    "text",
    data_dir="healthCareFacts",
    sample_by="paragraph"  # Adjust based on your data structure
).map(
    preprocess_function,
    batched=True,
    remove_columns=["text"]
)

# 2. Model Configuration
config = DeepseekV3Config(
    vocab_size=32000,
    hidden_size=4096,
    num_hidden_layers=32,
    num_attention_heads=32,
    n_routed_experts=64,  # For MoE configuration
    num_experts_per_tok=4,
    moe_intermediate_size=16384,
    _attn_implementation="flash_attention_2",
    rope_scaling={"type": "yarn", "factor": 8.0}
)

model = DeepseekV3ForCausalLM(config)

# 3. Training Setup
training_args = TrainingArguments(
    output_dir="./healthcare_model",
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Adjust based on GPU memory
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    fp16=True,
    optim="adamw_bnb_8bit",
    logging_dir="./logs",
    report_to="tensorboard",
    save_strategy="steps",
    save_steps=1000,
    deepspeed="./configs/zero3.json",  # For multi-GPU training
    gradient_checkpointing=True
)

# 4. Custom Data Collator
class HealthcareCollator:
    def __call__(self, features):
        return {
            "input_ids": torch.stack([torch.tensor(f["input_ids"]) for f in features]),
            "labels": torch.stack([torch.tensor(f["input_ids"]) for f in features])
        }

# 5. Start Training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=HealthcareCollator()
)

trainer.train()

# Load the CSV data
data = pd.read_csv('HealthCareFacts/HealthCareFacts.csv')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Prepare intents
intents = {"intents": []}

for index, row in data.iterrows():
    intents["intents"].append({
        "tag": row['tag'],
        "patterns": [p.strip() for p in row['patterns'].split(',')],  # Clean whitespace
        "responses": [row['responses']]
    })

# Save intents to JSON
with open('intents.json', 'w') as json_file:
    json.dump(intents, json_file)

# Prepare training data
words = []
classes = []
documents = []

ignore_letters = ["?", "!", ".", ","]

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Prepare training data
training = []
template = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(template)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Build the model
model = Sequential()
model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save the model
model.save("chatbot_model.h5")
print("Model trained and saved!")

print(tf.__version__)