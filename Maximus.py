import os
import tarfile
import urllib.request
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

app = Flask(__name__)

# Load pre-trained model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define the URL and the download path
url = 'https://dax-cdn.cdn.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet.tar.gz'
download_path = './Project_CodeNet.tar.gz'

# Download the Project CodeNet dataset
urllib.request.urlretrieve(url, download_path)

# Uncompress the dataset
with tarfile.open(download_path, 'r:gz') as tar:
    tar.extractall(path='./')

# Load the dataset
dataset = load_dataset('csv', data_files='./Project_CodeNet/...')  # Replace ... with the actual path to the csv or text file containing the dataset

# Preprocessing function
def preprocess_function(examples):
    return tokenizer(examples['code'], truncation=True, padding='max_length')

# Tokenize the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Specify the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train the model
trainer.train()

def generate_text(prompt, max_length=500):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, do_sample=True, temperature=0.7)
    output_text = tokenizer.decode(output[0])
    return output_text

@app.route('/', methods=['GET', 'POST'])
def index():
    response = ''
    if request.method == 'POST':
        data = request.form
        prompt = data['prompt']
        response = generate_text(prompt)
    return render_template('index.html', response=response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
