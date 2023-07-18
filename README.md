# Maximusv1
Maximus allows you to fetch, preprocess, tokenize, and train a GPT-2 model with IBM's project codenet to create a programming oriented generative AI model

# Maximus Version 1 AI Model For Programming Challenge Generation Tasks (For educators, code instructors, etc)

This AI model uses OpenAI's GPT-2 model and is trained on IBM's Project CodeNet dataset to generate programming challenges based on user prompts. It also includes a simple GUI to interact with the model in a user-friendly manner.

## Requirements

- Python 3.7 or higher
- Flask (web framework)
- Transformers library from Hugging Face (for GPT-2 model and tokenizer)
- Datasets library from Hugging Face (for dataset loading)

## Installation

To install the required libraries, use pip:

```bash
pip install flask transformers datasets
```

If you're using a Jupyter notebook, you can prefix each command with an exclamation mark:

```bash
!pip install flask transformers datasets
```

## Dataset

The AI model uses the Project CodeNet dataset from IBM. The dataset is automatically downloaded and uncompressed when the Flask application starts.

The dataset should be in a CSV or similar format and each row of the dataset should contain a column named 'code' which contains the source code examples.

Ensure to specify the correct path to the CSV or text file in the Maximus.py script.

## How to Use

1. Clone the repository or download the script.
2. Run the script using Python.
    ```bash
    python Maximus.py
    ```
3. Open your web browser and navigate to `http://localhost:5000`.
4. You will see a simple form where you can enter a prompt for the Maximus AI model. Press 'Generate Response' to get a response, 'Cancel Response' to clear the input field, or 'Report Response' to report an issue with the response. The 'More Commands' button can be used to show additional functionality.

## Notes

Please note that this script currently trains the model from scratch every time it starts up, which can be a resource-intensive process and may take a significant amount of time, especially on a standard CPU. If you feel froggy and up to it, you can edit the script and could consider training the model separately and loading the trained model when the application starts.

## Disclaimer

The Maximus model generates programming challenges based on the provided prompts, but it doesn't guarantee the correctness or the accuracy of the generated challenges. It is advised to manually review the generated challenges before use.

The model has been trained with IBM's Project CodeNet dataset, ensure to comply with the usage terms of this dataset.

## Credits

- OpenAI

- Adam Rivers https://abtzpro.github.io

- Hello Security LLC https://hellosecurityllc.github.io

- IBM

- HuggingFace

- Tensor

- Flask Maintainers

- Python Maintainers

- And all of you!

- These creations are only possible thanks to you. The fellow user who also enjoys them. And thus allows us to better each creation and development with each passing beta test and software testdrive. It is you the user, that inspires us to produce better software with each release. So thank you dearly. 
