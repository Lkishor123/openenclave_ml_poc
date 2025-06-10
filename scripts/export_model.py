import os
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

# 1. Define the model ID from Hugging Face and the local save directory
model_id = "distilbert-base-uncased-finetuned-sst-2-english"
save_directory = "distilbert-sst2-onnx"

# Create the save directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)
print(f"ONNX model will be saved to: {os.path.abspath(save_directory)}")

# 2. Download the model from the Hub and export it to ONNX format
# This single command handles the conversion and saves the model.onnx file.
model = ORTModelForSequenceClassification.from_pretrained(
    model_id,
    export=True # The 'export' flag triggers the conversion
)

# 3. Download the correct tokenizer for this model
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 4. Save the ONNX model and the tokenizer to the same directory
# This is the best practice for keeping everything together.
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print("\n--- Model and tokenizer saved successfully! ---")
print(f"Files in '{save_directory}': {os.listdir(save_directory)}")

# 5. (Optional) Example of how to use the exported ONNX model
print("\n--- Running a test inference ---")
text = "This movie is absolutely fantastic!"
inputs = tokenizer(text, return_tensors="pt")

# Perform inference with the ONNX model
outputs = model(**inputs)
logits = outputs.logits.detach().numpy()[0]
predicted_class_id = logits.argmax().item()

# The model's config contains the label mappings
label = model.config.id2label[predicted_class_id]

print(f"Text: '{text}'")
print(f"Predicted Label: {label}")
print(f"Logits (Scores): {logits}")