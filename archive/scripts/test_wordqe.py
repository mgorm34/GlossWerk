from comet import download_model, load_from_checkpoint

model_path = download_model("Unbabel/XCOMET-XL")
model = load_from_checkpoint(model_path)
model.eval()

data = [
    {
        "src": "Verschiedene Bügel werden hierbei eingesetzt, um eine sichere Befestigung und Abstützung einer Herzklappe zu erreichen.",
        "mt": "Various brackets are used to achieve secure attachment and support of a heart valve."
    }
]

output = model.predict(data, batch_size=1, gpus=1)
print("Score:", output.scores)
print("\nAll keys:", list(output.keys()))
if 'metadata' in output:
    print("\nMetadata:", output['metadata'])