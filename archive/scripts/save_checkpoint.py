"""Save checkpoint-17000 as the final v2 model."""
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer

checkpoint = r"C:\glosswerk\models\patent_ape_stageA\checkpoint-17000"
output = r"C:\glosswerk\models\patent_ape_stageA_v2\final"

if not os.path.exists(checkpoint):
    # Find the latest checkpoint
    import glob
    checkpoints = sorted(glob.glob(r"C:\glosswerk\models\patent_ape_stageA\checkpoint-*"))
    if checkpoints:
        checkpoint = checkpoints[-1]
        print(f"Using latest checkpoint: {checkpoint}")
    else:
        print("ERROR: No checkpoints found!")
        exit(1)

print(f"Loading model from: {checkpoint}")
model = T5ForConditionalGeneration.from_pretrained(checkpoint)
tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")

os.makedirs(output, exist_ok=True)
model.save_pretrained(output)
tokenizer.save_pretrained(output)
print(f"Saved v2 model to: {output}")
