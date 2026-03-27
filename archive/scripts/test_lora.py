# Save as test_lora.py
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from peft import PeftModel

base_path = r"C:\glosswerk\models\patent_ape_stageA\final"
lora_path = r"C:\glosswerk\models\a61f_lora"

tokenizer = T5TokenizerFast.from_pretrained(base_path)
base_model = T5ForConditionalGeneration.from_pretrained(base_path)
model = PeftModel.from_pretrained(base_model, lora_path)
model.to("cuda")
model.eval()

# Test sentences from the mitral valve patent
test_cases = [
    "Various brackets are used to achieve secure attachment and support of a heart valve.",
    "This is because, for example, the chordae tendineae that attach to the mitral valve are very important for ventricular function and should therefore also be preserved as far as possible.",
    "After reaching the implantation site, such a stent, which is composed of several self-expanding segments that can be angled relative to each other in its longitudinal direction, can be unfolded.",
    "It is therefore ideal to push the mitral valve aside as far as possible to make room for the new valve if it cannot be reconstructed.",
]

print("BASE MODEL vs LORA ADAPTER\n")
for deepl in test_cases:
    input_text = f"postedit: {deepl}"
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True).input_ids.to("cuda")
    
    # LoRA output
    with torch.no_grad():
        lora_out = model.generate(input_ids=input_ids, max_length=256, num_beams=5)
    lora_text = tokenizer.decode(lora_out[0], skip_special_tokens=True)
    
    # Base model output (disable adapter)
    # Base model output (disable adapter)
    model.disable_adapter_layers()
    with torch.no_grad():
        base_out = model.generate(input_ids=input_ids, max_length=256, num_beams=5)
    model.enable_adapter_layers()
    base_text = tokenizer.decode(base_out[0], skip_special_tokens=True)
    
    changed_base = base_text != deepl
    changed_lora = lora_text != deepl
    
    print(f"DeepL:  {deepl}")
    print(f"Base:   {base_text}" + (" *" if changed_base else ""))
    print(f"LoRA:   {lora_text}" + (" *" if changed_lora else ""))
    print()