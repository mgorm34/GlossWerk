import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast

# Load your APE model
model_path = r"C:\glosswerk\models\patent_ape_stageA\final"
tokenizer = T5TokenizerFast.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)
model.eval()
model.to("cuda")

# The problematic segment
deepl = "Various brackets are used to achieve secure attachment and support of a heart valve."
source = "Verschiedene Bügel werden hierbei eingesetzt, um eine sichere Befestigung und Abstützung einer Herzklappe zu erreichen."

# XCOMET found only "bracket" as an error (chars 7-15)
# So everything EXCEPT "bracket" is OK and should be preserved
# OK words: "Various", "s", "are", "used", "to", "achieve", "secure", 
#           "attachment", "and", "support", "of", "a", "heart", "valve."

# Extract OK words by removing error span text
error_text = "bracket"  # from XCOMET
ok_words = []
for word in deepl.split():
    clean = word.strip(".,;:()")
    if clean.lower() != error_text.lower() and error_text not in clean:
        ok_words.append(word)

print("OK words to force:", ok_words)

# Tokenize OK words as constraints
force_words_ids = []
for word in ok_words:
    ids = tokenizer(word, add_special_tokens=False).input_ids
    if ids:
        force_words_ids.append(ids)

# Standard APE input
input_text = f"postedit: {deepl}"
input_ids = tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True).input_ids.to("cuda")

# Generate WITHOUT constraints (current behavior)
output_unconstrained = model.generate(input_ids, max_length=256, num_beams=5)
text_unconstrained = tokenizer.decode(output_unconstrained[0], skip_special_tokens=True)

# Generate WITH constraints
output_constrained = model.generate(
    input_ids, 
    max_length=256, 
    num_beams=10,  # need more beams for constrained search
    force_words_ids=force_words_ids,
    no_repeat_ngram_size=0,
)
text_constrained = tokenizer.decode(output_constrained[0], skip_special_tokens=True)

print(f"\nDeepL original:  {deepl}")
print(f"APE unconstrained: {text_unconstrained}")
print(f"APE constrained:   {text_constrained}")