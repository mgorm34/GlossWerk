from comet import download_model, load_from_checkpoint
from docx import Document as DocxDocument
import re, json

doc = DocxDocument(r"C:\glosswerk\test_patent_glosswerk.docx")

segments = []
current = {}
for p in doc.paragraphs:
    text = p.text.strip()
    if not text:
        continue
    m = re.match(r'\[(\d+)\]\s+(PUBLISH|REVIEW|FULL_EDIT)\s+\(QE:\s+([\d.]+)\)', text)
    if m:
        if current.get('id'):
            segments.append(current)
        current = {'id': int(m.group(1)), 'triage': m.group(2), 'qe_score': float(m.group(3)),
                   'ape_corrected': 'APE corrected' in text, 'de': '', 'en': '', 'deepl': ''}
        continue
    if text.startswith('DE:') and current.get('id'):
        current['de'] = text[3:].strip()
    elif text.startswith('DeepL original:') and current.get('id'):
        current['deepl'] = text[len('DeepL original:'):].strip()
    elif text.startswith('EN:') and current.get('id'):
        current['en'] = text[3:].strip()
if current.get('id'):
    segments.append(current)

data = []
for seg in segments:
    mt = seg['deepl'] if seg['deepl'] else seg['en']
    if seg['de'] and mt:
        data.append({'src': seg['de'], 'mt': mt})

print(f"Parsed {len(segments)} segments, running XCOMET-XL on {len(data)}...\n")

model_path = download_model("Unbabel/XCOMET-XL")
model = load_from_checkpoint(model_path)
model.eval()

output = model.predict(data, batch_size=4, gpus=1)

# Access error_spans correctly from metadata
all_errors = output.metadata.error_spans

print("="*80)
print("XCOMET-XL WORD-LEVEL QE RESULTS")
print("="*80)

total_errors = 0
segments_with_errors = 0

for i, seg in enumerate(segments):
    if i >= len(output.scores):
        break
    score = output.scores[i]
    errors = all_errors[i] if i < len(all_errors) else []
    
    if errors:
        segments_with_errors += 1
        total_errors += len(errors)
    
    print(f"\n[{seg['id']}] {seg['triage']} | XCOMET: {score:.4f} | Old QE: {seg['qe_score']:.3f} | Errors: {len(errors)}")
    for err in errors:
        print(f"     -> '{err['text']}' | severity: {err['severity']} | conf: {err['confidence']:.3f}")

print(f"\n{'='*80}")
print(f"SUMMARY")
print(f"  Total segments: {len(segments)}")
print(f"  Segments with errors: {segments_with_errors} ({100*segments_with_errors/len(segments):.0f}%)")
print(f"  Total error spans: {total_errors}")
print(f"  Clean segments: {len(segments)-segments_with_errors} ({100*(len(segments)-segments_with_errors)/len(segments):.0f}%)")
print(f"  Avg XCOMET score: {sum(output.scores)/len(output.scores):.4f}")

results = []
for i, seg in enumerate(segments):
    if i >= len(output.scores):
        break
    errors = all_errors[i] if i < len(all_errors) else []
    results.append({
        'id': seg['id'], 'triage': seg['triage'], 'old_qe': seg['qe_score'],
        'xcomet_score': output.scores[i], 'de': seg['de'],
        'deepl': seg['deepl'] if seg['deepl'] else seg['en'],
        'ape': seg['en'], 'error_spans': errors
    })

with open(r"C:\glosswerk\data\xcomet_patent.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to C:\\glosswerk\\data\\xcomet_patent.json")