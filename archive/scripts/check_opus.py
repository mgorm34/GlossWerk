# Save as check_opus.py
import urllib.request
url = "https://opus.nlpl.eu/EuroPat/v3/xml/de-en.xml.gz"
req = urllib.request.Request(url, method='HEAD')
try:
    resp = urllib.request.urlopen(req)
    size = resp.headers.get('Content-Length', 'unknown')
    print(f"DE-EN XML: {int(size)/1e9:.1f} GB" if size != 'unknown' else f"Size: {size}")
except Exception as e:
    # Try the OPUS API instead
    print(f"Error: {e}")
    print("\nTrying OPUS index...")
    url2 = "https://opus.nlpl.eu/EuroPat-v3.php"
    resp = urllib.request.urlopen(url2)
    html = resp.read().decode()[:3000]
    print(html)