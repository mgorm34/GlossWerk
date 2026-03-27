import zipfile

z = zipfile.ZipFile(r"C:\glosswerk\data\raw\europat\europat.zip")
with z.open("EuroPat.de-en.xml") as f:
    # Read just the first 5000 bytes to see the structure
    head = f.read(5000).decode("utf-8", errors="replace")
    print(head)