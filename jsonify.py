import glob
import json

path = "./data/bbc/*/*.txt"
data = []

files = glob.glob(path)

print("Reading all files in bbc directory...")

for fle in files:
    with open(fle, 'rb') as f:
        lines = [line.rstrip() for line in f.readlines()]
        
        header = lines[0]
        body = b''.join(lines[1:])

        data.append({
            "head": header.decode("utf-8", 'ignore'),
            "body": body.decode("utf-8", 'ignore')
        })
        f.close()

print("Parsing to JSON...")

with open('./data/raw_data.json', 'w') as out:
    json.dump(data, out, ensure_ascii=False, indent=4, sort_keys=True)

print("Done!")