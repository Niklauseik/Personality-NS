"""Create the model integrity manifest required by the training runners."""
import argparse, hashlib, json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument("--model-path",type=Path,required=True);a=p.parse_args()
    files={}
    for f in sorted(a.model_path.iterdir()):
        if f.is_file() and f.name != "SHA256SUMS.json":
            h=hashlib.sha256()
            with f.open("rb") as stream:
                for chunk in iter(lambda:stream.read(8*1024*1024),b""):h.update(chunk)
            files[f.name]=h.hexdigest()
    (a.model_path/"SHA256SUMS.json").write_text(json.dumps(files,indent=2)+"\n")
if __name__ == "__main__":main()
