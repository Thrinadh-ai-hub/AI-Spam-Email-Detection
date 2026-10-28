
from pathlib import Path
from urllib.request import urlretrieve
import tarfile
from email import policy
from email.parser import BytesParser
import pandas as pd
import hashlib

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
EXTRACTED = ROOT / "data" / "extracted"

RAW.mkdir(parents=True, exist_ok=True)
EXTRACTED.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://spamassassin.apache.org/old/publiccorpus/"

ARCHIVES = {
    "20030228_easy_ham.tar.bz2": "ham",
    "20030228_easy_ham_2.tar.bz2": "ham",
    "20030228_hard_ham.tar.bz2": "ham",
    "20030228_spam.tar.bz2": "spam",
    "20030228_spam_2.tar.bz2": "spam",
}

def get_email_text(msg):
    subject = str(msg.get("subject", "") or "")
    parts = []

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_maintype() == "multipart":
                continue
            if part.get_content_disposition() == "attachment":
                continue
            if part.get_content_type() == "text/plain":
                try:
                    content = part.get_content()
                    if isinstance(content, str):
                        parts.append(content)
                except Exception:
                    pass
    elif msg.get_content_type() == "text/plain":
        try:
            parts.append(msg.get_content())
        except Exception:
            pass

    return f"Subject: {subject}\n\n" + "\n".join(parts)

def main():
    records = []
    seen = set()

    for archive, label in ARCHIVES.items():
        archive_path = RAW / archive
        folder = EXTRACTED / archive.replace(".tar.bz2", "")
        folder.mkdir(parents=True, exist_ok=True)

        if not archive_path.exists():
            print("Downloading:", archive)
            urlretrieve(BASE_URL + archive, archive_path)

        if not any(folder.iterdir()):
            print("Extracting:", archive)
            with tarfile.open(archive_path, "r:bz2") as tar:
                tar.extractall(folder, filter="data")

        files = [p for p in folder.rglob("*") if p.is_file()]
        print("Reading:", archive, "Files:", len(files))

        for path in files:
            try:
                raw = path.read_bytes()
                msg = BytesParser(policy=policy.default).parsebytes(raw)
                text = get_email_text(msg).strip()

                if len(text) < 15:
                    continue

                digest = hashlib.sha256(
                    text.encode("utf-8", errors="replace")
                ).hexdigest()

                if digest in seen:
                    continue
                seen.add(digest)

                records.append({
                    "text": text,
                    "label": label
                })
            except Exception as error:
                print("Skipped:", path.name, type(error).__name__)

    df = pd.DataFrame(records)
    output = ROOT / "data" / "emails.csv"
    df.to_csv(output, index=False, encoding="utf-8")

    print("\nDataset created!")
    print("Saved at:", output)
    print("Total messages:", len(df))
    print("\nClass counts:")
    print(df["label"].value_counts())

if __name__ == "__main__":
    main()
