import requests
import trafilatura
from supabase import create_client
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[3]
load_dotenv(ROOT_DIR / ".env")

# Isi dengan credential Supabase-mu
SUPABASE_URL = os.environ.get("SUPABASE_URL", "URL_KAMU")
SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "KEY_KAMU")

sb = create_client(SUPABASE_URL, SERVICE_KEY)

# Ambil 5 sample artikel GDELT acak
res = sb.table("raw_texts") \
        .select("title, source_url, published_at") \
        .like("source", "gdelt_%") \
        .limit(5).execute()

print("=== UJI KUALITAS URL & EXTRACTION ===\n")
for item in res.data:
    url = item['source_url']
    title = item['title']
    print(f"Title : {title}")
    print(f"URL   : {url}")
    print(f"Tahun : {item['published_at']}")
    
    try:
        # Coba fetch URL seperti yang akan dilakukan NLP Worker
        downloaded = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if downloaded.status_code == 200:
            text = trafilatura.extract(downloaded.text)
            if text:
                print(f"✅ BERHASIL DIEKSTRAK | Panjang teks: {len(text)} karakter")
            else:
                print("⚠️ URL 200 OK, tapi trafilatura gagal extract (mungkin halaman non-artikel)")
        else:
            print(f"❌ URL MATI / DITOLAK (HTTP {downloaded.status_code})")
    except Exception as e:
        print(f"❌ GAGAL FETCH: {e}")
        
    print("-" * 60)