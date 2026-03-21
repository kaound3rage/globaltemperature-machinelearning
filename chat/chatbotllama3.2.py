"""
====================================================
GlobalTemp - Hybrid AI Chatbot
====================================================
Model  : llama3.2 (lebih baik dari tinyllama)
API    : GlobalTemp Forecast API

Cara pakai:
  pip install ollama requests
  ollama pull llama3.2
  python chatbot.py
====================================================
"""

import requests
import ollama
import sys

# ─── Config ───────────────────────────────────────────────────────────────────
API_BASE    = "https://2ekahxvxmb.execute-api.us-east-1.amazonaws.com/prod"
API_KEY     = "K7esYYbTSv3fX592ZkHMc2C3GGIr1BXG9Sla4Fld"
MODEL_LLM   = "llama3.2"
TIMEOUT_SEC = 15

HEADERS = {
    "Content-Type": "application/json",
    "x-api-key"   : API_KEY,
}

# ─── Keyword detector ─────────────────────────────────────────────────────────
KEYWORDS_PREDIKSI = [
    "prediksi", "forecast", "suhu", "temperatur", "temperature",
    "panas", "dingin", "cuaca", "iklim", "pemanasan", "global warming",
    "masa depan", "tahun depan", "ke depan", "tren suhu",
    "berapa suhu", "bagaimana suhu", "perkiraan suhu",
    "akan naik", "akan turun", "climate", "weather", "heat", "warm",
]

KEYWORDS_DATASET = {
    "country": ["negara", "country", "albania", "andorra", "austria"],
    "city"   : ["kota", "city", "abidjan", "addis", "ahmadabad"],
    "state"  : ["provinsi", "state", "arkhangel", "belgorod", "bryansk"],
}

DEFAULT_LABELS = {
    "global" : "global",
    "country": "country_Albania",
    "city"   : "city_Abidjan",
    "state"  : "state_Arkhangel'Sk",
}

def butuh_prediksi(teks: str) -> bool:
    teks_lower = teks.lower()
    return any(kw in teks_lower for kw in KEYWORDS_PREDIKSI)

def deteksi_dataset(teks: str) -> tuple:
    teks_lower = teks.lower()
    dataset    = "global"
    for ds, keywords in KEYWORDS_DATASET.items():
        if any(kw in teks_lower for kw in keywords):
            dataset = ds
            break
    return dataset, DEFAULT_LABELS[dataset]


# ─── Function: predict_temperature ───────────────────────────────────────────
def predict_temperature(dataset: str, label: str) -> str:
    print(f"  [API] Memanggil -> dataset={dataset}, label={label}")
    try:
        r = requests.post(
            f"{API_BASE}/predict",
            headers=HEADERS,
            json={"dataset": dataset, "label": label},
            timeout=TIMEOUT_SEC
        )
        if r.status_code == 404:
            return f"Data tidak ditemukan: {r.json().get('error', '')}"
        if r.status_code == 403:
            return "API Key tidak valid."
        if r.status_code != 200:
            return f"API error: status {r.status_code}"

        data         = r.json()
        forecast     = data.get("forecast", [])
        future_index = data.get("future_index", [])
        mae          = data.get("mae", 0)
        rmse         = data.get("rmse", 0)

        if not forecast:
            return "API tidak mengembalikan data."

        avg_f = round(sum(forecast) / len(forecast), 2)
        diff  = round(forecast[-1] - forecast[0], 2)
        tren  = "NAIK" if diff > 0.5 else ("TURUN" if diff < -0.5 else "STABIL")

        # Format baris forecast
        baris = ""
        for idx, val in zip(future_index, forecast):
            baris += f"  {idx}: {val:.2f}°C\n"

        return (
            f"Dataset  : {dataset.upper()}\n"
            f"Lokasi   : {label.replace('_', ' ').title()}\n"
            f"Periode  : {len(forecast)} tahun ke depan\n\n"
            f"Forecast:\n{baris}\n"
            f"Rata-rata  : {avg_f}°C\n"
            f"Tertinggi  : {max(forecast):.2f}°C\n"
            f"Terendah   : {min(forecast):.2f}°C\n"
            f"Tren       : {tren} ({diff:+.2f}°C)\n"
            f"MAE        : {mae:.4f} | RMSE: {rmse:.4f}"
        )

    except requests.exceptions.Timeout:
        return "API timeout."
    except requests.exceptions.ConnectionError:
        return "Tidak bisa konek ke API."
    except Exception as e:
        return f"Error: {str(e)}"


# ─── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Kamu adalah GlobalTemp Assistant — asisten AI yang cerdas, ramah, dan bisa menjawab SEMUA pertanyaan bebas.

ATURAN MUTLAK:
1. Selalu jawab dalam Bahasa Indonesia yang benar, natural, dan mudah dipahami
2. Jika diberi DATA PREDIKSI SUHU, gunakan data itu untuk menjawab — jangan ubah angkanya
3. Jika tidak diberi data prediksi, jawab pertanyaan apapun secara normal seperti AI asisten
4. JANGAN pernah mengarang angka suhu atau data yang tidak ada
5. JANGAN tampilkan format JSON atau kode apapun kecuali diminta
6. Jawab singkat, jelas, dan langsung ke intinya

Kamu bisa menjawab topik apapun: sains, sejarah, matematika, coding, budaya, dll."""


# ─── Proses pesan ─────────────────────────────────────────────────────────────
def proses_pesan(user_input: str, riwayat: list) -> str:
    pesan_ke_llm = list(riwayat)

    if butuh_prediksi(user_input):
        dataset, label = deteksi_dataset(user_input)
        data_api       = predict_temperature(dataset, label)

        # Inject data ke prompt — instruksi sangat ketat
        konten = (
            f"Pertanyaan user: {user_input}\n\n"
            f"DATA PREDIKSI DARI API (GUNAKAN INI, JANGAN UBAH ANGKANYA):\n"
            f"{data_api}\n\n"
            f"Tugas kamu: Jelaskan data di atas dengan bahasa Indonesia yang natural "
            f"dan mudah dipahami. Sebutkan angka-angka penting dan berikan kesimpulan tren."
        )
    else:
        konten = user_input

    pesan_ke_llm.append({"role": "user", "content": konten})

    try:
        response = ollama.chat(model=MODEL_LLM, messages=pesan_ke_llm)
        return response.get("message", {}).get("content", "Maaf, tidak bisa menjawab.")
    except ollama.ResponseError as e:
        if "not found" in str(e).lower():
            return f"Model '{MODEL_LLM}' belum diinstall. Jalankan: ollama pull {MODEL_LLM}"
        return f"Ollama error: {str(e)}"
    except ConnectionRefusedError:
        return "Ollama tidak berjalan. Jalankan: ollama serve"
    except Exception as e:
        return f"Error: {str(e)}"


# ─── CLI Interaktif ───────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  GlobalTemp AI Assistant")
    print(f"  Model : {MODEL_LLM} (Ollama)")
    print("=" * 60)
    print("  Tanya apa saja — bebas tanpa limit!")
    print("  'reset' = mulai baru | 'keluar' = berhenti")
    print("=" * 60)
    print()

    riwayat = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("Anda: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("keluar", "exit", "quit"):
                print("\nSampai jumpa!")
                sys.exit(0)

            if user_input.lower() == "reset":
                riwayat = [{"role": "system", "content": SYSTEM_PROMPT}]
                print("\n[Chat direset.]\n")
                continue

            print("\nAssistant:", end=" ", flush=True)
            respons = proses_pesan(user_input, riwayat)
            print(respons, "\n")

            # Simpan riwayat versi bersih
            riwayat.append({"role": "user",      "content": user_input})
            riwayat.append({"role": "assistant",  "content": respons})

            # Batasi max 20 pesan
            if len(riwayat) > 21:
                riwayat = [riwayat[0]] + riwayat[-20:]

        except KeyboardInterrupt:
            print("\n\nSampai jumpa!")
            sys.exit(0)

if __name__ == "__main__":
    main()