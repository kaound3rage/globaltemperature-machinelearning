"""
====================================================
GlobalTemp - Hybrid AI Chatbot
====================================================
Sistem AI hybrid menggunakan:
  - Ollama (LLM lokal) model: tinyllama
  - API GlobalTemp untuk prediksi suhu

Cara pakai:
  pip install ollama requests
  ollama pull tinyllama
  python chatbot.py
====================================================
"""

import json
import re
import requests
import ollama
import sys

# ─── Config ───────────────────────────────────────────────────────────────────
API_BASE    = "https://2ekahxvxmb.execute-api.us-east-1.amazonaws.com/prod"
API_KEY     = "K7esYYbTSv3fX592ZkHMc2C3GGIr1BXG9Sla4Fld"
MODEL_LLM   = "tinyllama"
TIMEOUT_SEC = 15

HEADERS = {
    "Content-Type": "application/json",
    "x-api-key"   : API_KEY,
}

# ─── Keyword detector: apakah perlu panggil API? ─────────────────────────────
KEYWORDS_PREDIKSI = [
    "prediksi", "forecast", "suhu", "temperatur", "temperature",
    "panas", "dingin", "cuaca", "iklim", "pemanasan", "global warming",
    "masa depan", "tahun depan", "ke depan", "tren suhu", "naik", "turun",
    "berapa suhu", "bagaimana suhu", "perkiraan suhu", "akan naik", "akan turun",
    "climate", "weather", "heat", "cold", "warm",
]

KEYWORDS_DATASET = {
    "country": ["negara", "country", "albania", "andorra", "austria"],
    "city"   : ["kota", "city", "abidjan", "addis", "ahmadabad"],
    "state"  : ["provinsi", "state", "arkhangel", "belgorod", "bryansk"],
    "global" : ["global", "dunia", "bumi", "earth", "world"],
}

def butuh_prediksi(teks: str) -> bool:
    """Deteksi apakah pertanyaan user butuh data prediksi suhu."""
    teks_lower = teks.lower()
    return any(kw in teks_lower for kw in KEYWORDS_PREDIKSI)

def deteksi_dataset(teks: str) -> tuple:
    """Deteksi dataset dan label dari teks user."""
    teks_lower = teks.lower()

    dataset = "global"
    label   = "global"

    for ds, keywords in KEYWORDS_DATASET.items():
        if any(kw in teks_lower for kw in keywords):
            dataset = ds
            break

    # Default label per dataset
    default_labels = {
        "global" : "global",
        "country": "country_Albania",
        "city"   : "city_Abidjan",
        "state"  : "state_Arkhangel'Sk",
    }
    label = default_labels.get(dataset, "global")

    return dataset, label


# ─── Function: predict_temperature ───────────────────────────────────────────
def predict_temperature(dataset: str = "global", label: str = "global") -> str:
    """Panggil API prediksi suhu dan return hasil terformat."""
    print(f"\n  [API] Memanggil prediksi -> dataset={dataset}, label={label}")

    try:
        response = requests.post(
            f"{API_BASE}/predict",
            headers=HEADERS,
            json={"dataset": dataset, "label": label},
            timeout=TIMEOUT_SEC
        )

        if response.status_code == 404:
            return f"Data untuk {label} tidak ditemukan di API."
        if response.status_code == 403:
            return "API Key tidak valid."
        if response.status_code != 200:
            return f"API error: status {response.status_code}"

        data         = response.json()
        forecast     = data.get("forecast", [])
        future_index = data.get("future_index", [])
        mae          = data.get("mae", 0)
        rmse         = data.get("rmse", 0)

        if not forecast:
            return "API tidak mengembalikan data prediksi."

        label_clean   = label.replace("_", " ").title()
        dataset_clean = dataset.upper()
        avg_f         = sum(forecast) / len(forecast)
        diff          = forecast[-1] - forecast[0]

        if diff > 0.5:
            tren = f"cenderung NAIK {diff:.2f} derajat Celsius"
        elif diff < -0.5:
            tren = f"cenderung TURUN {abs(diff):.2f} derajat Celsius"
        else:
            tren = f"relatif STABIL (perubahan {diff:.2f} derajat Celsius)"

        baris_forecast = ""
        for idx, val in zip(future_index, forecast):
            baris_forecast += f"- {idx}: {val:.2f} derajat Celsius\n"

        return f"""
DATA PREDIKSI SUHU (dari API GlobalTemp):
Dataset  : {dataset_clean}
Lokasi   : {label_clean}
Periode  : {len(forecast)} tahun ke depan

Forecast:
{baris_forecast}
Statistik:
- Rata-rata  : {avg_f:.2f} derajat Celsius
- Tertinggi  : {max(forecast):.2f} derajat Celsius
- Terendah   : {min(forecast):.2f} derajat Celsius
- Tren       : Suhu {tren}
- MAE Model  : {mae:.4f}
- RMSE Model : {rmse:.4f}
""".strip()

    except requests.exceptions.Timeout:
        return "API timeout - server tidak merespons."
    except requests.exceptions.ConnectionError:
        return "Tidak bisa konek ke API. Cek koneksi internet."
    except Exception as e:
        return f"Error memanggil API: {str(e)}"


# ─── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Kamu adalah GlobalTemp Assistant, asisten AI yang ramah dan cerdas.
Kamu bisa menjawab semua pertanyaan bebas seperti AI pada umumnya — tentang sains, sejarah, teknologi, budaya, matematika, coding, kehidupan sehari-hari, dan topik lainnya.
Kamu juga ahli dalam iklim, cuaca, dan suhu global.
Jika diberi data prediksi suhu dari API, gunakan data tersebut untuk memberikan penjelasan yang natural dan mudah dipahami.
Jangan pernah mengarang angka suhu — gunakan hanya data yang diberikan.
Selalu gunakan bahasa Indonesia yang natural, ramah, dan jelas."""


# ─── Proses pesan ─────────────────────────────────────────────────────────────
def proses_pesan(user_input: str, riwayat_chat: list) -> str:
    """
    Deteksi intent secara manual, panggil API jika perlu,
    lalu kirim ke LLM untuk dirangkum jadi jawaban natural.
    """
    pesan_ke_llm = list(riwayat_chat)

    # Deteksi apakah butuh data prediksi
    if butuh_prediksi(user_input):
        dataset, label = deteksi_dataset(user_input)
        data_api       = predict_temperature(dataset, label)

        # Inject data API ke dalam prompt user
        pesan_user = f"""{user_input}

[DATA DARI API - gunakan ini untuk menjawab, jangan karang angka lain]:
{data_api}

Berikan jawaban yang natural, jelas, dan mudah dipahami berdasarkan data di atas."""
    else:
        # Pertanyaan biasa tanpa data API
        pesan_user = user_input

    pesan_ke_llm.append({"role": "user", "content": pesan_user})

    try:
        response = ollama.chat(
            model   = MODEL_LLM,
            messages= pesan_ke_llm,
        )
        return response.get("message", {}).get("content", "Maaf, saya tidak bisa menjawab saat ini.")

    except ollama.ResponseError as e:
        return f"[ERROR] Ollama: {str(e)}"
    except ConnectionRefusedError:
        return "[ERROR] Ollama tidak berjalan. Jalankan: ollama serve"
    except Exception as e:
        return f"[ERROR] {str(e)}"


# ─── CLI Interaktif ───────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  GlobalTemp AI Assistant")
    print("  LLM : tinyllama (Ollama)")
    print("  API : GlobalTemp Forecast API")
    print("=" * 60)
    print("  Tanya apa saja — bebas tanpa limit!")
    print("  Ketik 'keluar' untuk berhenti")
    print("  Ketik 'reset' untuk mulai percakapan baru")
    print("=" * 60)
    print()

    riwayat_chat = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    while True:
        try:
            user_input = input("Anda: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("keluar", "exit", "quit"):
                print("\nTerima kasih! Sampai jumpa.")
                sys.exit(0)

            if user_input.lower() == "reset":
                riwayat_chat = [{"role": "system", "content": SYSTEM_PROMPT}]
                print("\n[Chat direset.]\n")
                continue

            # Proses dan tampilkan respons
            print("\nAssistant:", end=" ", flush=True)
            respons = proses_pesan(user_input, riwayat_chat)
            print(respons)
            print()

            # Simpan ke riwayat (simpan versi bersih tanpa inject API)
            riwayat_chat.append({"role": "user",      "content": user_input})
            riwayat_chat.append({"role": "assistant",  "content": respons})

            # Batasi riwayat max 20 pesan
            if len(riwayat_chat) > 21:
                riwayat_chat = [riwayat_chat[0]] + riwayat_chat[-20:]

        except KeyboardInterrupt:
            print("\n\nTerima kasih! Sampai jumpa.")
            sys.exit(0)


if __name__ == "__main__":
    main()