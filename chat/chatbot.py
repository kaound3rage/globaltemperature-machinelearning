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

API yang digunakan:
  POST https://2ekahxvxmb.execute-api.us-east-1.amazonaws.com/prod/predict
  GET  https://2ekahxvxmb.execute-api.us-east-1.amazonaws.com/prod/list
====================================================
"""

import json
import requests
import ollama
import sys

# ─── Config API ───────────────────────────────────────────────────────────────
API_BASE    = "https://2ekahxvxmb.execute-api.us-east-1.amazonaws.com/prod"
API_KEY     = "K7esYYbTSv3fX592ZkHMc2C3GGIr1BXG9Sla4Fld"
MODEL_LLM   = "tinyllama"
TIMEOUT_SEC = 15

HEADERS = {
    "Content-Type": "application/json",
    "x-api-key"   : API_KEY,
}

# ─── Label cache ──────────────────────────────────────────────────────────────
_label_cache = {}

def get_available_labels(dataset: str) -> list:
    """Ambil daftar label yang tersedia untuk dataset tertentu."""
    global _label_cache
    if dataset in _label_cache:
        return _label_cache[dataset]
    try:
        r = requests.get(
            f"{API_BASE}/list",
            headers=HEADERS,
            params={"dataset": dataset},
            timeout=TIMEOUT_SEC
        )
        if r.status_code == 200:
            labels = r.json().get("labels", [])
            _label_cache[dataset] = labels
            return labels
    except Exception:
        pass
    return []


# ─── Function: predict_temperature ───────────────────────────────────────────
def predict_temperature(dataset: str = "global", label: str = None) -> str:
    """
    Panggil API prediksi suhu GlobalTemp.

    Args:
        dataset : "global" | "country" | "city" | "state"
        label   : label spesifik lokasi. Jika None, pakai default.

    Returns:
        String hasil prediksi yang sudah diformat natural
    """
    # Default label per dataset
    if not label:
        default_labels = {
            "global" : "global",
            "country": "country_Albania",
            "city"   : "city_Abidjan",
            "state"  : "state_Arkhangel'Sk",
        }
        label = default_labels.get(dataset, "global")

    print(f"\n  [API] Memanggil prediksi -> dataset={dataset}, label={label}")

    try:
        response = requests.post(
            f"{API_BASE}/predict",
            headers=HEADERS,
            json={"dataset": dataset, "label": label},
            timeout=TIMEOUT_SEC
        )

        if response.status_code == 404:
            return f"[ERROR] {response.json().get('error', 'Label tidak ditemukan')}"
        if response.status_code == 403:
            return "[ERROR] API Key tidak valid atau tidak ada akses."
        if response.status_code != 200:
            return f"[ERROR] API mengembalikan status {response.status_code}"

        data         = response.json()
        forecast     = data.get("forecast", [])
        future_index = data.get("future_index", [])
        mae          = data.get("mae", 0)
        rmse         = data.get("rmse", 0)

        if not forecast:
            return "[ERROR] API tidak mengembalikan data prediksi."

        label_clean   = label.replace("_", " ").title()
        dataset_clean = dataset.upper()

        # Baris forecast
        forecast_lines = ""
        for idx, val in zip(future_index, forecast):
            forecast_lines += f"    - {idx}: {val:.2f} derajat Celsius\n"

        diff  = forecast[-1] - forecast[0]
        avg_f = sum(forecast) / len(forecast)

        if diff > 0.5:
            tren = f"cenderung NAIK sebesar {diff:.2f} derajat Celsius"
        elif diff < -0.5:
            tren = f"cenderung TURUN sebesar {abs(diff):.2f} derajat Celsius"
        else:
            tren = f"relatif STABIL (perubahan hanya {diff:.2f} derajat Celsius)"

        hasil = f"""
[HASIL PREDIKSI SUHU]
Dataset  : {dataset_clean}
Lokasi   : {label_clean}
Periode  : {len(forecast)} tahun ke depan

Forecast per tahun:
{forecast_lines}
Ringkasan:
- Rata-rata suhu  : {avg_f:.2f} derajat Celsius
- Suhu tertinggi  : {max(forecast):.2f} derajat Celsius
- Suhu terendah   : {min(forecast):.2f} derajat Celsius
- Tren            : Suhu {tren}
- Akurasi model   : MAE={mae:.4f}, RMSE={rmse:.4f}
"""
        return hasil.strip()

    except requests.exceptions.Timeout:
        return "[ERROR] API timeout - server tidak merespons dalam 15 detik."
    except requests.exceptions.ConnectionError:
        return "[ERROR] Tidak bisa konek ke API. Cek koneksi internet."
    except requests.exceptions.JSONDecodeError:
        return "[ERROR] Response API bukan format JSON valid."
    except Exception as e:
        return f"[ERROR] Terjadi kesalahan: {str(e)}"


# ─── Tool definition untuk Ollama ────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name"       : "predict_temperature",
            "description": (
                "Prediksi suhu masa depan menggunakan data historis temperatur global. "
                "Gunakan function ini HANYA jika user bertanya tentang prediksi suhu, "
                "forecast suhu, atau tren perubahan suhu di masa depan. "
                "Jangan gunakan untuk pertanyaan umum atau percakapan biasa."
            ),
            "parameters": {
                "type"      : "object",
                "properties": {
                    "dataset": {
                        "type"       : "string",
                        "description": (
                            "Dataset yang digunakan. Pilihan: "
                            "'global' untuk suhu rata-rata global, "
                            "'country' untuk per negara, "
                            "'city' untuk per kota, "
                            "'state' untuk per provinsi. "
                            "Default: 'global'"
                        ),
                        "enum": ["global", "country", "city", "state"]
                    },
                    "label": {
                        "type"       : "string",
                        "description": (
                            "Label spesifik lokasi. Contoh: "
                            "'global', 'country_Albania', 'city_Abidjan', "
                            "'state_Belgorod'. "
                            "Kosongkan jika tidak disebutkan user."
                        )
                    }
                },
                "required": ["dataset"]
            }
        }
    }
]

# ─── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Kamu adalah GlobalTemp Assistant, asisten AI yang ahli dalam analisis dan prediksi suhu global.

Kemampuanmu:
1. Menjawab pertanyaan umum tentang iklim, cuaca, dan perubahan suhu
2. Memprediksi suhu masa depan menggunakan function predict_temperature

Aturan WAJIB:
- Jika user bertanya prediksi suhu, forecast, atau angka suhu masa depan -> SELALU gunakan function predict_temperature
- JANGAN pernah mengarang atau menebak angka suhu
- Jika pertanyaan tidak butuh data prediksi -> jawab seperti chatbot biasa
- Gunakan bahasa Indonesia yang natural, ramah, dan mudah dipahami
- Jelaskan hasil prediksi dengan kalimat yang mudah dimengerti

Dataset yang tersedia:
- global  : suhu rata-rata global
- country : suhu per negara (Albania, Andorra, Austria)
- city    : suhu per kota (Abidjan, Addis Abeba, Ahmadabad)
- state   : suhu per provinsi (Arkhangel'Sk, Belgorod, Bryansk)"""


# ─── Proses satu giliran percakapan ──────────────────────────────────────────
def proses_pesan(riwayat_chat: list) -> str:
    """
    Kirim riwayat chat ke Ollama, deteksi tool_calls,
    jalankan function jika perlu, kembalikan respons final.
    """
    try:
        # Round 1: LLM memutuskan jawab langsung atau panggil tool
        response = ollama.chat(
            model   = MODEL_LLM,
            messages= riwayat_chat,
            tools   = TOOLS,
        )

        pesan      = response.get("message", {})
        tool_calls = pesan.get("tool_calls", [])

        # Tidak ada tool call -> jawab langsung
        if not tool_calls:
            return pesan.get("content", "Maaf, saya tidak bisa menjawab saat ini.")

        # Ada tool call -> jalankan function
        hasil_tools = []

        for tool_call in tool_calls:
            func_name = tool_call.get("function", {}).get("name", "")
            func_args = tool_call.get("function", {}).get("arguments", {})

            # Pastikan args adalah dict
            if isinstance(func_args, str):
                try:
                    func_args = json.loads(func_args)
                except json.JSONDecodeError:
                    func_args = {}

            print(f"\n  [LLM] Meminta tool: {func_name}({func_args})")

            if func_name == "predict_temperature":
                dataset = func_args.get("dataset", "global")
                label   = func_args.get("label", None)
                hasil   = predict_temperature(dataset=dataset, label=label)
            else:
                hasil = f"[ERROR] Function '{func_name}' tidak dikenal."

            hasil_tools.append({
                "role"   : "tool",
                "content": hasil,
            })

        # Round 2: Kirim hasil tool ke LLM untuk dirangkum jadi kalimat natural
        riwayat_dengan_tool = riwayat_chat + [pesan] + hasil_tools

        response_final = ollama.chat(
            model   = MODEL_LLM,
            messages= riwayat_dengan_tool,
        )

        return response_final.get("message", {}).get(
            "content", "Maaf, saya tidak bisa merangkum hasil prediksi."
        )

    except ollama.ResponseError as e:
        if "model" in str(e).lower():
            return (
                f"[ERROR] Model '{MODEL_LLM}' tidak ditemukan di Ollama. "
                f"Jalankan perintah berikut:\n  ollama pull {MODEL_LLM}"
            )
        return f"[ERROR] Ollama error: {str(e)}"
    except ConnectionRefusedError:
        return (
            "[ERROR] Ollama tidak berjalan. "
            "Pastikan Ollama sudah diinstall dan jalankan: ollama serve"
        )
    except Exception as e:
        return f"[ERROR] Terjadi kesalahan tidak terduga: {str(e)}"


# ─── CLI Interaktif ───────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  GlobalTemp AI Assistant")
    print("  LLM    : tinyllama (Ollama)")
    print("  API    : GlobalTemp Forecast API")
    print("=" * 60)
    print("  Ketik 'keluar' atau 'exit' untuk berhenti")
    print("  Ketik 'reset' untuk mulai percakapan baru")
    print("=" * 60)
    print()

    # Riwayat dimulai dengan system prompt
    riwayat_chat = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    while True:
        try:
            user_input = input("Anda: ").strip()

            if not user_input:
                continue

            # Perintah khusus
            if user_input.lower() in ("keluar", "exit", "quit"):
                print("\nTerima kasih sudah menggunakan GlobalTemp Assistant. Sampai jumpa!")
                sys.exit(0)

            if user_input.lower() == "reset":
                riwayat_chat = [{"role": "system", "content": SYSTEM_PROMPT}]
                print("\n[Chat direset. Silakan mulai percakapan baru.]\n")
                continue

            if user_input.lower() == "help":
                print("""
Contoh pertanyaan:
  - "prediksi suhu global 10 tahun ke depan"
  - "bagaimana tren suhu di Albania?"
  - "forecast suhu kota Abidjan"
  - "apa itu pemanasan global?"
  - "apakah suhu bumi akan terus naik?"
""")
                continue

            # Tambah pesan user
            riwayat_chat.append({"role": "user", "content": user_input})

            # Proses dan tampilkan respons
            print("\nAssistant:", end=" ", flush=True)
            respons = proses_pesan(riwayat_chat)
            print(respons)
            print()

            # Simpan respons ke riwayat
            riwayat_chat.append({"role": "assistant", "content": respons})

            # Batasi riwayat max 20 pesan (+ 1 system prompt)
            if len(riwayat_chat) > 21:
                riwayat_chat = [riwayat_chat[0]] + riwayat_chat[-20:]

        except KeyboardInterrupt:
            print("\n\nTerima kasih! Sampai jumpa.")
            sys.exit(0)


if __name__ == "__main__":
    main()