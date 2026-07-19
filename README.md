# Vast Text Inference Lab

Reproduzierbare, kostenkontrollierte Text-Inference-Tests auf Vast.ai.

## Aktueller Stand

- Open WebUI laeuft lokal in WSL und ist nur unter `http://localhost:3000` erreichbar.
- Die Open-WebUI-Daten liegen in einem persistenten Docker-Volume.
- Vast CLI wird lokal in WSL ausgefuehrt.
- Vast-Instanzen und Modell-Endpoints werden erst in einer spaeteren Phase erstellt.

## Open WebUI

```bash
cd openwebui
docker compose up -d
docker compose ps
curl --fail http://127.0.0.1:3000/health
```

Beenden, ohne Chats oder Einstellungen zu loeschen:

```bash
cd openwebui
docker compose down
```

Das Volume darf nicht mit `docker compose down -v` entfernt werden, sofern die
lokalen Open-WebUI-Daten erhalten bleiben sollen.

## Sicherheitsgrenzen

- Keine API Keys, Tokens oder SSH-Schluessel in diesem Repository speichern.
- Open WebUI bleibt an `127.0.0.1` gebunden.
- Der spaetere vLLM-Endpoint wird ueber einen SSH-Tunnel erreicht.
- Provisioning-Skripte werden von einer festen Git-Revision geladen.
- Eine Vast-Instanz wird nur mit Preisgrenze, Startup-Timeout und Cleanup-Guard erstellt.

## Geplante Modell-Baseline

- `Qwen/Qwen3.6-27B-FP8`
- `bottlecapai/ThinkingCap-Qwen3.6-27B-FP8`
- ein noch festzulegendes abliterated/uncensored Modell

