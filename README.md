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

## Erste Modell-Baseline

Die erste Baseline ist
[`AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16)
auf genau einer A100/A800 mit 80 GB VRAM.

- Modellrevision: `da9996c35307783ecaf25bdd240aeade954dd2c6`
- Container: offizielles vLLM `v0.23.0`, gepinnt auf den `linux/amd64`-Digest
- Startkontext: 32K; keine spekulative Dekodierung
- Endpoint: nur Remote-Loopback `127.0.0.1:8000`, Zugriff per SSH-Tunnel
- Vast-Disk: 140 GB
- Preisgrenze: 0,95 USD/Stunde inklusive des von Vast berechneten Storage-Anteils
- Kostenvoranschlag beruecksichtigt zusaetzlich 75 GB fuer Modell und einen
  eventuell ungecachten Container
- Nicht verifizierte Hosts sind erlaubt, Zuverlaessigkeit muss mindestens 95 % betragen
- Privates Vast-Template: `aeon-bf16-a100-80gb`

NVFP4 ist fuer diese Baseline bewusst nicht gewaehlt: A100/A800 haben keine
native FP4-Beschleunigung. Der vom Modellautor empfohlene AEON-Container ist
zudem nur fuer `linux/arm64` publiziert; das Template nutzt daher den offiziellen
`linux/amd64`-Build von vLLM.

## Angebote suchen (kostenfrei)

```bash
python3 scripts/vast_lab.py search
```

Das Ergebnis ist nur ein Snapshot. Vor einer Miete wird das ausgewaehlte
Angebot erneut gegen GPU, Architektur, Speicher, Zuverlaessigkeit, Bandbreite
und Preisgrenze validiert.

## Deployment (kostenpflichtig, noch nicht ausfuehren)

Der private Vast-Template-Hash ist in der Profil-Datei hinterlegt:

```bash
python3 scripts/vast_lab.py quote MACHINE_ID --ttl-minutes 120

python3 scripts/vast_lab.py deploy MACHINE_ID \
  --ttl-minutes 120 \
  --confirm 'RENT MACHINE MACHINE_ID UP TO USD BETRAG' \
  --execute
```

Der Bestaetigungstext wird aus dem unmittelbar neu gelesenen Preis berechnet.
Ohne `--execute`, exakten Text, gueltiges Angebot und funktionierendes lokales
systemd wird keine Instanz erstellt. Direkt nach der Erstellung wird ein lokaler
Cleanup-Timer aktiviert; bei Startup-Fehler oder Timeout wird die Instanz sofort
zerstoert. Vast stellt in diesem Ablauf kein providerseitiges Kosten-Hardcap
bereit: Windows und WSL sollten daher bis zum Ende des Tests laufen. Der Timer
ist eine Notbremse, kein Ersatz fuer die Kontrolle in der Vast-Konsole.

Nach erfolgreichem Startup gibt das Tool den SSH-Tunnel aus. Wenn der Tunnel auf
dem WSL-Host laeuft, ist die API unter `http://127.0.0.1:8000/v1` erreichbar:

```bash
python3 scripts/smoke_test.py
```

Open WebUI verwendet danach als OpenAI-kompatible Basis-URL
`http://host.docker.internal:8000/v1`. Ein beliebiger nicht-leerer lokaler
API-Key kann in Open WebUI gesetzt werden; vLLM selbst bekommt in dieser
Test-Baseline keinen extern exponierten Port.
