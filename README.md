# Vast Text Inference Lab

Reproduzierbare, kostenkontrollierte Text-Inference-Tests auf Vast.ai.

## Aktueller Stand

- Die lokale Open-WebUI-Baseline in WSL bleibt fuer dauerhafte Accounts und
  Chats erhalten.
- Das Qwen3.8-FP8-Profil startet zusaetzlich eine eigene, ephemere Open-WebUI-
  Instanz auf dem gemieteten Vast-Host.
- Vast CLI wird lokal in WSL ausgefuehrt.
- Modell-API und Vast-Open-WebUI binden nur an Remote-Loopback und werden per
  SSH-Tunnel erreicht; es gibt keine oeffentlichen Modell- oder UI-Ports.

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
- vLLM und die optionale Vast-Open-WebUI werden ueber einen SSH-Tunnel erreicht.
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

## Qwen3.8 OrcaRouter FP8 mit Open WebUI auf Vast

Das zweite Profil ist
[`orcarouter/Qwen3.8-27B-Uncensored-FP8`](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8)
auf genau einer nativen FP8-GPU mit 48 GB VRAM.

- Profil: `config/orcarouter-qwen38-fp8-openwebui.json`
- Provisioning: `provisioning/orcarouter-qwen38-fp8-openwebui.sh`
- Modellrevision: `0787858da83e6640e289c0c22d092d92f4e97fdb`
- Container: vLLM `v0.24.0`, gepinnt auf den `linux/amd64`-Digest
- GPU-Allowlist: RTX 6000 Ada, RTX 5880 Ada oder L40S
- Startkontext: 32K, FP8-KV-Cache und MTP mit drei spekulativen Tokens
- Textbetrieb: `--language-model-only`, um auf 48 GB belastbare KV-Reserve zu
  behalten
- Open WebUI: `0.10.2`, in einem separaten Python-Venv innerhalb der
  Vast-Instanz
- Remote-Ports: nur Loopback `127.0.0.1:8000` fuer vLLM und
  `127.0.0.1:3000` fuer Open WebUI
- Vast-Disk: 120 GB
- Preisgrenze: 0,85 USD/Stunde inklusive Storage-Anteil
- Privates Vast-Template: `qwen38-orcarouter-uncensored-fp8-openwebui-48gb`

Das Modell-Repository ist zugangsbeschraenkt. `HF_TOKEN` muss beim Deployment
aus einer lokalen Secret-Quelle an Vast uebergeben werden; der Token gehoert
weder in Git noch in das private Template. Das Provisioning bricht ohne Token
fail-closed ab.

Open-WebUI-Accounts und Chats dieses Profils liegen unter `/workspace` der
Vast-Instanz. Sie ueberleben einen normalen Stop/Start derselben Instanz, aber
nicht deren Zerstoerung. Fuer dauerhafte Chats bleibt die lokale
Open-WebUI-Baseline die bessere Ablage.

## Angebote suchen (kostenfrei)

```bash
python3 scripts/vast_lab.py search

python3 scripts/vast_lab.py \
  --config config/orcarouter-qwen38-fp8-openwebui.json \
  search
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

python3 scripts/vast_lab.py \
  --config config/orcarouter-qwen38-fp8-openwebui.json \
  deploy MACHINE_ID \
  --ttl-minutes 240 \
  --hf-token-file ~/.cache/huggingface/token \
  --webui-public-url 'https://WINDOWS-NODE.TAILNET.ts.net:9443' \
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

Beim Qwen3.8-Profil enthaelt der ausgegebene SSH-Befehl zwei Forwardings. Danach
sind vLLM unter `http://127.0.0.1:8000/v1` und die auf Vast laufende Open WebUI
unter `http://127.0.0.1:3000` erreichbar.

Fuer privaten Handy-Zugriff kann der Windows-Tower ausschliesslich innerhalb
des Tailnets per Tailscale Serve auf den lokalen UI-Tunnel weiterleiten, zum
Beispiel auf einem freien HTTPS-Port:

```powershell
tailscale serve --bg --https=9443 http://127.0.0.1:3000
tailscale serve status
```

Das Handy muss mit demselben Tailnet verbunden sein. Tailscale Funnel und
Router-Portweiterleitungen bleiben deaktiviert. Nach dem ersten Open-WebUI-
Account wird die Registrierung geschlossen; weitere Nutzer bleiben mindestens
im Status `pending`.
