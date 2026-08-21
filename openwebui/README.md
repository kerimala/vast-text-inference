# Open WebUI

Die Compose-Konfiguration verwendet das gepinnte Release `v0.10.2`, ein
persistentes Docker-Volume und eine reine Loopback-Bindung.

## Start und Status

```bash
docker compose up -d
docker compose ps
docker compose logs --tail=100 open-webui
```

Healthcheck:

```bash
curl --fail http://127.0.0.1:3000/health
```

Die spaetere OpenAI-kompatible vLLM-API wird ueber den Hostnamen
`host.docker.internal` angesprochen. Der Compose-Stack enthaelt dafuer bereits
den lokalen Host-Gateway-Eintrag. Fuer den Vast-Test ist die Base-URL
`http://host.docker.internal:8000/v1` vorkonfiguriert. Sie ist nur erreichbar,
solange der SSH-Tunnel auf der WSL-Docker-Bridge aktiv ist.

Die Oberflaeche bleibt auf dem Tower persistent und wird im privaten Tailnet
unter `https://desktop-kdh6ecm.tail5a261a.ts.net:9443` bereitgestellt. Das
jeweilige Vast-Modell wird nur als austauschbarer OpenAI-kompatibler Backend-
Endpoint angebunden.

Mikrofon-Diktate verwenden standardmaessig das lokal laufende, mehrsprachige
Whisper-Modell `small` mit CPU-`int8` und VAD. Damit bleiben Audiodaten auf der
eigenen Infrastruktur und die Transkription funktioniert ohne API-Key. Eine
OpenAI-kompatible STT-API kann spaeter im Open-WebUI-Adminbereich unter
`Settings -> Audio` eingetragen werden.
