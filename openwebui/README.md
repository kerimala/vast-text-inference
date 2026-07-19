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
den lokalen Host-Gateway-Eintrag.
