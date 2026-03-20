# Communication with FreeAskWorld Simulator

## Start

conda create -n FreeAskWorld python=3.10
conda activate FreeAskWorld
pip install websockets fastapi uvicorn aiohttp
pip install -r requirements.txt


Python as server, so run server.py at first, then run unity

### Build Cloudflare Tunnel To NAT-DDNS
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb

### Run CLoudflare Tunnel And Websocket Sever
cloudflared tunnel --url http://localhost:8765
python server.py

Or directly run
FreeAskWorld.sh

Public network IP can be accessed in terminal with green colour words, like: 检测到公网地址: https://photographers-exotic-completing-holdings.trycloudflare.com

## Method
Communication Flow:
Unity (client) connects to the server via WebSocket.

Unity sends messages to the server using JsonSender, ImageSender, and DepthSender.

The server processes incoming messages and responds back.

Unity receives server replies through JsonReceiver.

## Run Baselines
cd path/to/your/FreeAskWorld
sh FreeAskWorldClosedLoopBEVBERT.sh
sh FreeAskWorldClosedLoopETPNav.sh

## OpenClaw Integration

The OpenClaw bridge is additive. It does not replace the existing Unity-facing websocket protocol, and it does not change the baseline server entrypoint semantics in `server_ETPNav.py` or `server_BEVBERT.py`.

### What stays unchanged

- Existing websocket message types and semantics remain intact: `img`, `depth`, `rgbd`, `json`.
- Existing dataclasses remain intact: `NavigationCommand`, `Step`, `TransformData`.
- Existing shared globals remain available: `rgb_array`, `depth_array`, `transform_data`, `instruction`, `Init`.
- Existing baseline servers still talk to Unity over the same websocket flow.

### New bridge surfaces

- HTTP API: `closed_loop/openclaw_server.py`
- CLI: `closed_loop/openclaw_cli.py`
- MCP-friendly tool wrapper: `closed_loop/openclaw_mcp.py`
- Core bridge service: `closed_loop/openclaw_bridge.py`

### Startup

Run the existing websocket/baseline server first so Unity can connect as before:

```bash
cd /home/wyabz/research/FreeAskWorld/closed_loop
python server_ETPNav.py
# or
python server_BEVBERT.py
```

Run the additive OpenClaw HTTP bridge in a second process:

```bash
cd /home/wyabz/research/FreeAskWorld/closed_loop
python openclaw_cli.py serve --host 0.0.0.0 --port 8000
```

### CLI examples

```bash
python openclaw_cli.py status --json
python openclaw_cli.py observe --json
python openclaw_cli.py move-forward --distance-m 1.0 --json
python openclaw_cli.py stop --json
python openclaw_cli.py step --json
python openclaw_cli.py turn-left --degrees 15 --json
python openclaw_cli.py turn-right --degrees 15 --json
python openclaw_cli.py ask-human "Where is the target?" --json
```

### HTTP API

- `GET /healthz`
- `GET /v1/status`
- `GET /v1/observation`
- `POST /v1/action`
- `POST /v1/navigation-command`
- `POST /v1/step`

Example generic action:

```bash
curl -X POST http://127.0.0.1:8000/v1/action \
  -H "Content-Type: application/json" \
  -d '{"action":"move_forward","parameters":{"distance_m":1.0}}'
```

### Supported actions

- `move_forward(distance_m)`
- `stop()`
- `step()`
- `turn_left(degrees)`
- `turn_right(degrees)`
- `ask_human(prompt)`
- `navigation_command(local_position_offset, local_rotation_offset, is_stopped)`

### Limitations

- The bridge requires an active Unity websocket connection before movement or step commands can be delivered.
- `turn_left` and `turn_right` use a conservative `LocalRotationOffset` compatibility mapping because the existing simulator rotation contract is not fully documented here.
- `ask_human` is currently a V1 placeholder that records the prompt in bridge state and returns a clear result, but it does not yet inject a new simulator-side interaction flow on its own.
