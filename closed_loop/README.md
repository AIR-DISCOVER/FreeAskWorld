# Communication with FreeAskWorld Simulator

For the current recommended local runtime path, start from the main [README.md](../README.md) and the ROS2-first integration notes in [docs/agent_ros2_integration.md](../docs/agent_ros2_integration.md).

If you need the currently validated ROS2 environment setup, see [docs/ros2_setup.md](../docs/ros2_setup.md).

> Note: this `closed_loop` path is the legacy/additive websocket compatibility path. For the current local Unity setup, external users should treat the ROS2-first path as the primary starting point.

## Start

Use this section if you intentionally want the websocket/closed-loop compatibility path.

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

## Agent Integration (OpenClaw / Claude Code / Codex / custom)

The agent bridge is additive. It does not replace the existing Unity-facing websocket protocol, and it does not change the baseline server entrypoint semantics in `server_ETPNav.py` or `server_BEVBERT.py`.

### What stays unchanged

- Existing websocket message types and semantics remain intact: `img`, `depth`, `rgbd`, `json`.
- Existing dataclasses remain intact: `NavigationCommand`, `Step`, `TransformData`.
- Existing shared globals remain available: `rgb_array`, `depth_array`, `transform_data`, `instruction`, `Init`.
- Existing baseline servers still talk to Unity over the same websocket flow.

### New bridge surfaces

- HTTP API: `closed_loop/agent_server.py`
- CLI: `closed_loop/agent_cli.py`
- MCP-friendly tool wrapper: `closed_loop/agent_mcp.py`
- Core bridge service: `closed_loop/agent_bridge.py`

### Startup

Run the existing websocket/baseline server first so Unity can connect as before:

```bash
cd closed_loop
python server_ETPNav.py
# or
python server_BEVBERT.py
```

Run the additive agent HTTP bridge in a second process:

```bash
cd closed_loop
python agent_cli.py serve --host 0.0.0.0 --port 8000
```

### CLI examples

```bash
python agent_cli.py status --json
python agent_cli.py observe --json
python agent_cli.py move-forward --distance-m 1.0 --json
python agent_cli.py stop --json
python agent_cli.py step --json
python agent_cli.py turn-left --degrees 15 --json
python agent_cli.py turn-right --degrees 15 --json
python agent_cli.py ask-human "Where is the target?" --json
```

### Multi-agent usage (same interface)

Any coding/assistant agent can call the same bridge interface. The agent identity changes, but the simulator API does not.

Examples:

```bash
# Claude Code / Codex / custom script can all call the same HTTP endpoint
curl -X POST http://127.0.0.1:8000/v1/action \
  -H "Content-Type: application/json" \
  -d '{"action":"move_forward","parameters":{"distance_m":1.0}}'
```

Recommended mapping:
- OpenClaw: call `agent_cli.py` or HTTP endpoints directly.
- Claude Code: generate JSON action payloads and post to `/v1/action`.
- Codex: same as Claude; keep tool wrapper logic outside simulator core.
- Custom agent: implement a tiny adapter that maps your internal action schema to bridge actions.

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
