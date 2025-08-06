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