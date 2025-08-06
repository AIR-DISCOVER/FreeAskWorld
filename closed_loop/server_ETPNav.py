# server.py

import asyncio
import websockets
import json
import logging
from ws_handlers import handle

from baselines import *

logging.basicConfig(level=logging.INFO)

async def main():
    try:
        server = await websockets.serve(
            etpnav_handle,
            "0.0.0.0",
            8765,
            ping_interval=20,
            ping_timeout=30,
            max_size=30_485_760
        )

        import socket
        host_ip = socket.gethostbyname(socket.gethostname())
        logging.info(f"✅ WebSocket server running:")
        logging.info(f"  - Local: ws://localhost:8765")
        logging.info(f"  - Network: ws://{host_ip}:8765")
        logging.info(f"  - All interfaces: ws://0.0.0.0:8765")

        await asyncio.Future()  # 持续运行
    except Exception as e:
        logging.error(f"❌ Failed to start server: {e}")

if __name__ == "__main__":
    if sys.version_info >= (3, 7):
        asyncio.run(main())
    else:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
