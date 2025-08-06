import asyncio
import websockets

async def test_ws():
    url = "wss://j-practitioners-subscription-assured.trycloudflare.com"
    try:
        async with websockets.connect(url) as ws:
            print("连接成功")
            await ws.send("hello")
            msg = await ws.recv()
            print("收到消息:", msg)
    except Exception as e:
        print("连接失败:", e)

asyncio.run(test_ws())
