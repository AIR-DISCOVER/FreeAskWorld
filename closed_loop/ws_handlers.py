# ws_handlers.py
import base64
import io
import os
import json
import logging
from PIL import Image
from datetime import datetime

from messages import *
import shared_state

logging.basicConfig(level=logging.INFO)


async def handle(websocket):
    client_ip = websocket.remote_address[0]
    logging.info(f"🔗 Client connected from {client_ip}")
    try:
        welcome = {"type": "system", "message": "Connected to server"}
        await websocket.send(json.dumps(welcome))

        async for message in websocket:
            data = json.loads(message)
            msg_type = data.get("type")
            logging.info(f"📩 Received '{msg_type}' message from {client_ip}")
            try:
                if msg_type == "img":
                    await handle_img(data, websocket)
                elif msg_type == "depth":
                    await handle_depth(data, websocket)
                elif msg_type == "rgbd":
                    await handle_rgbd(data, websocket)
                elif msg_type == "json":
                    await handle_json(data, websocket)
                else:
                    await handle_unknown(data, websocket)
                    
                # 或者异步推理
                # result = await run_neural_network_async(input_content)

                # 把推理结果封装成 JSON 发送回客户端
                # 通知仿真器步进
                # nav_cmd = NavigationCommand(
                #     LocalPositionOffset=np.array([0.0, 0.0, 3.0]),
                #     LocalRotationOffset=np.array([0.0, 0.0, 0.0, 1.0])  # x, y, z, w
                # )
                # output_response = {
                #     "type": "json",
                #     "json_type": "NavigationCommand",
                #     "content": nav_cmd.to_dict()
                # }
                # await websocket.send(json.dumps(output_response))
                
                # step = Step(IsStep=True)
                # step_response = {
                #     "type": "json",
                #     "json_type": "Step",
                #     "content": step.to_dict()
                # }
                # await websocket.send(json.dumps(step_response))

            except json.JSONDecodeError:
                logging.error(f"❌ Invalid JSON from {client_ip}: {message}")
                # error_msg = {"type": "error", "message": "Invalid JSON format"}
                # await websocket.send(json.dumps(error_msg))
    except Exception as e:
        logging.error(f"❌ Error with client {client_ip}: {e}")
    finally:
        logging.info(f"❎ Client disconnected: {client_ip}")

# # 接收图像并还原
# async def handle_img(data, websocket):
#     try:
#         os.makedirs("received", exist_ok=True)
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

#         if 'rgb' in data and data['rgb']:
#             rgb_bytes = base64.b64decode(data['rgb'])
#             rgb_image = Image.open(io.BytesIO(rgb_bytes))
#             rgb_path = f"received/rgb_{timestamp}.png"
#             rgb_image.save(rgb_path)
#             print(f"✅ RGB image saved to {rgb_path}")

#     except Exception as e:
#         print(f"❌ Failed to handle RGB image: {e}")
#         await websocket.send(json.dumps({
#             "type": "error",
#             "message": f"RGB handling failed: {str(e)}"
#         }))

# async def handle_depth(data, websocket):
#     try:
#         os.makedirs("received", exist_ok=True)
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

#         # 处理 EXR 格式深度图
#         if 'jpg' in data and data['jpg']:
#             jpg_bytes = base64.b64decode(data['jpg'])
#             jpg_path = f"received/depth_{timestamp}.jpg"
#             with open(jpg_path, 'wb') as f:
#                 f.write(jpg_bytes)
#             print(f"✅ EXR depth image saved to {jpg_path}")
#         else:
#             raise ValueError("No valid 'exr' or 'depth' field in message")

#     except Exception as e:
#         print(f"❌ Failed to handle depth image: {e}")
#         await websocket.send(json.dumps({
#             "type": "error",
#             "message": f"Depth handling failed: {str(e)}"
#         }))
        
        
async def handle_rgbd(data, websocket):
    try:
        os.makedirs("received", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 检查字段名（与 Unity 代码一致）
        if 'color' in data and 'depth' in data:
            color_bytes = base64.b64decode(data['color'])
            depth_bytes = base64.b64decode(data['depth'])

            rgb_path = f"received/rgb.png"
            depth_path = f"received/depth.png"

            with open(rgb_path, 'wb') as f:
                f.write(color_bytes)
            with open(depth_path, 'wb') as f:
                f.write(depth_bytes)
            color_img = Image.open(io.BytesIO(color_bytes))
            depth_img = Image.open(io.BytesIO(depth_bytes))

            shared_state.rgb_array = np.array(color_img)
            shared_state.depth_array = np.array(depth_img)

            # print(f"✅ RGBD 图像已保存：\n  RGB: {rgb_path}\n  Depth: {depth_path}")
        else:
            raise ValueError("缺少 'color' 或 'depth' 字段")

    except Exception as e:
        print(f"❌ 接收 RGBD 失败: {e}")
        # await websocket.send(json.dumps({
        #     "type": "error",
        #     "message": f"RGBD 处理失败: {str(e)}"
        # }))

        
async def handle_json(data, websocket):
    try:
        os.makedirs("received", exist_ok=True)

        # 生成文件名: 形如 received/{json_type}_{时间戳}.json
        json_type = data["json_type"]
        filename = f"received/{json_type}.json"

        # 写入json文件，确保内容格式正确
        content = data.get("content", {})
        # with open(filename, "w", encoding="utf-8") as f:
        #     json.dump(content, f, ensure_ascii=False, indent=2)
        
        if json_type == "TransformData":
            shared_state.transform_data = TransformData.from_dict(content)
        elif json_type == "Instruction":
            shared_state.instruction = content
        elif json_type == "Init":
            shared_state.Init = True

        #logging.info(f"📦 JSON content saved to {filename}")

        await websocket.send(json.dumps({
            "type": "ack",
            "message": f"JSON content saved as {filename}"
        }))
    except Exception as e:
        logging.error(f"❌ Failed to save JSON content: {e}")
        # await websocket.send(json.dumps({
        #     "type": "error",
        #     "message": f"Saving JSON failed: {str(e)}"
        # }))
        

