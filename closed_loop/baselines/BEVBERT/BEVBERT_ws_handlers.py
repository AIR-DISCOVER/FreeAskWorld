# ws_handlers.py
import base64
import io
import os
import json
import logging
from PIL import Image
from datetime import datetime
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from messages import *
from ws_handlers import *
import shared_state

import time
sys.path.append('/home/wuyou/yxy/VLN-BEVBert/bevbert_ce')  # 注意路径是模块目录，而不是模块本身
from bevbert_inference_service import BEVBertInferenceService, BEVBertConfig
import numpy as np
import time
logging.basicConfig(level=logging.INFO)

bev_bert_service = None

def bev_bert_init():

    # 创建配置
    config = BEVBertConfig(
        use_real_model=True,
        checkpoint_path="data/logs/checkpoints/bevbert/best_model.pth",
        output_dir="my_bevbert_outputs",
        save_results=True,
        enable_callbacks=True,
        device="cpu"  # 🔑 重要：添加这个参数避免设备问题
    )

    # 创建服务
    service = BEVBertInferenceService(config=config)

    # 注册回调
    def on_complete(result):
        print(f"BEVBert推理完成: {result.content}")

    service.register_callback('on_inference_complete', on_complete)

    # 启动服务
    service.start_service()

    # 🔑 重要：等待服务完全启动
    time.sleep(2)
    return service

def register_bevbert(instance):
    global bev_bert_service
    bev_bert_service = instance
    
def bev_bert_cleanup():
    global bev_bert_service
    if bev_bert_service is not None:
        try:
            print("🧹 正在清理 ETPNav 服务...")
            bev_bert_service.stop_service()  # 停止服务
            del bev_bert_service             # 删除变量引用（可选）
            bev_bert_service = None          # 显式释放
            print("✅ ETPNav 服务已清理")
        except Exception as e:
            print(f"⚠️ 清理失败: {e}")


async def bev_bert_service_handle(websocket, path):
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
                if msg_type == "rgbd":
                    await handle_rgbd(data, websocket)
                elif msg_type == "json":
                    await handle_json(data, websocket)
                else:
                    await handle_unknown(data, websocket)
                
                if(shared_state.Init):
                    bev_bert_cleanup()
                    bev_bert = bev_bert_init()
                    register_bevbert(bev_bert)  # 注册给 handler
                    shared_state.Init = False
                # 等待注册完成
                while bev_bert_service is None:
                    logging.info("等待 etpnav 初始化完成")
                    time.sleep(1)    
                cur_rgb_array = shared_state.rgb_array
                cur_depth_array = shared_state.depth_array
                cur_transform_data = shared_state.transform_data
                cur_instruction = shared_state.instruction
                if cur_rgb_array is not None and cur_depth_array is not None and cur_transform_data is not None and cur_instruction is not None:
                    # 只保留 RGB
                    if cur_rgb_array.shape[2] > 3:
                        cur_rgb_array = cur_rgb_array[:, :, :3]
                    cur_rgb_array = np.transpose(cur_rgb_array, (2, 0, 1))  # HWC -> CHW
                    cur_rgb_array = np.expand_dims(cur_rgb_array, axis=0)   # -> NCHW
                    # 发送推理请求
                    inference_data = {
                        'rgb': cur_rgb_array,
                        'depth': cur_depth_array,
                        'instruction': cur_instruction
                    }
                    future = bev_bert_service.inference(inference_data)
                    result = future.result()
                    shared_state.clear_shared_state()   # 使用后清空缓存全局数据
                    if result:
                        navigation_cmd = NavigationCommand.from_dict(result.content)
                        output_response = {
                            "type": "json",
                            "json_type": "NavigationCommand",
                            "content": navigation_cmd.to_dict()
                        }
                        step = Step(IsStep=True)
                        step_response = {
                            "type": "json",
                            "json_type": "Step",
                            "content": step.to_dict()
                        }
                        try:
                            await websocket.send(json.dumps(output_response))
                            await websocket.send(json.dumps(step_response))
                        except Exception as send_err:
                            logging.error(f"❌ Send error to {client_ip}: {send_err}")
                        print(f"✅ 获取结果成功")
                    else:
                        print("Inference Time Out")

            except json.JSONDecodeError:
                logging.error(f"❌ Invalid JSON from {client_ip}: {message}")
                error_msg = {"type": "error", "message": "Invalid JSON format"}
                await websocket.send(json.dumps(error_msg))
    except Exception as e:
        logging.error(f"❌ Error with client {client_ip}: {e}")
    finally:
        logging.info(f"❎ Client disconnected: {client_ip}")

