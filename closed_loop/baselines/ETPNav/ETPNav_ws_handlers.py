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
sys.path.append('/home/wuyou/yizhou/ETPNav')  # 注意路径是模块目录，而不是模块本身
from configurable_etpnav_service import *

logging.basicConfig(level=logging.INFO)

etp_nav_service = None


def etp_nav_init():
    """主测试函数 - 展示真实推理和瞬时信号"""
    print("🚀 真实ETPNav推理服务测试")
    print("=" * 60)

    # 创建服务（尝试加载真实模型）
    service = ConfigurableETPNavService(config_file="/home/wuyou/pengyh/FreeAskWorld/baselines/ETPNav/etpnav_config.yaml")

    # 定义回调函数
    def on_inference_start(request_id: str, data: Dict):
        """推理开始时的回调"""
        print(f"\n⚡ [回调] 推理开始: {request_id}")
        print(f"   指令: {data['instruction']}")

    def on_inference_complete(result: InferenceResult):
        """推理完成瞬间的回调"""
        print(f"\n🎯 [回调] 推理完成瞬间信号!")
        print(f"   请求ID: {result.request_id}")
        print(f"   耗时: {result.duration:.3f}秒")
        print(f"   结果: {result.content}")

        # 这里可以立即触发Unity中的动作
        # unity_controller.apply_navigation(result.content)

    def on_status_change(request_id: str, status: InferenceStatus, update: Dict):
        """状态变更回调"""
        print(f"   📊 [状态] {request_id}: {status.value}")

    # 注册回调
    service.register_callback('on_inference_start', on_inference_start)
    service.register_callback('on_inference_complete', on_inference_complete)
    service.register_callback('on_status_change', on_status_change)

    # 启动服务
    service.start_service()
    # rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    # depth = np.zeros((480, 640), dtype=np.uint8)
    # # 第一次推理触发模型启动
    # request_id = trigger_inference(service, rgb, depth, "turn right to the moon", [0, 0, 0], [0, 0, 0, 1])
    # get_results(service, request_id)
    # time.sleep(10)
    #service.stop_service()
    return service

def register_etpnav(instance):
    global etp_nav_service
    etp_nav_service = instance
    
def etp_nav_cleanup():
    global etp_nav_service
    if etp_nav_service is not None:
        try:
            print("🧹 正在清理 ETPNav 服务...")
            etp_nav_service.stop_service()  # 停止服务
            del etp_nav_service             # 删除变量引用（可选）
            etp_nav_service = None          # 显式释放
            print("✅ ETPNav 服务已清理")
        except Exception as e:
            print(f"⚠️ 清理失败: {e}")


async def etpnav_handle(websocket, path):
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
                    etp_nav_cleanup()
                    etpnav = etp_nav_init()
                    register_etpnav(etpnav)  # 注册给 handler
                    shared_state.Init = False
                # 等待注册完成
                while etp_nav_service is None:
                    logging.info("等待 etpnav 初始化完成")
                    time.sleep(1)    
                cur_rgb_array = shared_state.rgb_array
                cur_depth_array = shared_state.depth_array
                cur_transform_data = shared_state.transform_data
                cur_instruction = shared_state.instruction
                if cur_rgb_array is not None and cur_depth_array is not None and cur_transform_data is not None and cur_instruction is not None:
                    request_id = etp_nav_service.send_inference_request(
                        # rgb_image=rgb,
                        # depth_image=depth,
                        # instruction="navigate to kitchen",
                        # position=[0, 0, 0],
                        # rotation=[0, 0, 0, 1]
                        rgb_image=cur_rgb_array,
                        depth_image=cur_depth_array,
                        instruction=cur_instruction,
                        position=cur_transform_data.position, 
                        rotation=cur_transform_data.rotation
                    )
                    shared_state.clear_shared_state()   # 使用后清空缓存全局数据
                    # 获取结果
                    result = etp_nav_service.get_inference_result(request_id, timeout=20.0)
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
