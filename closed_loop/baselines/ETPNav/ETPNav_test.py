import sys
import time
sys.path.append('/home/wuyou/yizhou/ETPNav')  # 注意路径是模块目录，而不是模块本身

from configurable_etpnav_service import *

def trigger_inference(service, rgb, depth, instruction, pos, rot):
    request_id = service.send_inference_request(
        # rgb_image=rgb,
        # depth_image=depth,
        # instruction="navigate to kitchen",
        # position=[0, 0, 0],
        # rotation=[0, 0, 0, 1]
        rgb_image=rgb,
        depth_image=depth,
        instruction=instruction,
        position=pos,
        rotation=rot
    )
    
    return request_id
    
def get_results(service, request_id):
    # 获取结果
    result = service.get_inference_result(request_id)
    if result:
        print(f"✅ 获取结果成功")
    else:
        print("Inference Time Out")
    
    
def main():
    """主测试函数 - 展示真实推理和瞬时信号"""
    print("🚀 真实ETPNav推理服务测试")
    print("=" * 60)

    # 创建服务（尝试加载真实模型）
    service = ConfigurableETPNavService(config_file="/home/wuyou/pengyh/FreeAskWorld/baselines/etpnav_config.yaml")

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
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.zeros((480, 640), dtype=np.uint8)
    # 第一次推理触发模型启动
    request_id = trigger_inference(service, rgb, depth, "turn right to the moon", [0, 0, 0], [0, 0, 0, 1])
    get_results(service, request_id)
    time.sleep(10)
    request_id = trigger_inference(service, rgb, depth, "turn right to the moon", [0, 0, 0], [0, 0, 0, 1])
    get_results(service, request_id)
    #service.stop_service()

if __name__ == "__main__":
    
    main()