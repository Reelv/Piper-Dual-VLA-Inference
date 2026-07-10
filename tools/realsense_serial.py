import pyrealsense2 as rs

def find_connected_realsense_devices():
    # 创建上下文对象
    ctx = rs.context()
    
    # 获取所有连接的设备
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("No RealSense device detected.")
        return
    
    print(f"Detected {len(devices)} RealSense device(s).")
    
    for i, dev in enumerate(devices):
        serial_number = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        physical_port = dev.get_info(rs.camera_info.physical_port)
        
        print(f"\n device {i + 1}:")
        print(f"  name: {name}")
        print(f"  serial: {serial_number}")
        print(f"  port: {physical_port}")

if __name__ == "__main__":
    find_connected_realsense_devices()