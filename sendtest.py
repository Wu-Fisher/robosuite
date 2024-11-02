import socket
import time
import random

# 定义服务器的 IP 地址和端口
HOST = '127.0.0.1'  # 目标主机（这里是本地主机）
PORT = 65432        # 目标端口号

# 创建 UDP 套接字

if __name__ == "__main__":
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        while True:
            # 生成随机的 XYZ 加速度数据
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            z = random.uniform(-1, 1)
            
            # 将数据打包成字符串并发送
            data = f"{x},{y},{z}\n"
            s.sendto(data.encode('utf-8'), (HOST, PORT))
            
            # 打印发送的数据
            print(f"Sent: {data.strip()}")
            
            # 模拟数据生成的间隔
            time.sleep(0.05)  # 每 50 毫秒发送一次数据


def send_acc_data(acc_data):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        x, y, z = acc_data
        data = f"{x},{y},{z}\n"
        s.sendto(data.encode('utf-8'), (HOST, PORT))