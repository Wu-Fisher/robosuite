import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import socket
import threading
import matplotlib.animation as animation
# 设置窗口大小
window_size = 2000

# 使用 deque 来存储动态数据，最大长度为 window_size
data_x = deque([0] * window_size, maxlen=window_size)
data_y = deque([0] * window_size, maxlen=window_size)
data_z = deque([0] * window_size, maxlen=window_size)

# 创建图形和三个子图
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

# 为每个子图创建折线对象
line_x, = ax1.plot(data_x, label='X')
line_y, = ax2.plot(data_y, label='Y')
line_z, = ax3.plot(data_z, label='Z')

# 设置每个子图的边界
ax1.set_ylim(-10, 10)
ax1.set_xlim(0, window_size)
ax1.set_title('X Acceleration')

ax2.set_ylim(-10, 10)
ax2.set_xlim(0, window_size)
ax2.set_title('Y Acceleration')

ax3.set_ylim(-10, 10)
ax3.set_xlim(0, window_size)
ax3.set_title('Z Acceleration')

# 设置图例
ax1.legend()
ax2.legend()
ax3.legend()

# 数据接收线程，负责从网络接收数据
def data_receiver():
    HOST = '127.0.0.1'  # 本地监听的 IP
    PORT = 65432        # 本地监听的端口

    # 创建 UDP 套接字
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind((HOST, PORT))  # 绑定 IP 和端口
        print(f"Listening on {HOST}:{PORT}")
        
        while True:
            # 接收数据
            data, addr = s.recvfrom(1024)  # 从 UDP 套接字接收数据
            data = data.decode('utf-8')
            
            # 解析接收到的 XYZ 数据
            lines = data.strip().split('\n')
            for line in lines:
                try:
                    x, y, z = map(float, line.split(','))
                    
                    # 将数据添加到各自的 deque 中
                    data_x.append(x)
                    data_y.append(y)
                    data_z.append(z)
                except ValueError:
                    print(f"Invalid data received: {line}")

# 更新图表的函数
def update(frame):
    # 更新折线图的数据
    line_x.set_ydata(data_x)
    line_y.set_ydata(data_y)
    line_z.set_ydata(data_z)
    
    return line_x, line_y, line_z

# 创建并启动数据接收线程
receiver_thread = threading.Thread(target=data_receiver)
receiver_thread.daemon = True  # 设置为守护线程，主程序退出时该线程也会退出
receiver_thread.start()

# 创建动画
ani = animation.FuncAnimation(
        fig, update, interval=50, blit=True  # interval 表示更新的间隔时间（毫秒）
    )

# 显示图表
plt.tight_layout()
plt.show()