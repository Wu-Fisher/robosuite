import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import random
import matplotlib.animation as animation
import threading



if __name__ == "__main__":
    # 设置窗口大小
    window_size = 2000

    # 使用 deque 来存储动态数据，最大长度为 window_size
    data1 = deque(maxlen=window_size)
    data2 = deque(maxlen=window_size)
    data3 = deque(maxlen=window_size)

    # 初始化数据
    for _ in range(window_size):
        data1.append(0)
        data2.append(0)
        data3.append(0)

    # 创建图形和三个子图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))  # 3行1列的子图布局

    # 为每个子图创建折线对象
    line1, = ax1.plot(data1)
    line2, = ax2.plot(data2)
    line3, = ax3.plot(data3)

    # 设置每个子图的边界
    ax1.set_ylim(-1, 1)  # 根据数据范围设置上下边界
    ax1.set_xlim(0, window_size)

    ax2.set_ylim(-1, 1)
    ax2.set_xlim(0, window_size)

    ax3.set_ylim(-1, 1)
    ax3.set_xlim(0, window_size)

    # 更新函数
    def update(frame):
        # 生成新数据并加入到每个 deque 中
        new_value1 = random.uniform(-1, 1)  # 模拟动态数据
        new_value2 = random.uniform(-1, 1)
        new_value3 = random.uniform(-1, 1)
        
        data1.append(new_value1)
        data2.append(new_value2)
        data3.append(new_value3)

        # 更新折线图的数据
        line1.set_ydata(data1)
        line2.set_ydata(data2)
        line3.set_ydata(data3)

        return line1, line2, line3

    # 创建动画
    ani = animation.FuncAnimation(
        fig, update, interval=50, blit=True  # interval 表示更新的间隔时间（毫秒）
    )

    # 调整子图布局
    plt.tight_layout()

    # 显示图表
    plt.show()


class AccDrawer():
    def __init__(self,env,sensor_id,window_size=2000):
        self.window_size = window_size
        self.data1 = deque(maxlen=window_size)
        self.data2 = deque(maxlen=window_size)
        self.data3 = deque(maxlen=window_size)
        for _ in range(window_size):
            self.data1.append(0)
            self.data2.append(0)
            self.data3.append(0)

        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(10, 8))
        self.line1, = self.ax1.plot(self.data1)
        self.line2, = self.ax2.plot(self.data2)
        self.line3, = self.ax3.plot(self.data3)

        self.ax1.set_ylim(-1, 1)
        self.ax1.set_xlim(0, window_size)

        self.ax2.set_ylim(-1, 1)
        self.ax2.set_xlim(0, window_size)

        self.ax3.set_ylim(-1, 1)
        self.ax3.set_xlim(0, window_size)

        self.ax1.set_title('acc_x')
        self.ax2.set_title('acc_y')
        self.ax3.set_title('acc_z')
        self.env = env
        self.sensor_id = sensor_id


    def update(self,frame):
        acc_x,acc_y,acc_z = self.env.sim.data.sensordata[self.sensor_id:self.sensor_id+3]
        self.data1.append(acc_x)
        self.data2.append(acc_y)
        self.data3.append(acc_z)

        self.line1.set_ydata(self.data1)
        self.line2.set_ydata(self.data2)
        self.line3.set_ydata(self.data3)

        return self.line1, self.line2, self.line3
    
    def show(self):
        def run_animation():
                ani = animation.FuncAnimation(
                    self.fig, self.update, interval=50, blit=True
                )
                plt.tight_layout()
                plt.show()

            # 创建并启动新线程
        animation_thread = threading.Thread(target=run_animation)
        animation_thread.start()