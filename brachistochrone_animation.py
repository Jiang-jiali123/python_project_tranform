import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

def cycloid(t, start=(0,0), end=(2,1)):
    # 缩放摆线以匹配起点和终点
    scale_x = end[0] / (2.8 - np.sin(2.8))
    scale_y = end[1] / (1 - np.cos(2.8))
    x = scale_x * (t - np.sin(t))
    y = scale_y * (1 - np.cos(t))
    return x, y

def straight_line(x, start=(0,0), end=(2,1)):
    return start[1] + (end[1] - start[1]) * (x - start[0]) / (end[0] - start[0])

def circular_arc(x, start=(0,0), end=(2,1)):
    # 计算圆弧
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    R = (dx**2 + dy**2) / (4 * dy)
    center_x = (start[0] + end[0]) / 2
    center_y = start[1] - R
    return center_y + np.sqrt(R**2 - (x - center_x)**2)

class BrachistochroneAnimation:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_xlim(-0.5, 2.5)
        self.ax.set_ylim(-0.5, 1.5)  # 调整y轴范围使向下为正
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        
        # 定义起点和终点
        self.start = (0, 0)
        self.end = (2, 1)
        
        # 创建路径
        self.x = np.linspace(0, self.end[0], 100)
        self.t = np.linspace(0, 2.8, 100)
        self.cycloid_x, self.cycloid_y = cycloid(self.t, self.start, self.end)
        self.straight_y = [straight_line(x, self.start, self.end) for x in self.x]
        self.arc_y = [circular_arc(x, self.start, self.end) for x in self.x]
        
        # 绘制路径
        self.ax.plot(self.x, self.straight_y, 'b-', label='直线')
        self.ax.plot(self.x, self.arc_y, 'g-', label='圆弧')
        self.ax.plot(self.cycloid_x, self.cycloid_y, 'r-', label='摆线(最速降线)')
        
        # 创建小球
        self.balls = [
            Circle((0, 0), 0.05, fc='blue'),  # 直线上的球
            Circle((0, 0), 0.05, fc='green'),  # 圆弧上的球
            Circle((0, 0), 0.05, fc='red')    # 摆线上的球
        ]
        for ball in self.balls:
            self.ax.add_patch(ball)
            
        self.ax.legend()
        self.ax.set_title('最速降线与其他下降路径对比')

    def init(self):
        for ball in self.balls:
            ball.center = (0, 0)
        return self.balls

    def animate(self, frame):
        progress = frame / 100
        
        # 直线上的球
        x_straight = self.end[0] * progress
        self.balls[0].center = (x_straight, straight_line(x_straight, self.start, self.end))
        
        # 圆弧上的球
        x_arc = self.end[0] * progress
        self.balls[1].center = (x_arc, circular_arc(x_arc, self.start, self.end))
        
        # 摆线上的球
        t = 2.8 * progress
        x, y = cycloid(t, self.start, self.end)
        self.balls[2].center = (x, y)
        
        return self.balls

    def show(self):
        anim = FuncAnimation(self.fig, self.animate, init_func=self.init,
                           frames=100, interval=50, blit=True)
        plt.show()

if __name__ == '__main__':
    animation = BrachistochroneAnimation()
    animation.show()
