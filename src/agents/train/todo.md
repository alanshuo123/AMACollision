#11/20
image size change from 320*160 to 320*120
ego reward: set negative reward when collision, to train a normal agent


场景：
换一个带红绿灯的Ｔ型路口场景

训练：
１．　三个车的位置可以调换一下，防止过拟合（前提是能够实现碰撞，再考虑过拟合）
２．　

状态空间中相对速度和相对距离的归一化，是不是合理
command应该使用onehot编码