import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 假设freq_dict是一个包含序号和频数的字典
# 例如：freq_dict = {0: 10, 1: 20, ..., 999: 5}
# freq_dict = {i: np.random.randint(1, 100) for i in range(1000)}  # 随机生成频数示例
filename = "20240621_total_params_lfq.json"
freq_dict = json.load(open(filename))
print("read data: ", len(freq_dict))

# 将字典转换为两个数组，一个用于序号，一个用于频数
keys = np.array(list(freq_dict.keys()))
values = np.array(list(freq_dict.values()))
values_sum = np.sum(values)

print("get value: ", keys[0:10], values[0:10], values_sum)

keys = keys[::1000]
values = values[::1000]

# 绘制密度函数图
plt.figure(figsize=(10, 6))
plt.plot(keys, values, label='Density Function')
plt.title('Density Function of the Dictionary')
plt.xlabel('Index')
plt.ylabel('Density')
plt.legend()
plt.grid(True)

# 保存图片
save_file = filename.replace(".json", ".png")
print("save to: ", save_file)
plt.savefig(save_file, dpi=100)