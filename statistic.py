import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

df = pd.read_csv('./data/5d8cd542766c880017188948_2021_12_28_06_16_26_final.csv')

motorcycle = df.iloc[:, 2].values 
car = df.iloc[:, 3].values 
bus = df.iloc[:, 4].values 
truck = df.iloc[:, 5].values 

fig, ax = plt.subplots()
fig.canvas.draw()
step = 20
ax.plot(np.arange(2466, step=step), truck[np.arange(2466, step=step)])
ax.set_xticks([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400])
ax.set_xticklabels(['6h', '7h', '8h', '9h', '10h', '11h', '12h', '13h', '14h', '15h', '16h', '17h', '18h'])
ax.set_yticks([0,1,2,3, 4])
plt.title('Biểu Đồ Lưu Lượng Xe Tải \nTại Nút Giao Võ Văn Ngân - Đăng Văn Bi Ngày 28/12/2021')
plt.xlabel('Thời Gian')
plt.ylabel('Số Lượng')
plt.show()