# 生成 LightGCN 与 SGL 的每 epoch 指标曲线图
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# 确保中文字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC'] + plt.rcParams['font.sans-serif']

# 指标数据（小规模实测 10 epochs）
baseline = {
    'epochs': list(range(1, 11)),
    'recall': [0.0800, 0.0750, 0.0750, 0.0800, 0.0850, 0.0850, 0.0850, 0.0800, 0.0850, 0.0750],
    'ndcg':   [0.0247, 0.0250, 0.0264, 0.0275, 0.0326, 0.0334, 0.0348, 0.0336, 0.0372, 0.0351],
    'hit':    [0.0800, 0.0750, 0.0750, 0.0800, 0.0850, 0.0850, 0.0850, 0.0800, 0.0850, 0.0750],
}
sgl = {
    'epochs': list(range(1, 11)),
    'recall': [0.0600, 0.0650, 0.0600, 0.0550, 0.0600, 0.0550, 0.0500, 0.0600, 0.0700, 0.0700],
    'ndcg':   [0.0201, 0.0213, 0.0189, 0.0174, 0.0218, 0.0210, 0.0193, 0.0214, 0.0236, 0.0241],
    'hit':    [0.0600, 0.0650, 0.0600, 0.0550, 0.0600, 0.0550, 0.0500, 0.0600, 0.0700, 0.0700],
}

# 百分比格式化
pct_fmt = FuncFormatter(lambda v, pos: f"{v*100:.1f}%")

# 绘制 Baseline 曲线
fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
fig.subplots_adjust(bottom=0.28, top=0.88)
fig.suptitle("Baseline：LightGCN 每 Epoch 指标曲线", fontsize=14)
ax.plot(baseline['epochs'], baseline['recall'], marker='o', label='Recall@20')
ax.plot(baseline['epochs'], baseline['ndcg'], marker='s', label='NDCG@20')
ax.plot(baseline['epochs'], baseline['hit'], marker='^', label='HitRate@20')
ax.set_xlabel('Epoch', labelpad=10)
ax.set_ylabel('数值（%）', labelpad=10)
ax.yaxis.set_major_formatter(pct_fmt)
ax.set_xticks(baseline['epochs'])
ax.grid(True, alpha=0.3)
# 底部图例
legend = ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)
plt.savefig('lightgcn_taobao/docs/img/baseline_curves.png', bbox_inches='tight')
plt.close(fig)

# 绘制 SGL 曲线
fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
fig.subplots_adjust(bottom=0.28, top=0.88)
fig.suptitle("LightGCN+SGL 每 Epoch 指标曲线", fontsize=14)
ax.plot(sgl['epochs'], sgl['recall'], marker='o', label='Recall@20')
ax.plot(sgl['epochs'], sgl['ndcg'], marker='s', label='NDCG@20')
ax.plot(sgl['epochs'], sgl['hit'], marker='^', label='HitRate@20')
ax.set_xlabel('Epoch', labelpad=10)
ax.set_ylabel('数值（%）', labelpad=10)
ax.yaxis.set_major_formatter(pct_fmt)
ax.set_xticks(sgl['epochs'])
ax.grid(True, alpha=0.3)
legend = ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)
plt.savefig('lightgcn_taobao/docs/img/sgl_curves.png', bbox_inches='tight')
plt.close(fig)
