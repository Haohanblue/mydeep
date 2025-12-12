# 生成小规模 Taobao UserBehavior 数据（4000 行）
import random, time
from datetime import datetime, timedelta

random.seed(42)

start = datetime(2017, 11, 25, 0, 0, 1)
end = datetime(2017, 12, 3, 23, 59, 59)
span = (end - start).total_seconds()

n_users = 220
n_items = 320
n_rows = 4000

behaviors = ['pv','click','cart','fav','buy']

lines = []
# 保证每个用户至少 2 条交互
for u in range(n_users):
    k = random.randint(2, 6)
    for _ in range(k):
        item = random.randint(0, n_items-1)
        cat = random.randint(0, 1000)
        b = random.choice(behaviors)
        ts = int(time.mktime((start + timedelta(seconds=random.randint(0, int(span)))).timetuple()))
        lines.append(f"{u},{item},{cat},{b},{ts}")

# 补足到 n_rows
while len(lines) < n_rows:
    u = random.randint(0, n_users-1)
    item = random.randint(0, n_items-1)
    cat = random.randint(0, 1000)
    b = random.choice(behaviors)
    ts = int(time.mktime((start + timedelta(seconds=random.randint(0, int(span)))).timetuple()))
    lines.append(f"{u},{item},{cat},{b},{ts}")

content = "\n".join(lines)
open('lightgcn_taobao/data/UserBehavior_gen.py','w', encoding='utf-8').write(content)
