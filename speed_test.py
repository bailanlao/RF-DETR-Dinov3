import os
import supervision as sv
from inference import get_model
from PIL import Image
import time
from rfdetr import RFDETRNano, RFDETRBase, RFDETRMedium, RFDETRLarge
import torch

url = "/home/cobot/github_code/RF-DETR/shen_test.png"
image = Image.open(url)
size='large'
if size == 'base':
	model = RFDETRBase(pretrain_weights=f"/home/cobot/github_code/RF-DETR/output-{size}/checkpoint_best_total.pth")
	image = image.resize((332,332))
elif size=='large':
	model = RFDETRLarge(pretrain_weights=f"/home/cobot/github_code/RF-DETR/output-{size}/checkpoint_best_total.pth")
	image = image.resize((332,332))
elif size=='nano':
	model = RFDETRNano(pretrain_weights=f"/home/cobot/github_code/RF-DETR/output-{size}/checkpoint_best_total.pth")
	image = image.resize((320,320))
elif size=='medium':
	model = RFDETRMedium(pretrain_weights=f"/home/cobot/github_code/RF-DETR/output-{size}/checkpoint_best_total.pth")
	image = image.resize((320,320))

model.optimize_for_inference(compile=True, batch_size=1, dtype=torch.float32)

# 1. 预热阶段：运行2次预测，避免首次加载的额外开销影响测速
print("开始预热...")
for _ in range(2):
    model.predict(image, threshold=0.5)
print("预热完成，开始正式测速...\n")

# 2. 正式测速：重复预测1000次，记录每次时间
total_runs = 1000
times = []  # 存储每次预测的时间（秒）

for i in range(total_runs):
    # 记录开始时间
    start_time = time.perf_counter()
    # 执行预测（仅测量predict方法的时间）
    model.predict(image, threshold=0.5)
    # 记录结束时间并计算耗时
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    times.append(elapsed)
    
    # 可选：打印进度（每100次打印一次）
    if (i + 1) % 100 == 0:
        print(f"已完成 {i + 1}/{total_runs} 次预测")

# 3. 计算统计结果
avg_time_sec = sum(times) / total_runs  # 平均时间（秒）
avg_time_ms = avg_time_sec * 1000       # 转换为毫秒
fps = 1 / avg_time_sec                 # FPS（每秒处理帧数）

# 4. 输出结果
print("\n===== 测速结果 =====")
print(f"平均预测时间: {avg_time_ms:.2f} 毫秒/帧")
print(f"推理速度: {fps:.2f} FPS")
detections = model.predict(image, threshold=0.5)
labels = [
    f"{class_id} {confidence:.2f}"
    for class_id, confidence
    in zip(detections.class_id, detections.confidence)
]
annotated_image = image.copy()
annotated_image = sv.BoxAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(annotated_image, detections, labels)
annotated_image.save("./annotated_image.jpg")
