import json
import random
from collections import Counter

data_list = "/root/DSR/wenet/kespeech/data/Subdialects_few/train_phase1/data.list"
output_data_list = (
    "/root/DSR/wenet/kespeech/data/Subdialects_few/train_phase1/reduced_data.list"
)
output_wav_scp = (
    "/root/DSR/wenet/kespeech/data/Subdialects_few/train_phase1/reduced_wav.scp"
)
output_text = "/root/DSR/wenet/kespeech/data/Subdialects_few/train_phase1/reduced_text"


def reduce_data(data, reduce_rate):
    # 提取所有的 subdialect
    subdialect_values = [entry["subdialect"] for entry in data]
    # 统计每种 subdialect 的出现次数
    subdialect_counts = Counter(subdialect_values)

    reduced_data = []
    for subdialect, count in subdialect_counts.items():
        print(f"Subdialect: {subdialect}, Count: {count}")
        # 筛选出方言类的数据
        subdialect_data = [entry for entry in data if entry["subdialect"] == subdialect]
        target_count = int(count * reduce_rate)
        # 如果方言数据数量大于target_count，则进行随机抽样
        if len(subdialect_data) > target_count:
            subdialect_data = random.sample(subdialect_data, target_count)
        reduced_data += subdialect_data

    return reduced_data


if __name__ == "__main__":
    # 读取你的数据文件，这里假设数据是保存在 JSON 格式的文件中
    with open(data_list, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # 计算权重
    # weight = {}
    # print(subdialect_counts)
    # for subdialect, count in subdialect_counts.items():
    #     weight[subdialect] = 1.0 / count
    # print(weight)

    rate = 0.01
    reduced_data = reduce_data(data, rate)

    print(f"Reduced data count: {len(reduced_data)}")

    # 生成wav.scp
    with open(output_wav_scp, "w", encoding="utf-8") as f:
        for entry in reduced_data:
            f.write(f"{entry['key']} {entry['wav']}\n")

    # 生成text
    with open(output_text, "w", encoding="utf-8") as f:
        for entry in reduced_data:
            f.write(f"{entry['key']} {entry['txt']}\n")

    # 生成data.list
    with open(output_data_list, "w", encoding="utf-8") as f:
        for entry in reduced_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")
