import json
import ijson

input_file = '/data/YUXUAN/MME_RealWorld.json'  # 输入的 JSON 文件路径
output_file = '/data/YUXUAN/Z_cot/AD/Reasoning/AD_Reasoning.json'  # 输出的 JSON 文件路径

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:

    parser = ijson.items(infile, 'item') 

    # Modify the filter condition here
    filtered_data = [
        item for item in parser 
        if (item.get("Subtask") == "Autonomous_Driving") and (item.get("Task") == "Reasoning")
    ]
    
    json.dump(filtered_data, outfile, ensure_ascii=False, indent=4)

print(f"筛选完成，结果已保存到 {output_file}")