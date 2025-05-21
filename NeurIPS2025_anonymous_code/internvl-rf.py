import os
import base64
import io
import re
import json
from PIL import Image, ImageFile
from openai import OpenAI
from tqdm import tqdm
import math # 用于坐标裁剪
import copy # To safely copy input data
import tempfile # <-- Import tempfile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- 配置参数 (保持不变) ---
API_KEY = "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiIyNjIwOTgxOCIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc0NzEyNTQ3OSwiY2xpZW50SWQiOiJlYm1ydm9kNnlvMG5semFlazF5cCIsInBob25lIjoiMTMwNjQ2MzM2MzAiLCJvcGVuSWQiOm51bGwsInV1aWQiOiI1MjQzZjY1OS03MjQwLTRiOTAtYThjNi0xNDdlMDFiOTY1YjciLCJlbWFpbCI6IiIsImV4cCI6MTc2MjY3NzQ3OX0.MnxdXY2FGnSoHILq-PT8mJZOPhQaGM7wXqYD1MKoDoC5U_HFwA5cJKX7s5wRB_4ZA2o4AweRX-CdQSIfH3oOqA" # 请替换成你的 API Key
BASE_URL = "https://chat.intern-ai.org.cn/api/v1/"
MODEL_NAME = "internvl3-latest"
DATASET_PATH = "/data/YUXUAN/Z_cot/AD/Reasoning/AD_Reasoning_convert.json"
OUTPUT_PATH = "/data/YUXUAN/Z_cot/AD/Reasoning/AD_Reasoning_vl3_wcot.json" # 修改输出文件名
IMAGE_BASE_PATH = "/data/YUXUAN/datasets/MME-HD-CN"
TEMPERATURE = 0

MAX_WIDTH = 4096
MAX_HEIGHT= 4096
MAX_SIZE_BYTES = 7 * 1024 * 1024 # 7MB

# --- 辅助函数 ---
# (encode_image_bytes, compress_image_to_memory, parse_bbox, expand_bbox, parse_final_answer, extract_letter_choice, parse_rechecked_answer, crop_image_from_bbox, call_llm_api 保持不变)
def encode_image_bytes(image_bytes):
    if image_bytes is None:
        return None
    return base64.b64encode(image_bytes).decode('utf-8')

def compress_image_to_memory(image_path, max_size_bytes=MAX_SIZE_BYTES, initial_quality=85, min_quality=10, max_width=MAX_WIDTH, max_height=MAX_HEIGHT):
    """ (代码保持不变) """
    try:
        with Image.open(image_path) as img:
            original_width, original_height = img.size
            needs_resize = original_width > max_width or original_height > max_height
            current_img = img
            if needs_resize:
                 print(f"缩放图像 {os.path.basename(image_path)} 从 {original_width}x{original_height}...")
                 resampling_method = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
                 current_img = img.copy()
                 current_img.thumbnail((max_width, max_height), resampling_method)
                 print(f"缩放后尺寸: {current_img.size}")

            if current_img.mode in ('RGBA', 'P'):
                # print(f"转换图像 {os.path.basename(image_path)} 为 RGB...")
                if current_img is img:
                     current_img = img.copy()
                current_img = current_img.convert('RGB')

            temp_bytes_io = io.BytesIO()
            # Use optimize=True and maybe progressive=True for potentially smaller JPEGs
            current_img.save(temp_bytes_io, format='JPEG', quality=initial_quality, optimize=True) # Removed progressive=True for broader compatibility if needed
            current_size = len(temp_bytes_io.getvalue())
            temp_bytes_io.close() # Close the BytesIO object

            if current_size > max_size_bytes:
                print(f"图像 {os.path.basename(image_path)} 初始大小 (quality {initial_quality}): {current_size / (1024 * 1024):.2f} MB. 开始压缩...")
                quality = initial_quality
                compressed_image_bytes = io.BytesIO()
                current_img.save(compressed_image_bytes, format='JPEG', quality=quality, optimize=True)
                compressed_size = len(compressed_image_bytes.getvalue())

                while compressed_size > max_size_bytes and quality > min_quality:
                    quality -= 5
                    # Important: Re-create BytesIO for each save attempt
                    compressed_image_bytes.seek(0) # Reset stream pointer
                    compressed_image_bytes.truncate() # Clear previous content
                    current_img.save(compressed_image_bytes, format='JPEG', quality=quality, optimize=True)
                    compressed_size = len(compressed_image_bytes.getvalue())
                    # print(f"降低质量到 {quality}. 新大小: {compressed_size / (1024 * 1024):.2f} MB")

                if compressed_size > max_size_bytes:
                    print(f"警告: 无法将图像 {os.path.basename(image_path)} 压缩到 {max_size_bytes / (1024 * 1024):.2f}MB 以下 (最低质量 {min_quality})。")
                    # Fallthrough to return the current best attempt

                final_bytes = compressed_image_bytes.getvalue()
                compressed_image_bytes.close() # Close the final BytesIO object
                # print(f"图像 {os.path.basename(image_path)} 最终压缩大小: {len(final_bytes) / (1024 * 1024):.2f} MB")
                # Return original dimensions alongside the compressed bytes
                return final_bytes, original_width, original_height
            else:
                # print(f"图像 {os.path.basename(image_path)} 在质量 {initial_quality} 下大小为 {current_size / (1024 * 1024):.2f} MB，在限制内。")
                final_bytes_io = io.BytesIO()
                current_img.save(final_bytes_io, format='JPEG', quality=initial_quality, optimize=True)
                final_bytes = final_bytes_io.getvalue()
                final_bytes_io.close()
                return final_bytes, original_width, original_height # Return original dimensions

    except FileNotFoundError:
        print(f"错误: 图像文件未找到于 {image_path}")
        return None, None, None
    except Image.UnidentifiedImageError:
         print(f"错误: 无法识别图像文件: {image_path}")
         return None, None, None
    except Exception as e:
        print(f"错误: 处理图像时发生异常 {image_path}: {e}")
        return None, None, None

def parse_bbox(response_text):
    """ 解析 BBox (保持不变) """
    regex = r"\s*(?:\*{2})?(?:Bounding box|bbox)(?:\*{2})?\s*[:：]?\s*(?:\*{2})?\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]"
    matches = re.findall(regex, response_text, re.IGNORECASE)
    if not matches:
        return None
    last_match_coords_str = matches[-1]
    try:
        coords = [float(c) for c in last_match_coords_str]
        if len(coords) == 4 and coords[0] < coords[2] and coords[1] < coords[3]:
             coords_clamped = [
                 max(0.0, min(1.0, coords[0])),
                 max(0.0, min(1.0, coords[1])),
                 max(0.0, min(1.0, coords[2])),
                 max(0.0, min(1.0, coords[3])),
             ]
             if coords_clamped[0] >= coords_clamped[2]: coords_clamped[2] = coords_clamped[0] + 1e-6
             if coords_clamped[1] >= coords_clamped[3]: coords_clamped[3] = coords_clamped[1] + 1e-6
             coords_final = [min(1.0, c) for c in coords_clamped]
             if coords_final[0] < coords_final[2] and coords_final[1] < coords_final[3]:
                 return coords_final
             else:
                 print(f"警告: 修正后的坐标仍然无效: {coords_final}")
                 return None
        else:
            print(f"警告: 最后一个解析得到的坐标无效或顺序错误: {coords}")
            return None
    except ValueError:
        print(f"错误: 转换最后一个 BBox 的坐标值为浮点数失败: {last_match_coords_str}")
        return None

def expand_bbox(bbox_norm, padding=0.1):
    """ (代码保持不变) """
    if not (isinstance(bbox_norm, list) and len(bbox_norm) == 4):
        print("错误: expand_bbox 接收到无效的输入 bbox_norm")
        return None
    if not (0.0 <= padding <= 1.0):
         print(f"警告: expand_bbox 的 padding 值 ({padding}) 无效或过大，将使用 0.1。")
         padding = 0.1 # 使用默认值

    x1, y1, x2, y2 = bbox_norm

    # 计算扩展后的坐标
    new_x1 = x1 - padding
    new_y1 = y1 - padding
    new_x2 = x2 + padding
    new_y2 = y2 + padding

    # 将坐标限制在 [0.0, 1.0] 范围内
    final_x1 = max(0.0, new_x1)
    final_y1 = max(0.0, new_y1)
    final_x2 = min(1.0, new_x2)
    final_y2 = min(1.0, new_y2)

    # 确保扩展后仍然有效 (x1 < x2, y1 < y2)
    if final_x1 >= final_x2:
        # print(f"警告: 扩展后 x1 ({final_x1:.3f}) >= x2 ({final_x2:.3f})，可能原始框太小或padding太大。将尝试调整。")
        mid_x = (x1 + x2) / 2
        final_x1 = max(0.0, mid_x - (x2-x1)/2 - padding/2) # 稍微保守点
        final_x2 = min(1.0, mid_x + (x2-x1)/2 + padding/2)
        if final_x1 >= final_x2: # 如果还是不行
             print("错误: 调整后 x 坐标仍然无效，扩展失败。")
             return None # 或者返回原始 bbox_norm

    if final_y1 >= final_y2:
        # print(f"警告: 扩展后 y1 ({final_y1:.3f}) >= y2 ({final_y2:.3f})，可能原始框太小或padding太大。将尝试调整。")
        mid_y = (y1 + y2) / 2
        final_y1 = max(0.0, mid_y - (y2-y1)/2 - padding/2)
        final_y2 = min(1.0, mid_y + (y2-y1)/2 + padding/2)
        if final_y1 >= final_y2:
             print("错误: 调整后 y 坐标仍然无效，扩展失败。")
             return None # 或者返回原始 bbox_norm


    expanded_coords = [final_x1, final_y1, final_x2, final_y2]
    print(f"原始 BBox: [{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}], 扩展后 BBox: [{expanded_coords[0]:.3f}, {expanded_coords[1]:.3f}, {expanded_coords[2]:.3f}, {expanded_coords[3]:.3f}]")
    return expanded_coords

def parse_final_answer(response_text):
    """ (代码保持不变) """
    regex = r"\s*(?:\*{2})?Answer(?:\*{2})?\s*[:：]\s*(.*)"
    match = re.search(regex, response_text, re.IGNORECASE | re.DOTALL)
    if match:
        final_answer_text = match.group(1).strip()
        return final_answer_text
    return None

def extract_letter_choice(response_text):
    """ (代码保持不变) """
    if not response_text: return None
    prefix_regex = r"(?:Answer|Rechecked Answer)\s*[:：]\s*"
    match_prefix_paren = re.search(prefix_regex + r"\(?\s*([A-E])\s*\)?", response_text, re.IGNORECASE | re.DOTALL)
    if match_prefix_paren:
        return match_prefix_paren.group(1).upper()
    match_prefix_letter = re.search(prefix_regex + r"([A-E])\b", response_text, re.IGNORECASE | re.DOTALL)
    if match_prefix_letter:
        return match_prefix_letter.group(1).upper()
    # match_paren = re.search(r"\(\s*([A-E])\s*\)", response_text, re.IGNORECASE)
    # if match_paren:
    #     return match_paren.group(1).upper()
    # match_isolated = re.search(r"(?<![a-zA-Z])([A-E])(?![a-zA-Z])", response_text, re.IGNORECASE)
    # if match_isolated:
    #     preceding_text = response_text[:match_isolated.start()].lower()
    #     if "My Answer:" not in preceding_text[-20:] and "rechecked answer:" not in preceding_text[-20:]:
    #        return match_isolated.group(1).upper()
    return None

def parse_rechecked_answer(response_text):
    """ (代码保持不变) """
    if not response_text: return None
    match = re.search(r"(?:\*{2})?Rechecked Answer(?:\*{2})?\s*[:：]\s*\(?([A-E])\)?", response_text, re.IGNORECASE | re.DOTALL)
    if match:
        letter = match.group(1).upper()
        return letter
    return None

def crop_image_from_bbox(original_image_path, bbox_norm, original_width, original_height):
    """ (代码保持不变) """
    try:
        with Image.open(original_image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')

            x1_norm, y1_norm, x2_norm, y2_norm = bbox_norm
            x1_pix = max(0, min(original_width - 1, math.floor(x1_norm * original_width)))
            y1_pix = max(0, min(original_height - 1, math.floor(y1_norm * original_height)))
            x2_pix = max(0, min(original_width, math.ceil(x2_norm * original_width)))
            y2_pix = max(0, min(original_height, math.ceil(y2_norm * original_height)))

            if x1_pix >= x2_pix: x2_pix = x1_pix + 1
            if y1_pix >= y2_pix: y2_pix = y1_pix + 1
            x2_pix = min(original_width, x2_pix)
            y2_pix = min(original_height, y2_pix)

            if x1_pix >= x2_pix or y1_pix >= y2_pix:
                 print(f"错误: 计算得到的裁剪区域无效 ({x1_pix}, {y1_pix}, {x2_pix}, {y2_pix}) (来自 BBox: {bbox_norm})")
                 return None

            cropped_img = img.crop((x1_pix, y1_pix, x2_pix, y2_pix))
            cropped_bytes_io = io.BytesIO()
            cropped_img.save(cropped_bytes_io, format='JPEG', quality=95, optimize=True) # Save initially at high quality
            return cropped_bytes_io.getvalue()
    except FileNotFoundError:
         print(f"错误: 裁剪时未找到原始图像 {original_image_path}")
         return None
    except Exception as e:
        print(f"错误: 裁剪图像时发生异常 {original_image_path}: {e}")
        return None

def call_llm_api(client, model_name, messages, temperature=0):
    """ (代码保持不变) """
    try:
        chat_rsp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=4096
        )
        if chat_rsp.choices and chat_rsp.choices[0].message:
            return chat_rsp.choices[0].message
        else:
            print("错误: API 响应无效或 choices 为空。")
            return None
    except Exception as e:
        print(f"错误: 调用 API 时发生异常: {e}")
        return None


# --- 主处理逻辑 (修改部分) ---

def process_single_entry(entry_index, entry_data, client, model_name, prompts, image_base_path, temperature=0):
    """
    处理单个数据条目，生成包含指定字段的干净输出字典。
    在裁剪前扩展 BBox。
    对裁剪后的图像进行大小/尺寸检查，必要时压缩。
    """
    # print(f"\n--- 处理条目 {entry_index} ---")

    # 1. 初始化输出结构 (保持不变)
    output_result = {
        "messages": copy.deepcopy(entry_data.get("messages", [])),
        "images": copy.deepcopy(entry_data.get("images", [])),
        "Answer": None,
        "Bounding_Box": None,
        "Rechecked Answer": None,
        "Error": None
    }

    # 2. 提取信息和路径 (保持不变)
    try:
        if not output_result["messages"] or not isinstance(output_result["messages"][0].get("content"), str):
             raise ValueError("Invalid 'messages' structure in input.")
        original_question = output_result["messages"][0]["content"]
        phrase_to_remove = "Respond with only the letter (A, B, C, D, or E) of the correct option"
        question_for_logic = original_question.replace(phrase_to_remove, "").strip()
        if not output_result["images"] or not isinstance(output_result["images"][0], str):
             raise ValueError("Invalid 'images' structure in input.")
        relative_image_path = output_result["images"][0]
        original_image_path = os.path.join(image_base_path, relative_image_path)
    except (KeyError, IndexError, TypeError, ValueError) as e:
        print(f"错误: 解析条目 {entry_index} 结构失败: {e}")
        output_result['Error'] = f"Invalid JSON structure or missing data: {e}"
        return output_result

    # 3. 图像处理 (原始图像 - 保持不变)
    image_bytes_for_llm = None
    original_width = None
    original_height = None
    processing_error_message = None
    if not os.path.exists(original_image_path):
        processing_error_message = f"Original image file not found at {original_image_path}"
    else:
        try:
            needs_processing = False
            file_size = os.path.getsize(original_image_path)
            if file_size > MAX_SIZE_BYTES: needs_processing = True
            else:
                 try:
                     with Image.open(original_image_path) as img_check:
                         width_check, height_check = img_check.size
                         original_width, original_height = width_check, height_check
                         if width_check > MAX_WIDTH or height_check > MAX_HEIGHT: needs_processing = True
                 except Image.UnidentifiedImageError: processing_error_message = "Error: Corrupt or unsupported image format"
                 except Exception as img_err: processing_error_message = f"Error: Could not read image dimensions - {img_err}"

            if processing_error_message is None:
                if needs_processing:
                    print(f"条目 {entry_index}: 原始图像需要处理...")
                    processed_bytes, ow, oh = compress_image_to_memory(original_image_path, MAX_SIZE_BYTES, max_width=MAX_WIDTH, max_height=MAX_HEIGHT)
                    image_bytes_for_llm = processed_bytes
                    # Important: Use dimensions returned by compress_image_to_memory
                    # as they are the original dimensions needed for cropping later
                    original_width, original_height = ow, oh
                    if image_bytes_for_llm is None: processing_error_message = "Error: Original image processing/compression failed"
                else:
                    print(f"条目 {entry_index}: 原始图像无需处理...")
                    try:
                        with open(original_image_path, 'rb') as f: image_bytes_for_llm = f.read()
                        # Ensure original dimensions were read earlier or read them now
                        if original_width is None or original_height is None:
                             with Image.open(original_image_path) as img_final_check: original_width, original_height = img_final_check.size
                    except Exception as read_err: processing_error_message = f"Error: Failed to read original image file - {read_err}"; image_bytes_for_llm = None
        except Exception as e: processing_error_message = f"Error: Unexpected error checking original image - {e}"; image_bytes_for_llm = None

    if processing_error_message or image_bytes_for_llm is None or original_width is None or original_height is None:
        error_msg = processing_error_message if processing_error_message else "Image processing failed (unknown reason)"
        print(f"原始图像处理失败，条目 {entry_index} 中止。错误: {error_msg}")
        output_result['Error'] = error_msg
        return output_result

    # 4. Base64 编码 (原始图像 - 保持不变)
    encoded_image_initial = encode_image_bytes(image_bytes_for_llm)
    if encoded_image_initial is None:
        output_result['Error'] = "Base64 encoding failed for initial image"
        return output_result

    # 5. 构建第一次 API 调用 messages (保持不变)
    messages_for_api = [
        {"role": "system", "content": [{"type": "text", "text": prompts['system_prompt_1']}]},
        {"role": "user", "content": [
            {"type": "text", "text": prompts['user_template_1'].format(question=question_for_logic)},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image_initial}"}}
        ]}
    ]

    # print("\n--- 开始第一次 API 调用 ---")
    assistant_message_1_obj = call_llm_api(client, model_name, messages_for_api, temperature)

    if assistant_message_1_obj is None or not assistant_message_1_obj.content:
        output_result['Error'] = "First API call failed or returned empty content"
        return output_result

    response_1_content = assistant_message_1_obj.content
    # print(f"第一次交互响应 (部分): {response_1_content[:1500]}...")
    print(f"第一次交互响应 (部分): {response_1_content}...")

    # 6. 解析第一次响应 (保持不变)
    final_answer_text = parse_final_answer(response_1_content)
    final_answer_letter = extract_letter_choice(response_1_content if final_answer_text else response_1_content)
    bbox_norm_original = parse_bbox(response_1_content)

    # 7. 存储解析到的信息 (保持不变)
    if final_answer_letter: output_result['Answer'] = final_answer_letter
    if bbox_norm_original: output_result['Bounding_Box'] = bbox_norm_original

    # 8. 判断是否需要进行第二次调用 (保持不变)
    if bbox_norm_original is None:
        if not final_answer_letter:
             output_result['Error'] = "Could not parse the Answer letter or Bounding Box from first response."
        # print(f"条目 {entry_index}: 未找到 BBox，跳过第二次调用。")
        return output_result

    # --- 如果有 BBox，则继续 ---

    # 9. 扩展 BBox (保持不变)
    padding_amount = 0.1
    # print(f"条目 {entry_index}: 原始 BBox: {bbox_norm_original}")
    bbox_to_crop = expand_bbox(bbox_norm_original, padding=padding_amount)
    if bbox_to_crop is None:
        print(f"警告: 扩展 BBox 失败，将尝试使用原始 BBox 进行裁剪。")
        bbox_to_crop = bbox_norm_original
    # print(f"条目 {entry_index}: 用于裁剪的 BBox (扩展后): {bbox_to_crop}")

    # 10. 裁剪图像 (使用扩展后的 bbox_to_crop - 保持不变)
    # print(f"条目 {entry_index}: 使用原始尺寸 {original_width}x{original_height} 和扩展后 BBox 进行裁剪...")
    cropped_image_bytes = crop_image_from_bbox(original_image_path, bbox_to_crop, original_width, original_height)

    if cropped_image_bytes is None:
        output_result['Error'] = "Cropping high-resolution image failed (using expanded bbox)"
        print(f"条目 {entry_index}: 裁剪失败，无法进行第二次调用。")
        return output_result

    # --- 新增逻辑：检查并处理裁剪后的图像 ---
    image_bytes_for_llm_step2 = None
    cropped_processing_error = None
    try:
        cropped_size = len(cropped_image_bytes)
        needs_processing_cropped = False

        # 检查大小
        if cropped_size > MAX_SIZE_BYTES:
            needs_processing_cropped = True
            # print(f"条目 {entry_index}: 裁剪图大小 {cropped_size / (1024*1024):.2f} MB 超出限制，需要压缩。")

        # 检查尺寸
        if not needs_processing_cropped:
            try:
                with Image.open(io.BytesIO(cropped_image_bytes)) as img_cropped_check:
                    cropped_width, cropped_height = img_cropped_check.size
                    if cropped_width > MAX_WIDTH or cropped_height > MAX_HEIGHT:
                        needs_processing_cropped = True
                        # print(f"条目 {entry_index}: 裁剪图尺寸 {cropped_width}x{cropped_height} 超出限制，需要缩放和压缩。")
            except Exception as img_err:
                cropped_processing_error = f"Error:无法读取裁剪图尺寸 - {img_err}"

        if cropped_processing_error:
             raise Exception(cropped_processing_error) # 跳到外层 except

        # 如果需要处理 (压缩/缩放)
        if needs_processing_cropped:
            # print(f"条目 {entry_index}: 开始处理裁剪后的图像...")
            temp_path_cropped = None
            try:
                # 创建临时文件来保存裁剪的字节
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_f:
                    temp_f.write(cropped_image_bytes)
                    temp_path_cropped = temp_f.name

                # 使用现有的压缩函数处理临时文件
                compressed_bytes_final, _, _ = compress_image_to_memory(
                    temp_path_cropped,
                    max_size_bytes=MAX_SIZE_BYTES,
                    max_width=MAX_WIDTH,
                    max_height=MAX_HEIGHT
                )

                if compressed_bytes_final is None:
                    cropped_processing_error = "Error: 裁剪图压缩失败"
                    raise Exception(cropped_processing_error)
                else:
                    image_bytes_for_llm_step2 = compressed_bytes_final
                    # print(f"条目 {entry_index}: 裁剪图处理完成，最终大小 {len(image_bytes_for_llm_step2) / (1024*1024):.2f} MB。")

            finally:
                # 确保删除临时文件
                if temp_path_cropped and os.path.exists(temp_path_cropped):
                    try:
                        os.unlink(temp_path_cropped)
                        # print(f"条目 {entry_index}: 临时裁剪图文件 {temp_path_cropped} 已删除。")
                    except Exception as e_unlink:
                        print(f"警告: 删除临时文件 {temp_path_cropped} 失败: {e_unlink}")
        else:
            # 如果不需要处理，直接使用原始裁剪字节
            # print(f"条目 {entry_index}: 裁剪图无需处理。")
            image_bytes_for_llm_step2 = cropped_image_bytes

    except Exception as e_crop_proc:
         cropped_processing_error = f"Error: 处理裁剪图时出错 - {e_crop_proc}"

    # 如果处理裁剪图时出错
    if cropped_processing_error or image_bytes_for_llm_step2 is None:
        error_msg_step2 = cropped_processing_error if cropped_processing_error else "Cropped image processing failed (unknown)"
        output_result['Error'] = error_msg_step2
        print(f"条目 {entry_index}: {error_msg_step2}，无法进行第二次调用。")
        return output_result
    # --- 新增逻辑结束 ---


    # 11. Base64 编码裁剪后的（可能压缩过的）图像
    encoded_cropped_image = encode_image_bytes(image_bytes_for_llm_step2) # 使用处理后的字节
    if encoded_cropped_image is None:
         output_result['Error'] = "Base64 encoding processed cropped image failed"
         print(f"条目 {entry_index}: Base64编码处理后的裁剪图失败，无法进行第二次调用。")
         return output_result

    # 12. 构建第二次 API 调用 messages (保持不变)
    messages_for_api.append(assistant_message_1_obj.model_dump())
    bbox_str = f"[{bbox_norm_original[0]:.3f}, {bbox_norm_original[1]:.3f}, {bbox_norm_original[2]:.3f}, {bbox_norm_original[3]:.3f}]"
    user_content_follow_up = prompts['user_template_follow_up'].format(
        question=original_question, # Still provide original question context
        bbox_coordinates=bbox_str,
    )
    follow_up_user_message = {
         "role": "user",
         "content": [
             {"type": "text", "text": user_content_follow_up},
             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_cropped_image}"}} # Send the processed crop
         ]
     }
    messages_for_api.append(follow_up_user_message)

    # print("\n--- 开始第二次 API 调用 (提供处理后的裁剪图) ---")
    assistant_message_2_obj = call_llm_api(client, model_name, messages_for_api, temperature)

    if assistant_message_2_obj is None or not assistant_message_2_obj.content:
        output_result['Error'] = "Follow-up API call failed or returned empty content"
        print(f"条目 {entry_index}: 第二次API调用失败。")
        return output_result

    response_2_content = assistant_message_2_obj.content
    # print(f"第二次交互响应 (部分): {response_2_content[:1500]}...")
    print(f"第二次交互响应 (部分): {response_2_content}...")

    # 13. 解析第二次响应 (保持不变)
    rechecked_answer_letter = parse_rechecked_answer(response_2_content)

    if rechecked_answer_letter:
        output_result['Rechecked Answer'] = rechecked_answer_letter
        # print(f"解析到 Rechecked Answer Letter: {rechecked_answer_letter}")
    # else:
        # print("未能从第二次响应解析到 Rechecked Answer Letter")

    # 14. 返回最终结果 (保持不变)
    return output_result


# --- 主执行区 (保持不变) ---
if __name__ == "__main__":

    prompts = {
        "system_prompt_1": """You are an advanced image understanding assistant.You will be given an image and a question about it.
""",
        "user_template_1": """ Question: {question}
Your task:
1.**Provide Your Answer:** Examining the image and the question thoroughly, answer the question in the format **"Answer: [Letter]"**, where [Letter] is only the letter (A, B, C, D) of the correct option.
2.**Identify Critical Area:**After output the answer,please **Reasoning** to determine which area most relevant to the question.Then,please determine a bounding box (normalized coordinates, 0 <= x1, y1, x2, y2 <= 1, x1 < x2, y1 < y2) of the area. Ensure the bounding box is large enough to include all the surrounding context that might be relevant to answering the question (such as information about the table row or column, nearby text, interacting objects, relevant background).Then output in the format**Bounding box: [x1, y1, x2, y2]**



""",
        "user_template_follow_up": """I will provide you an extra image which is cropped from the original image.Just treat the newly input cropped image as an additional information to the local detail information.(The cropped image is clearer)
Please examine the cropped image.(The cropped image may not contain the information needed to answer the question,then ignore this cropped image)
Review the original image and combine with the information from the cropped image.    
Identify potential omission of information in visual perception or calculation based on the image and question.
If you think your previous answer is correct,please **Provide Your Reasoning** to explain why and conclude your response with the format **"Rechecked Answer: [Letter]"**, where [Letter] is only the letter (A, B, C, D) of the correct option.
If you think your previous answer is wrong, please **Provide Your Reasoning** to explain why and correct the answer. Conclude your response with the format **"Rechecked Answer: [Letter]"**, where [Letter] is only the letter (A, B, C, D) of the correct option.
"""}
#     prompts = {
#         "system_prompt_1": """You are an advanced image understanding assistant.
# """,
#         "user_template_1": """ Question: {question}
# Your task:
# 1.**Provide Your Answer:** Examining the image and the question thoroughly, answer the question in the format **"Answer: [Letter]"**, where [Letter] is only the letter (A, B, C, D) of the correct option.
# 2.**Identify Critical Area:**After output the answer,**Thinking** to determine which area most relevant to the question.Then,please determine a bounding box (normalized coordinates, 0 <= x1, y1, x2, y2 <= 1, x1 < x2, y1 < y2) of the area. Ensure the bounding box is large enough to include all the surrounding context that might be relevant to answering the question (such as information about the table row or column, nearby text, interacting objects, relevant background).Then output in the format**"Bounding box: [x1, y1, x2, y2]"**



# """,
#         "user_template_follow_up": """I will provide you an extra image which is cropped from the original image.Just treat the newly input cropped image as an additional information to the local detail information.(The cropped image is clearer)
# Please examine the cropped image.(The cropped image may not contain the information needed to answer the question,then ignore this cropped image)
# Review the original image and combine with the information from the cropped image.    
# Identify potential omission of information in visual perception or calculation based on the image and question.
# If you think your previous answer is correct, please **Reasoning** to explain why and conclude your response with the format **"Rechecked Answer: [Letter]"**, where [Letter] is only the letter (A, B, C, D) of the correct option.
# If you think your previous answer is wrong, please **Reasoning** to explain why and correct the answer. Conclude your response with the format **"Rechecked Answer: [Letter]"**, where [Letter] is only the letter (A, B, C, D) of the correct option.
# """}
    


    # --- 初始化客户端 (保持不变) ---
    try:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    except Exception as e:
        print(f"初始化 OpenAI 客户端失败: {e}")
        exit(1)

    # --- 加载数据集 (保持不变) ---
    print(f"加载数据集: {DATASET_PATH}")
    try:
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"错误: 数据集文件未找到 {DATASET_PATH}")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"错误: 无法解码 JSON {DATASET_PATH}: {e}")
        exit(1)
    except Exception as e:
        print(f"加载数据集时发生错误: {e}")
        exit(1)

    if not isinstance(dataset, list):
        print(f"错误: 加载的数据不是列表格式。")
        exit(1)
    print(f"数据集加载成功，共 {len(dataset)} 条。")

    results = []

    # --- 循环处理 (保持不变) ---
    for index, entry in enumerate(tqdm(dataset, desc="Processing entries")):
        if not isinstance(entry, dict):
            print(f"警告: 跳过索引 {index}，因为条目不是字典格式。")
            results.append({"Error": "Skipped non-dictionary entry", "OriginalIndex": index})
            continue

        processed_result = process_single_entry(
            index, entry, client, MODEL_NAME, prompts, IMAGE_BASE_PATH, TEMPERATURE
        )
        results.append(processed_result)

    # --- 保存结果 (保持不变) ---
    print(f"\n处理完成，保存结果到 {OUTPUT_PATH}")
    try:
        output_dir = os.path.dirname(OUTPUT_PATH)
        if output_dir and not os.path.exists(output_dir):
             os.makedirs(output_dir)
             print(f"创建输出目录: {output_dir}")

        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print("结果保存成功。")
    except Exception as e:
        print(f"错误: 保存结果失败: {e}")
        exit(1)

    print("脚本执行完毕。")