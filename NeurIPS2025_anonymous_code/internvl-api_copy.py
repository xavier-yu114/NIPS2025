from openai import OpenAI
import os
import base64
from PIL import Image
import io
import json
from tqdm import tqdm
import time # <--- Import the time module

# --- Rate Limiting Settings ---
RATE_LIMIT_PER_MINUTE = 60
SECONDS_PER_MINUTE = 60
# Calculate delay needed between requests
REQUIRED_DELAY = SECONDS_PER_MINUTE / RATE_LIMIT_PER_MINUTE
# Add a small buffer to be safe (e.g., 0.1 seconds)
BUFFER = 0.1
SLEEP_DURATION = REQUIRED_DELAY + BUFFER
print(f"API Rate Limit: {RATE_LIMIT_PER_MINUTE}/min. Applying delay of {SLEEP_DURATION:.2f} seconds between requests.")

# --- Existing Functions (encode_compressed_bytes, encode_image_from_path, compress_image_to_memory) ---
# (Keep your existing functions here, unchanged)
# 定义判断条件
MAX_SIZE_BYTES = 7 * 1024 * 1024
MAX_WIDTH = 4096
MAX_HEIGHT = 4096

# 原有的 base 64 编码函数，用于处理压缩后的字节数据
def encode_compressed_bytes(compressed_picture_bytes):
    """Encodes image bytes (already in memory) to base64."""
    base64_encoded_image = base64.b64encode(compressed_picture_bytes).decode('utf-8')
    return base64_encoded_image

# 新增的直接从文件编码的函数
def encode_image_from_path(image_path):
    """Reads an image file directly and returns its base64 encoded string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image directly from path {image_path}: {e}")
        return None # 返回 None 表示编码失败

# 原有的压缩函数 (保持不变)
def compress_image_to_memory(image_path, max_size_bytes=MAX_SIZE_BYTES, initial_quality=85, min_quality=10, max_width=MAX_WIDTH, max_height=MAX_HEIGHT):
    """
    Compress an image to less than max_size_bytes and keep it in memory without saving to disk.
    Also resizes if dimensions exceed max_width or max_height.

    :param image_path: Path to the original image
    :param max_size_bytes: Target maximum size in bytes
    :param initial_quality: Starting compression quality (1-100)
    :param min_quality: Minimum allowable compression quality (1-100)
    :param max_width: Maximum width of the image
    :param max_height: Maximum height of the image
    :return: Compressed image byte data or None if error
    """
    try:
        with Image.open(image_path) as img:
            original_width, original_height = img.size
            needs_resize = original_width > max_width or original_height > max_height
            if needs_resize:
                 print(f"Resizing image {image_path} from {original_width}x{original_height}...")
                 # Use Image.Resampling.LANCZOS for newer Pillow versions
                 if hasattr(Image, "Resampling"):
                     img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                 else: # Fallback for older Pillow versions
                     img.thumbnail((max_width, max_height), Image.LANCZOS)


            # Ensure image is in RGB format
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')

            temp_bytes_io = io.BytesIO()
            img.save(temp_bytes_io, format='JPEG', quality=initial_quality) # Use initial quality for check
            current_size = len(temp_bytes_io.getvalue())
            temp_bytes_io.close() # Close the temporary buffer
            quality = initial_quality
            compressed_image_bytes = io.BytesIO()
            img.save(compressed_image_bytes, format='JPEG', quality=quality, optimize=True)
            compressed_size = len(compressed_image_bytes.getvalue())
            # Only print if compression is actually starting/needed
            if compressed_size > max_size_bytes:
                 print(f"Initial size at quality {quality}: {compressed_size / (1024 * 1024):.2f} MB. Compressing further...")

            while compressed_size > max_size_bytes and quality > min_quality:
                quality -= 5
                compressed_image_bytes = io.BytesIO() # Create a new BytesIO object for each attempt
                img.save(compressed_image_bytes, format='JPEG', quality=quality, optimize=True)
                compressed_size = len(compressed_image_bytes.getvalue())
                print(f"Reducing quality to {quality}. New size: {compressed_size / (1024 * 1024):.2f} MB")
                # No need to close compressed_image_bytes here, getvalue() works on open buffer

            if compressed_size > max_size_bytes:
                print(f"Warning: Unable to compress image {image_path} to under {max_size_bytes / (1024 * 1024):.2f}MB even at quality {min_quality}.")
                # Still return the oversized bytes, let the calling code decide if it's usable
                # return None # Changed: Return oversized bytes instead of None

            final_bytes = compressed_image_bytes.getvalue()
            compressed_image_bytes.close() # Close the final buffer
            print(f"Final compressed size for {image_path}: {len(final_bytes) / (1024 * 1024):.2f} MB")
            return final_bytes

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Image.UnidentifiedImageError:
         print(f"Error: Cannot identify image file (may be corrupt or unsupported format): {image_path}")
         return None
    except Exception as e:
        print(f"Error compressing image {image_path}: {e}")
        return None


# --- Main Script ---

client = OpenAI(
    api_key="eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiIyOTkwMDI5OCIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc0NzA2MDA5OSwiY2xpZW50SWQiOiJlYm1ydm9kNnlvMG5semFlazF5cCIsInBob25lIjoiMTMwOTQ2NTQ3NjIiLCJvcGVuSWQiOm51bGwsInV1aWQiOiJhMzQzZTQyMC1hOTEwLTQ0NTAtOTQwZS05ZGRlYTJiM2FhMTgiLCJlbWFpbCI6IiIsImV4cCI6MTc2MjYxMjA5OX0.qK5gGer2bi1lBIr4m3vjWVm3RROwCokyi91FPFWIVwVcVySYc8Vispaob8ZmG5GJQ6kf2TwPdfmK3h7h5ZKUyA", 
    base_url="https://chat.intern-ai.org.cn/api/v1/",
)

# Input and Output JSON file paths
input_json_path = '/data/YUXUAN/Z_cot/AD/Reasoning/AD_Reasoning_convert.json'
output_json_path = '/data/YUXUAN/Z_cot/AD/Reasoning/AD_Reasoning_cost.json' # Changed output name slightly
image_base_dir = '/data/YUXUAN/datasets/MME-HD-CN'

# 读取测试集 JSON 文件
try:
    with open(input_json_path, 'r', encoding='utf-8') as f:
        test_set = json.load(f)
except FileNotFoundError:
    print(f"Error: Input JSON file not found at {input_json_path}")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {input_json_path}")
    exit()


# 使用 tqdm 添加进度条处理每个条目
for entry in tqdm(test_set, desc="Processing entries", unit="entry"):
    if 'images' not in entry or not entry['images']:
        print(f"Skipping entry with no images.")
        entry['predicted_answer'] = "Error: No image specified"
        continue # Skip to next entry

    image_filename = entry['images'][0]
    image_path = os.path.join(image_base_dir, image_filename)

    base64_image = None
    processing_error_message = None
    api_request_made = False # Flag to track if we attempt an API call

    # --- Image Processing Logic (Your existing logic) ---
    # 首先检查文件是否存在
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        processing_error_message = "Error: Image file not found"
    else:
        try:
            # 获取文件大小和尺寸
            file_size = os.path.getsize(image_path)
            width, height = 0, 0
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
            except Image.UnidentifiedImageError:
                 print(f"Error: Cannot identify image file (may be corrupt or unsupported): {image_path}")
                 processing_error_message = "Error: Corrupt or unsupported image format"
            except Exception as img_err:
                 print(f"Error opening image {image_path} to get dimensions: {img_err}")
                 processing_error_message = "Error: Could not read image dimensions"

            # 只有在能获取尺寸和大小，且没有其他错误时才继续判断
            if processing_error_message is None:
                # --- 判断是否需要压缩 ---
                if file_size <= MAX_SIZE_BYTES and width <= MAX_WIDTH and height <= MAX_HEIGHT:
                    # 条件满足，直接从文件编码
                    print(f"Image {image_filename} is within limits. Encoding directly.")
                    base64_image = encode_image_from_path(image_path)
                    if base64_image is None:
                         processing_error_message = "Error: Failed to encode image directly"
                else:
                    # 条件不满足 (过大或过宽/高)，需要压缩
                    print(f"Image {image_filename} exceeds limits (Size: {file_size/(1024*1024):.2f}MB, Dim: {width}x{height}). Compressing...")
                    compressed_picture_bytes = compress_image_to_memory(image_path) # 使用原函数压缩
                    if compressed_picture_bytes:
                        # 使用处理压缩后字节的函数进行编码
                        base64_image = encode_compressed_bytes(compressed_picture_bytes)
                        if len(base64_image) * 3 / 4 > MAX_SIZE_BYTES: # Approx check if still too large after compression attempt
                            print(f"Warning: Image {image_filename} still potentially too large after compression attempt. API might reject.")
                            # Decide if you want to set an error or proceed anyway
                            # processing_error_message = "Error: Image too large even after compression"
                    else:
                        # 压缩函数返回了 None
                        print(f"Compression failed for image: {image_filename}")
                        processing_error_message = "Error: Image compression failed"
                # --- 判断结束 ---

        except Exception as e:
            # 处理获取大小或尺寸时可能发生的其他异常
            print(f"An unexpected error occurred while checking image {image_filename}: {e}")
            processing_error_message = f"Error: Unexpected error processing image - {e}"

    # 如果在图片处理阶段出现任何错误
    if processing_error_message:
        entry['predicted_answer'] = processing_error_message
        # No API call was made, so no sleep needed here
        continue # Skip to next entry

    # 安全检查：如果 base64_image 仍然是 None (e.g., encoding failed silently)
    if base64_image is None:
         entry['predicted_answer'] = "Error: Failed to get base64 image data after processing"
         # No API call was made, so no sleep needed here
         continue # Skip to next entry

    # 获取用户问题 (假设结构固定)
    if 'messages' not in entry or not entry['messages'] or 'content' not in entry['messages'][0]:
        print(f"Skipping entry with missing message structure.")
        entry['predicted_answer'] = "Error: Invalid message structure in input JSON"
        # No API call was made, so no sleep needed here
        continue # Skip to next entry
    user_message = entry['messages'][0]['content']

    # 构建 API 请求的消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_message},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }
    ]

    # 发送 API 请求并获取回答
    answer = None # Initialize answer
    try:
        # print(f"Sending API request for {image_filename}...") # Optional: uncomment for debugging
        chat_rsp = client.chat.completions.create(
            model="internvl3-latest",
            messages=messages,
            max_tokens=1024,
            temperature=0     
        )
        answer = chat_rsp.choices[0].message.content
        api_request_made = True # Mark that an API call was attempted/successful
    except Exception as e:
        print(f"Error during API request for image {image_filename}: {e}")
        answer = f"Error: API request failed - {e}"
        # We still attempted an API call, even if it failed, so we should count it towards the rate limit.
        api_request_made = True

    # 将预测回答添加到条目中
    entry['predicted_answer'] = answer

    # --- Add the delay HERE ---
    # Only sleep if an API request was actually attempted in this iteration
    if api_request_made:
        # print(f"Sleeping for {SLEEP_DURATION:.2f} seconds to respect rate limit...") # Optional
        time.sleep(SLEEP_DURATION)


# 保存结果到新的 JSON 文件
try:
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(test_set, f, indent=4, ensure_ascii=False)
    print(f"\nProcessing complete. Results saved to {output_json_path}")
except Exception as e:
    print(f"\nError saving results to {output_json_path}: {e}")




