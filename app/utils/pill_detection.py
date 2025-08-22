
# import easyocr
import base64

from rembg import remove
import argparse
import cv2
import numpy as np


# 初始化 OpenOCR 引擎
from openocr import OpenOCR
import logging

from app.utils.image_io import read_image_safely

logging.getLogger("openrec").setLevel(logging.ERROR)
ocr_engine = OpenOCR(backend='onnx', device='cpu')
# ocr_engine = OpenOCR(backend="onnx", det_model_path="models/openocr_det_model.onnx", rec_model_path="models/openocr_rec_model.onnx")


from app.utils.matcher import lcs_score, match_ocr_to_front_back_by_permuted_ocr
from app.utils.ocr_utils import recognize_with_openocr
from app.utils.shape_color_utils import (
    rotate_image_by_angle,
    enhance_contrast,
    desaturate_image,
    enhance_for_blur,
    extract_dominant_colors_by_ratio,
    detect_shape_from_image
)

# Import YOLO
import ultralytics
# 套用字體（用 FontProperties）
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.cluster import KMeans

zh_font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")

import itertools
import pandas as pd
from pathlib import Path
from pillow_heif import register_heif_opener

register_heif_opener()
from collections import defaultdict
from inference_sdk import InferenceHTTPClient
from PIL import Image
from pathlib import Path
import cv2
# import pyheif
import numpy as np
import json, re
from rembg import remove

# from google.colab.patches import cv2_imshow
# === 初始化 ===
###
# CLIENT = InferenceHTTPClient(
#    api_url="https://serverless.roboflow.com",
#    api_key="kylIYUWNLWHPy2RXUVOe"
# )
# MODEL_ID = "ai-drug-analysis-service/3"
###
# create an inference client
# 套用字體（用 FontProperties）

from matplotlib.font_manager import FontProperties


zh_font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


from pillow_heif import register_heif_opener

register_heif_opener()

from inference_sdk import InferenceHTTPClient

import cv2

from rembg import remove
from ultralytics import YOLO
det_model = YOLO("/path/to/best.pt")

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="SOlzinVqG2xuWsPUUGRp"
    # api_key="kylIYUWNLWHPy2RXUVOe"
)
MODEL_ID = "pill-detection-poc-i0b3g/1"


####

def generate_image_versions(base_img):
    v1 = enhance_contrast(base_img, 1.5, 1.5, -0.5)
    v2 = desaturate_image(v1)
    v3 = enhance_contrast(base_img, 5.5, 2.0, -1.0)
    v4 = desaturate_image(v3)
    v5 = enhance_for_blur(base_img)
    return [
        (base_img, "原圖"),
        (v1, "增強1"),
        (v2, "去飽和1"),
        (v3, "增強2"),
        (v4, "去飽和2"),
        (v5, "模糊優化")
    ]


def get_best_ocr_texts(image_versions, angles=[0, 45, 90, 135, 180, 225, 270, 315], ocr_engine=None):
    version_results = {}
    score_dict = {}
    for img_v, version_name in image_versions:
        for angle in angles:
            rotated = rotate_image_by_angle(img_v, angle)
            full_name = f"{version_name}_旋轉{angle}"
            texts, score = recognize_with_openocr(rotated, ocr_engine=ocr_engine, name=full_name, min_score=0.8)

            # print(f"🔍 {full_name} => {texts} (score={score:.3f})")#註解SSS
            version_results[full_name] = texts
            score_dict[full_name] = score

    score_combined = {
        k: sum(len(txt) for txt in version_results[k]) * score_dict[k]
        for k in version_results
    }
    best_name = max(score_combined, key=score_combined.get)
    return version_results[best_name], best_name, score_dict[best_name]


def get_bbox_from_rembg_alpha(img_path):
    input_img = cv2.imread(img_path)
    rembg_img = remove(input_img)

    if rembg_img.shape[2] == 4:
        alpha = rembg_img[:, :, 3]
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        _, mask = cv2.threshold(alpha, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            return rembg_img, (x, y, w, h)  # ➜ return cropped image & bounding box
    return None, None


# === 模組化：從完整圖片與偵測框中擷取藥物區域 ===
# def extract_pill_region(img_path, detection_result, margin=10):
def extract_pill_region(input_img, detection_result, margin=10):
    #    input_img = read_image_safely(img_path)
    if input_img is None:
        # print(f"❌ extract_pill_region: 無法讀取圖片：{img_path}")#註解SSS
        return None, None

    try:
        h_img, w_img = input_img.shape[:2]
        cx, cy = detection_result["x"], detection_result["y"]
        bw, bh = detection_result["width"], detection_result["height"]

        x0 = max(0, int(cx - bw / 2) - margin)
        y0 = max(0, int(cy - bh / 2) - margin)
        x1 = min(w_img, int(cx + bw / 2) + margin)
        y1 = min(h_img, int(cy + bh / 2) + margin)

        cropped_original = input_img[y0:y1, x0:x1]

        try:
            cropped_removed = remove(cropped_original)
        except Exception as e:
            # print(f"❌ rembg 去背失敗：{e}")#註解SSS
            return cropped_original, None

        return cropped_original, cropped_removed

    except Exception as e:
        # print(f"❗ extract_pill_region 錯誤：{e}")#註解SSS
        return None, None


# def fallback_rembg_bounding(img_path):
# input_img = read_image_safely(img_path)
def fallback_rembg_bounding(input_img):
    if input_img is None:
        # print(f"❌ fallback: 無法讀取圖片：{img_path}")#註解SSS
        return None, None

    try:
        rembg_img = remove(input_img)
    except Exception as e:
        print(f"❌ rembg 去背失敗：{e}")
        return None, None

    if rembg_img is None or rembg_img.shape[2] < 4:
        print(f"⚠️ rembg 回傳結果異常")
        return None, None

    try:
        alpha = rembg_img[:, :, 3]
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        _, mask = cv2.threshold(alpha, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            cropped = rembg_img[y:y + h, x:x + w]
            return input_img[y:y + h, x:x + w], cropped  # 返回原圖區塊、去背區塊
        else:
            print("⚠️ fallback 沒有偵測到輪廓")
    except Exception as e:
        print(f"❗ fallback rembg bounding 出錯：{e}")

    return None, None


def test_batch_all_images(ROOT_FOLDER: Path, excel_path: str, start_index=1, end_index=403):
    roboflow_success = 0
    roboflow_total = 0

    shape_total = 0
    shape_success = 0

    total_images = 0
    text_success_total = 0
    shape_success_total = 0

    total_success = 0
    per_drug_stats = defaultdict(lambda: {"total": 0, "success": 0})
    fail_logs = []

    # === 藥物圖片根資料夾 ===
    # ROOT_FOLDER = Path("/content/drive/MyDrive/畢業專題/drug_photo_copy")

    # === 讀取 Excel 資料 ===
    # excel_path = "/content/drive/MyDrive/TEST_DRUG/TESTData.xlsx"
    df = pd.read_excel(excel_path)

    # === 設定你想測試的起始與結束「用量排序」範圍 ===
    start_index = 1
    end_index = 403

    # === 過濾資料表範圍 ===
    df_range = df[(df["用量排序"] >= start_index) & (df["用量排序"] <= end_index)]
    # === 初始化 PRUNE 分類字典 ===
    shape_dict = {"圓形": [], "長圓形": [], "其他": []}
    text_dict = {"has_text": [], "none": []}

    for _, row in df_range.iterrows():
        usage_order = int(row.get("用量排序", -1))

        # === 外型分類 ===
        shape = str(row.get("形狀", "")).strip()
        if shape == "圓形":
            shape_dict["圓形"].append(usage_order)
        elif shape == "長圓形":
            shape_dict["長圓形"].append(usage_order)
        else:
            shape_dict["其他"].append(usage_order)

        # === 文字分類 ===
        text = str(row.get("文字", "")).replace(" ", "").upper()
        if text == "F:NONE|B:NONE":
            text_dict["none"].append(usage_order)
        else:
            text_dict["has_text"].append(usage_order)

    # === 記錄找不到的資料夾 ===
    missing_entries = []

    # === 逐筆處理 ===
    for _, row in df_range.iterrows():
        raw_name = str(row["學名"]).strip()
        drug_name = raw_name.replace("/", " ")
        folder_path = ROOT_FOLDER / drug_name

        if not folder_path.exists():
            # print(f"❌ 找不到資料夾：{folder_path}")
            missing_entries.append({
                "用量排序": row["用量排序"],
                "學名": raw_name
            })
            continue

        # print(f"\n📦 開始辨識藥物：{drug_name}（來源：{folder_path}）")#註解SSS

        # ✅ 你的辨識程式碼貼在這裡（記得 BASE_DIR = folder_path）
        BASE_DIR = folder_path
        exts = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
        # ✅ 過濾掉 NEW_PHOTOS 資料夾 + 變形圖檔名
        skip_keywords = ["_rot", "_bright", "_noise", "_flip", "_removed"]
        img_files = [
            p for p in BASE_DIR.rglob("*")
            if p.suffix.lower() in exts
               and "NEW_PHOTOS" not in p.parts
               and not any(keyword in p.name for keyword in skip_keywords)
        ]

        version_results = {}
        score_dict = {}

        # 每 45 度旋轉一次
        rotation_angles = [0, 45, 90, 135, 180, 225, 270, 315]

        # === 主處理流程 ===

        for img_path in img_files:
            total_images += 1
            # 預期文字與是否應有文字
            # row 是目前圖片對應到的 row（這張圖原始的）
            raw_row = row
            expected_text = str(raw_row.get("文字", "")).strip()
            expected_text_clean = expected_text.replace(" ", "").upper()
            has_expected_text = not (expected_text_clean == "F:NONE|B:NONE")

            try:
                # 讀取圖片（自動判斷是否為 HEIC）
                input_img = read_image_safely(img_path)
                if input_img is None:
                    print(f"⚠️ 無法讀取圖片：{img_path.name}，跳過")
                    continue

                # ✅ Roboflow 推理時改用 PIL.Image.fromarray 轉換
                rgb_pil = Image.fromarray(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
                result = CLIENT.infer(rgb_pil, model_id=MODEL_ID)
                preds = result.get("predictions", [])
                # ➤ 加入統計 Roboflow 成功率
                roboflow_total += 1
                if preds:
                    roboflow_success += 1
                    cropped_original, cropped_removed = extract_pill_region(img_path, preds[0])
                    # print("✅ Roboflow 偵測成功")#註解SSS
                else:
                    # print("⚠️ Roboflow 偵測失敗，改用 rembg fallback")#註解SSS
                    cropped_original, cropped_removed = fallback_rembg_bounding(img_path)
                    if cropped_original is None:
                        print("❌ fallback 也失敗，跳過此圖片")
                        # cv2_imshow(input_img)
                        continue

                # （可選）顯示去背圖
                # cv2_imshow(cropped_removed)#註解SSS
                # 形狀辨識
                shape, result = detect_shape_from_image(cropped_removed, cropped_original, expected_shape=None,
                                                        debug=False)
                ##顏色辨識##
                colors = extract_dominant_colors_by_ratio(cropped_removed, visualize=False)

                # 六個圖片增強版本
                image_versions = generate_image_versions(cropped_removed)
                best_texts, best_name, best_score = get_best_ocr_texts(image_versions)

                is_blurry = (not best_texts) and (best_score < 0.6)

                # print(f"\n🌟 最佳版本：{best_name}")
                # print(f"📋 合併辨識結果：{best_texts}")
                # 若完全無文字辨識，就標記為 NONE

                ocr_is_none = (not best_texts) or (best_score < 0.5)
                ocr_match_result = None  # 初始化結果變數

                if ocr_is_none:
                    # 無文字，直接使用外型與是否無文字條件 PRUNE（不執行 OCR 比對）
                    candidate_orders = list(set(text_dict["none"]) & set(shape_dict.get(shape, [])))
                    pruned_df = df_range[df_range["用量排序"].isin(candidate_orders)] if candidate_orders else df_range
                    ocr_match_result = None  # 無需比對

                else:
                    # ➤ 有文字：先直接比對全部資料
                    ocr_match_result = match_ocr_to_front_back_by_permuted_ocr(best_texts, df_range)

                    # 若分數偏低，則 fallback 進行 PRUNE
                    front_score = ocr_match_result.get("front", {}).get("score", 0) if ocr_match_result else 0
                    back_score = ocr_match_result.get("back", {}).get("score", 0) if ocr_match_result else 0
                    low_score = max(front_score, back_score) < 0.8

                    if low_score:
                        candidate_orders = shape_dict.get(shape, [])
                        pruned_df = df_range[
                            df_range["用量排序"].isin(candidate_orders)] if candidate_orders else df_range
                        ocr_match_result = match_ocr_to_front_back_by_permuted_ocr(best_texts, pruned_df)

                is_correct = False  # 預設為錯

                # ➤ 情境 1：資料本來就沒文字，不需辨識
                if not has_expected_text:
                    # print("📄 資料標註為無文字，略過比對")#註解SSS
                    is_correct = True  # 直接標記為成功
                elif ocr_match_result:
                    if "front" in ocr_match_result:
                        r = ocr_match_result["front"]
                        matched_row = r["row"]  # LCS 對到的藥品
                        expected_name = raw_name.strip().upper()
                        matched_name = matched_row["學名"].strip().upper()
                        is_correct = (expected_name == matched_name)

                        correctness = "🎯 正確辨識" if is_correct else f"❌ 錯誤辨識（預期：{expected_name}）"

                        # print(f"✅ 正面整體比對：{r['text']} → {r['match']} (score={r['score']:.2f})") #註解
                        # print(f"{correctness} | 🔍 學名：{row['學名']}，用量排序：{row['用量排序']}")#註解

                    if "back" in ocr_match_result:
                        r = ocr_match_result["back"]
                        matched_row = r["row"]  # LCS 對到的藥品

                        expected_name = raw_name.strip().upper()
                        matched_name = matched_row["學名"].strip().upper()
                        is_back_correct = (expected_name == matched_name)
                        if not is_correct:  # 如果正面錯了但背面對，依然視為成功
                            is_correct = is_back_correct
                        correctness = "🎯 正確辨識" if is_back_correct else f"❌ 錯誤辨識（預期：{expected_name}）"
                        # print(f"✅ 背面比對：{r['text']} → {r['match']} (score={r['score']:.2f})")#註解
                        # print(f"{correctness} | 🔍 學名：{row['學名']}，用量排序：{row['用量排序']}")#註解
                else:
                    # print("❌ 無法與任何藥物代碼匹配")#註解SSS
                    # ➤ 額外條件：若 OCR 無結果，且某一面剛好是 NONE，則視為成功
                    if ocr_is_none and ("F:NONE" in expected_text or "B:NONE" in expected_text):
                        # print("✅ OCR 雖無結果，但與標註 NONE 對應，視為成功")#註解SSS
                        is_correct = True

                per_drug_stats[raw_name]["total"] += 1
                # ✅ 不管有沒有 expected text，只要 is_correct 就記成功
                if is_correct:
                    total_success += 1
                    per_drug_stats[raw_name]["success"] += 1

                # ✅ 統計總數僅在有標註時計入（避免 NONE 對 NONE 影響基底）
                if has_expected_text:

                    if not is_correct:
                        fail_logs.append({
                            "圖片檔名": img_path.name,
                            "圖片路徑": str(img_path),
                            "藥品學名": raw_name,
                            "用量排序": row["用量排序"],
                            "正確文字": row.get("文字", ""),
                            "正確顏色": row.get("顏色", ""),
                            "OCR結果_文字": best_texts if best_texts else "無文字",
                            "OCR結果_顏色": colors
                        })

                # print(f"📷 圖片：{img_path.name}")#註解SSS
                # print(f"🔠 OCR 結果：{best_texts}（最佳版本：{best_name}，信心分數：{best_score:.3f}）")#註解SSS

                # print(f"🎨 辨識顏色：{colors}")#註解SSS
                # print(f"🟫 辨識外型：{shape}")#註解SSS
                # print(f"📌 正確文字：{expected_text}")#註解SSS
                # print(f"📌 正確顏色：{row.get('顏色', '')}")#註解SSS
                # print(f"📌 正確外型：{row.get('形狀', '')}")#註解SSS

                # 顯示 LCS 最終比對到的藥名

                # print(f"📋 LCS 對應藥名（vs 正確藥名）：")註解SSS
                expected_name = raw_name.strip().upper()
                if ocr_match_result:
                    if "front" in ocr_match_result:
                        matched_name = ocr_match_result["front"]["row"]["學名"].strip().upper()
                    #       print(f" - FRONT 對到：{matched_name} ｜正確：{expected_name}")註解SSS
                    if "back" in ocr_match_result:
                        matched_name = ocr_match_result["back"]["row"]["學名"].strip().upper()
                        # print(f" - BACK  對到：{matched_name} ｜正確：{expected_name}")註解SSS

                # === 新增辨識成功統計 ===
                # ➤ 文字成功：is_correct 代表已成功找到正確藥名
                is_text_success = is_correct

                # ➤ 外型成功：比對預測 shape 與 row['形狀'] 是否一致（忽略空白大小寫）
                expected_shape = str(row.get("形狀", "")).strip().lower()
                is_shape_success = (shape.strip().lower() == expected_shape) if expected_shape else False

                # print(f"✅ 是否正確辨識：{'是' if is_text_success else '否'}")#註解SSS
                # print(f"📈 文字成功辨識：{'✔️ 成功' if is_text_success else '❌ 失敗'}")#註解SSS
                # print(f"📈 外型成功辨識：{'✔️ 成功' if is_shape_success else '❌ 失敗'}")#註解SSS
                if is_text_success:
                    text_success_total += 1
                if is_shape_success:
                    shape_success_total += 1

                # print('--------------------------------------------------------------------')#註解SSS

            except Exception as e:
                print(f"❗ 發生錯誤：{e}")

    # === 最後輸出結果 ===
    if missing_entries:
        print("\n❗❗❗ 以下資料夾不存在，請檢查： ❗❗❗\n")
        for entry in missing_entries:
            print(f"用量排序：{entry['用量排序']}，學名：{entry['學名']}")
        print(f"\n🔴 總共 {len(missing_entries)} 個藥物資料夾缺失。")
    else:
        print("\n✅ 所有資料夾皆成功讀取。")

    print("\n📊 總體統計：")

    print("🔠 文字辨識：")
    print(f" - 辨識結果：{text_success_total} 張正確")
    print(f" - 正式結果：{total_images} 張（總圖片數）")
    print(f" - 辨識成功率：{text_success_total / total_images:.2%}")

    print("\n🟫 外型辨識：")
    print(f" - 辨識結果：{shape_success_total} 張正確")
    print(f" - 正確結果：{total_images} 張（總圖片數）")
    print(f" - 辨識成功率：{shape_success_total / total_images:.2%}")

    print("\n🎨 顏色辨識：")
    print(f" - 辨識結果：未統計（如需統計請加上統計變數）")
    print(f" - 正確結果：{total_images} 張（總圖片數）")
    print(f" - 辨識成功率：尚未實作")

    print("\n💊 藥品名稱比對：")
    print(f" - 辨識結果：{total_success} 張比對成功")
    print(f" - 正確結果：{total_images} 張（總圖片數）")
    print(f" - 整體辨識成功率（以文字為主）：{total_success / total_images:.2%}")

    print("\n🔍 Roboflow 偵測統計：")
    print(f" - 成功偵測圖片數：{roboflow_success} / {roboflow_total}")
    print(f" - 偵測成功率：{roboflow_success / roboflow_total:.2%}")

    print("📦 各藥品辨識情況：")
    for drug, stats in per_drug_stats.items():
        print(f"- {drug}: {stats['success']} / {stats['total']} 成功")


from ultralytics import YOLO

# ✅ 全域載入本地 YOLO 模型（請提前初始化一次）
det_model = YOLO("/path/to/best.pt")  # 替換成你的 Roboflow 匯出 .pt 路徑


def process_image(img_path: str):
    """
<<<<<<< HEAD
    單張藥品圖片辨識流程（給 Flask 呼叫）
    - img_path: 圖片路徑（base64 decode 後的暫存圖）
    - return: dict，包含文字、顏色、形狀
    """
    from PIL import Image

    # 讀取圖片（自動判斷是否為 HEIC）
    from PIL import Image
    import base64

    # === 讀取圖片 ===
    input_img = read_image_safely(img_path)
    if input_img is None:
        return {"error": "無法讀取圖片"}

    # Roboflow 偵測
    rgb_pil = Image.fromarray(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
    result = CLIENT.infer(rgb_pil, model_id=MODEL_ID)
    preds = result.get("predictions", [])

    if preds:

        cropped_original, cropped_removed = extract_pill_region(input_img, preds[0])
    else:
        cropped_original, cropped_removed = fallback_rembg_bounding(input_img)

        if cropped_removed is None:
            return {"error": "藥品擷取失敗"}

    # === 使用本地 YOLO 模型進行推論 ===
    results = det_model(input_img)

    # === 處理 YOLO 偵測結果 ===
    preds = results[0].boxes
    if preds and len(preds) > 0:
        # 取最大框（信心分數最高的）
        boxes = preds.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        best_idx = preds.conf.argmax().item()
        box = boxes[best_idx]
        x1, y1, x2, y2 = map(int, box)
        cropped_original = input_img[y1:y2, x1:x2]
        cropped_removed = remove(cropped_original)  # 仍使用 rembg 去背
    else:
        # YOLO 偵測不到 ➜ fallback
        cropped_original, cropped_removed = fallback_rembg_bounding(input_img)
        if cropped_removed is None:
            return {"error": "藥品擷取失敗"}

    # === 裁切圖轉 Base64 給前端展示 ===
    _, buffer = cv2.imencode(".jpg", cropped_original)
    cropped_base64 = base64.b64encode(buffer).decode("utf-8")
    cropped_base64 = f"data:image/jpeg;base64,{cropped_base64}"

    result["cropped_image"] = cropped_base64
    # 形狀辨識
    shape, result = detect_shape_from_image(cropped_removed, cropped_original, expected_shape=None, debug=False)
    # 顏色辨識
    colors = extract_dominant_colors_by_ratio(cropped_removed, visualize=False)

    # 六個圖片增強版本
    image_versions = generate_image_versions(cropped_removed)
    # best_texts, best_name, best_score = get_best_ocr_texts(image_versions)
    best_texts, best_name, best_score = get_best_ocr_texts(image_versions, ocr_engine=ocr_engine)
    # === 外型、顏色分析 ===
    shape, _ = detect_shape_from_image(cropped_removed, cropped_original, expected_shape=None, debug=False)
    colors = extract_dominant_colors_by_ratio(cropped_removed, visualize=False)

    # === 多版本 OCR 辨識 ===
    image_versions = generate_image_versions(cropped_removed)
    best_texts, best_name, best_score = get_best_ocr_texts(image_versions, ocr_engine=ocr_engine)

    print("文字辨識：" + str(best_texts if best_texts else ["None"]))
    print("最佳版本：" + str(best_name))
    print("信心分數：" + str(round(best_score, 3)))
    print("顏色：" + str(colors))
    print("外型：" + str(shape))

    return {
        "文字辨識": best_texts if best_texts else ["None"],
        "最佳版本": best_name,
        "信心分數": round(best_score, 3),
        "顏色": colors,
        "外型": shape,
        "cropped_image": cropped_base64   # <=== 這是裁切圖
    }
