import cv2
from matplotlib.font_manager import FontProperties
from sklearn.cluster import KMeans
import numpy as np

zh_font = FontProperties(fname="C:/Windows/Fonts/msjh.ttc")


def rotate_image_by_angle(image, angle):
    """
    將圖片依指定角度旋轉。
    - image: 輸入圖片（OpenCV 格式）
    - angle: 順時針旋轉角度（例如 90, 180）
    - return: 旋轉後的圖片
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return rotated


def enhance_contrast(img, clip_limit, alpha, beta):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhance_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(enhance_img, (5, 5), 3.0)
    return cv2.addWeighted(enhance_img, alpha, blurred, beta, 0)


def extract_dominant_colors_by_ratio(cropped_img, k=6, min_ratio=0.4, visualize=True):
    import colorsys
    from collections import Counter
    import matplotlib.pyplot as plt

    def rgb2hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

    def rgb_to_color_group(rgb):
        r, g, b = rgb / 255.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        h_deg = h * 360
        if v < 0.2:
            return "黑色"
        if s < 0.1 and v > 0.9:
            return "白色"
        if s < 0.05 and v > 0.6:
            return "透明"
        if h_deg < 15 or h_deg >= 345:
            return "紅色"
        elif h_deg < 40:
            return "橘色"
        elif h_deg < 55:
            return "皮膚色"
        elif h_deg < 65:
            return "黃色"
        elif h_deg < 170:
            return "綠色"
        elif h_deg < 250:
            return "藍色"
        elif h_deg < 290:
            return "紫色"
        elif h_deg < 345:
            return "粉紅色"
        if s > 0.2 and v < 0.5:
            return "棕色"
        return "未知"

    # 相近色映射表（不互為雙向）
    similar_color_map = {
        "皮膚色": "黃色",
        "橘色": "紅色",
        "粉紅色": "紅色",
        "透明": "白色",
        "棕色": "黑色"
    }

    img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(img_rgb, (64, 64), interpolation=cv2.INTER_AREA)
    img_list = resized_img.reshape((-1, 3))
    img_list = img_list[np.sum(img_list, axis=1) > 30]

    clt = KMeans(n_clusters=k, random_state=42)
    labels = clt.fit_predict(img_list)
    label_counts = Counter(labels)
    total = sum(label_counts.values())
    center_colors = clt.cluster_centers_

    # 語意色統計
    semantic_counter = Counter()
    for label, count in label_counts.items():
        color = rgb_to_color_group(center_colors[label])
        if color not in ["未知", "透明"]:
            semantic_counter[color] += count

    # 篩選主色（佔比 ≥ min_ratio，最多2種）
    filtered = [
        (color, count / total) for color, count in semantic_counter.items()
        if count / total >= min_ratio
    ]
    filtered = sorted(filtered, key=lambda x: -x[1])[:2]
    dominant_colors = [color for color, _ in filtered]

    # 擴充相近色（不能與主色重複）
    extended_colors = dominant_colors.copy()
    for color in dominant_colors:
        if color in similar_color_map:
            similar = similar_color_map[color]
            if similar not in extended_colors:
                extended_colors.append(similar)

    # 圖示
    if visualize:
        ordered_labels = list(label_counts.keys())
        ordered_colors = [center_colors[i] / 255 for i in ordered_labels]
        hex_colors = [rgb2hex(center_colors[i]) for i in ordered_labels]

        plt.figure(figsize=(14, 4))
        plt.subplot(131)
        plt.imshow(img_rgb)
        plt.axis('off')

        plt.subplot(132)
        plt.pie(
            [label_counts[i] for i in ordered_labels],
            labels=hex_colors,
            colors=ordered_colors,
            startangle=90
        )
        plt.axis('equal')
        plt.title("Raw RGB Cluster Pie")

        plt.subplot(133)
        plt.bar(semantic_counter.keys(), semantic_counter.values(), color='gray')
        # 用時加上 fontproperties
        plt.xticks(rotation=45, fontproperties=zh_font)
        plt.title("Color Group Count")
        plt.tight_layout()
        plt.show()

    return extended_colors


# ===外型辨識函式 ===
def detect_shape_from_image(cropped_img, original_img=None, expected_shape=None, debug=False):
    try:
        output = cropped_img.copy()
        thresh = preprocess_with_shadow_correction(output)
        if debug:
            # cv2_imshow(output)#註解SSS
            # cv2_imshow(thresh)
            cv2.imshow("thresh", thresh)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        shape = "其他"

        if not contours and original_img is not None:
            # print("⚠️ 無偵測到輪廓，改用原圖嘗試")#註解SSS
            gray_fallback = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            _, thresh_fallback = cv2.threshold(gray_fallback, 127, 255, cv2.THRESH_BINARY)
            contours_fallback, _ = cv2.findContours(thresh_fallback, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours_fallback:
                main_contour = max(contours_fallback, key=cv2.contourArea)
                shape = detect_shape_three_classes(main_contour, img_debug=output if debug else None)
            else:
                print("⚠️ 二次嘗試仍無輪廓，標記為其他")  # 註解SSS
        elif contours:
            main_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(main_contour)
            img_area = cropped_img.shape[0] * cropped_img.shape[1]
            area_ratio = area / img_area
            # print(f"📐 輪廓面積：{area:.1f}，圖片面積：{img_area:.1f}，佔比：{area_ratio:.2%}")#註解SSS
            shape = detect_shape_three_classes(main_contour, img_debug=output if debug else None)

        if shape != "其他" and debug:
            cv2.drawContours(output, [main_contour], -1, (0, 0, 255), 3)
            x, y, w, h = cv2.boundingRect(main_contour)
            cv2.putText(output, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # cv2_imshow(output)#註解SSS

        if expected_shape:
            result = "✅" if shape == expected_shape else "❌"
            # print(f"📏 預測結果：{shape}，正確結果：{expected_shape} {result}")#註解SSS
            return shape, result
        return shape, None

    except Exception as e:
        # print(f"❗ 發生錯誤：{e}")#註解SSS
        return "錯誤", None


# === 增強處理函式 ===

def desaturate_image(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s.fill(0)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)


def enhance_for_blur(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    contrast_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(contrast_img, (3, 3), 1.0)
    sharpened = cv2.addWeighted(contrast_img, 1.8, blurred, -0.8, 0)
    return cv2.bilateralFilter(sharpened, d=9, sigmaColor=75, sigmaSpace=75)


# === 3. 定義讀取 HEIC 的函式 ===


def rgb2hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


# === 形狀辨識相關 ===
def preprocess_with_shadow_correction(img_bgr):
    """校正陰影與自動二值化，改善輪廓品質"""
    # Step 1: 灰階轉換
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Step 2: 高斯模糊計算背景亮度
    blur = cv2.GaussianBlur(gray, (55, 55), 0)

    # Step 3: 灰階除以背景亮度 => 修正陰影
    corrected = cv2.divide(gray, blur, scale=255)

    # Step 4: 自適應 threshold => 適合局部亮度變化
    thresh = cv2.adaptiveThreshold(
        corrected, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 10
    )

    return thresh


# === 放在檔案頂部定義統計用清單 ===
ratios_list = []


def detect_shape_three_classes(contour, img_debug=None):
    shape = "其他"
    # print(len(contour))#註解SSS
    try:
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (center, axes, angle) = ellipse
            major, minor = axes

            if minor == 0:
                return shape

            ratio = max(major, minor) / min(major, minor)
            ratios_list.append(ratio)
            # print(f"🔍 Ellipse ratio: {ratio:.3f}")#註解SSS

            # ➤ 畫橢圓
            if img_debug is not None:
                cv2.ellipse(img_debug, ellipse, (0, 255, 0), 2)

                # ➤ 畫主軸線
                cx, cy = int(center[0]), int(center[1])
                length = int(max(axes) / 2)
                angle_rad = np.deg2rad(angle)
                dx = int(length * np.cos(angle_rad))
                dy = int(length * np.sin(angle_rad))
                pt1 = (cx - dx, cy - dy)
                pt2 = (cx + dx, cy + dy)
                cv2.line(img_debug, pt1, pt2, (255, 0, 255), 2)

                # ➤ 印出比值
                cv2.putText(img_debug, f"{ratio:.2f}", (cx, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # ➤ 分類
            if 0.88 <= ratio <= 1.25:
                shape = "圓形"
            elif ratio <= 1.9:
                shape = "橢圓形"
            else:
                shape = "其他"

            # print(f"📏 shape ratio: {ratio:.2f} => 判斷為 {shape}")

    except:
        pass

    return shape
