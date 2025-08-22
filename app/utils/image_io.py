
def read_image_safely(img_path):
    from pathlib import Path
    from PIL import Image
    import cv2
    import numpy as np
    import pillow_heif
    # 必須註冊 HEIC 支援
    pillow_heif.register_heif_opener()
    try:
        img_path = Path(img_path)
        if not img_path.exists():
            print(f"❗ 圖片路徑不存在：{img_path}")
            return None

        suffix = img_path.suffix.lower()
        if suffix in {".heic", ".heif"}:
            print(f"📄 使用 PIL 讀取 HEIC 圖片：{img_path}")
            pil_img = Image.open(img_path).convert("RGB")
            np_img = np.array(pil_img)
            if np_img is None:
                print("⚠️ PIL 無法轉成 numpy")
            return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        else:
            print(f"📄 使用 OpenCV 讀取圖片：{img_path}")
            img = cv2.imread(str(img_path))
            if img is None:
                print("⚠️ OpenCV 無法讀取此圖片")
            return img
    except Exception as e:
        print(f"❌ 圖片讀取錯誤：{img_path} ➜ {e}")
        return None
