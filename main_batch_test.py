from pathlib import Path
from app.utils.pill_detection import test_batch_all_images

if __name__ == "__main__":
    ROOT_FOLDER = Path("C:/Users/92102/OneDrive - NTHU/桌面/大三下/畢業專題/APP_新版本/data/drug_photo_copy")
    EXCEL_PATH = Path("data/TESTData.xlsx")
    test_batch_all_images(ROOT_FOLDER, EXCEL_PATH, start_index=1, end_index=403)
