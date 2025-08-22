// 獲取 DOM 元素
const uploadInput = document.getElementById('uploadInput');
const customFileButton = document.getElementById('customFileButton');
const fileInput = document.getElementById('fileInput');

const textField = document.getElementById('recognizedText');
const color1Select = document.getElementById('color1');
const color2Select = document.getElementById('color2');
const shapeSelect = document.getElementById('shape');
const confirmButton = document.getElementById('confirmButton');

customFileButton.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = async function (e) {
            Detection(e.target.result);
        };
        reader.readAsDataURL(file);
    }
});

uploadInput.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            Detection(e.target.result);
        };
        reader.readAsDataURL(file);
    }
});


async function Detection(imageData) {
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({image: imageData})
        });

        const parsedData = await response.json();
        const result = parsedData.result;

        const container = document.getElementById('photo-container');
        container.innerHTML = ''; // 清空
        if (result?.cropped_image) {


            const croppedImg = new Image();
            croppedImg.src = result.cropped_image;
            croppedImg.alt = "裁切後圖片";

            // === 改小圖片尺寸 ===
            croppedImg.style.maxWidth = '60%';     // 寬度限制 60% 容器
            croppedImg.style.maxHeight = '250px';  // 限制最大高度
            croppedImg.style.objectFit = 'contain'; // 保持比例
            croppedImg.style.border = '2px solid #888';
            croppedImg.style.borderRadius = '10px';
            croppedImg.style.margin = '0 auto 10px auto';
            croppedImg.style.display = 'block';

            container.appendChild(croppedImg);
        }


        // ✅ 填入辨識結果
        if (result) {
            textField.value = result["文字辨識"]?.join(', ') || '';
            const colors = result["顏色"] || [];
            color1Select.value = colors[0] || '';
            color2Select.value = colors[1] || '';
            shapeSelect.value = result["外型"] || '';
        } else {
            alert('辨識失敗，未找到藥品資訊。');
        }

    } catch (error) {
        alert('🚨 圖片辨識錯誤：' + error.message);
    }
}

function showDrugDetail(drug) {
    // 更新左側圖片
    document.getElementById('modalDrugImage').src = drug.drug_image || '';

    // 更新右側文字資訊
    document.getElementById('modalDrugName').innerText = drug.name;
    document.getElementById('modalSymptoms').innerText = drug.symptoms;
    document.getElementById('modalPrecautions').innerText = drug.precautions;
    document.getElementById('modalSideEffects').innerText = drug.side_effects;

    // 隱藏候選列表 (可選，或保留讓使用者再切換)
    // document.getElementById('modalCandidates').style.display = 'none';
}

confirmButton.addEventListener('click', async () => {
    const selectedText = textField.value.trim();
    const selectedColor1 = color1Select.value.trim();
    const selectedColor2 = color2Select.value.trim();
    const selectedShape = shapeSelect.value.trim();

    const payload = {
        texts: selectedText.split(',').map(t => t.trim()).filter(Boolean),
        colors: [selectedColor1, selectedColor2].filter(Boolean),
        shape: selectedShape
    };

    try {
        const response = await fetch('/match', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });

        // 嘗試解析 JSON
        const text = await response.text();
        let result;
        try {
            result = JSON.parse(text);
        } catch (err) {
            alert("⚠️ JSON 解析錯誤：" + err.message + "\n原始回應：\n" + text);
            return;
        }

        // 錯誤處理
        if (result.error) {
            alert("❌ 錯誤訊息：" + result.error);
            return;
        }

        // =============== 多候選模式 ===============
        if (result.candidates) {
            // === 多候選模式 ===
            const candidateList = document.getElementById('candidateList');
            candidateList.innerHTML = ''; // 清空

            result.candidates.forEach((drug) => {
                const wrapper = document.createElement('div');
                wrapper.style.display = 'flex';
                wrapper.style.flexDirection = 'column';
                wrapper.style.alignItems = 'center';
                wrapper.style.border = '1px solid #ccc';
                wrapper.style.borderRadius = '8px';
                wrapper.style.padding = '10px';
                wrapper.style.background = '#fafafa';

                // 圖片
                const img = new Image();
                img.src = drug.drug_image || '';
                img.alt = drug.name;
                img.style.maxWidth = '100%';
                img.style.maxHeight = '50vh';
                img.style.objectFit = 'contain';
                img.style.border = '1px solid #ccc';
                img.style.borderRadius = '5px';
                wrapper.appendChild(img);

                // 文字
                const textArea = document.createElement('div');
                textArea.style.width = '100%';
                textArea.style.marginTop = '10px';
                textArea.style.textAlign = 'left';
                textArea.innerHTML = `
            <h3>${drug.name}</h3>
            <p><strong>適應症：</strong>${drug.symptoms || '無'}</p>
            <p><strong>注意事項：</strong>${drug.precautions || '無'}</p>
            <p><strong>副作用：</strong>${drug.side_effects || '無'}</p>
        `;
                wrapper.appendChild(textArea);

                candidateList.appendChild(wrapper);
            });

            document.getElementById('resultModal').style.display = 'flex';

        } else if (result.name) {
            // === 單一結果模式 ===
            const candidateList = document.getElementById('candidateList');
            candidateList.innerHTML = ''; // 清空

            const wrapper = document.createElement('div');
            wrapper.style.display = 'flex';
            wrapper.style.flexDirection = 'column';
            wrapper.style.alignItems = 'center';

            wrapper.innerHTML = `
    <img src="${result.drug_image || ''}" alt="藥物圖片"
         style="max-width:100%; max-height:50vh; border:1px solid #ccc; border-radius:5px; object-fit:contain; margin-bottom:10px;">
    <div style="width: 100%; text-align: left;">   <!-- 包一層 div -->
        <h3>${result.name}</h3>
        <p><strong>適應症：</strong>${result.symptoms || '無'}</p>
        <p><strong>注意事項：</strong>${result.precautions || '無'}</p>
        <p><strong>副作用：</strong>${result.side_effects || '無'}</p>
    </div>
`;

            candidateList.appendChild(wrapper);

            document.getElementById('resultModal').style.display = 'flex';

        } else {
            alert("❌ 沒有收到符合的結果");
        }


        // 關閉 Modal
        document.getElementById('closeModal').addEventListener('click', () => {
            document.getElementById('resultModal').style.display = 'none';
        });

        // 點背景關閉
        document.getElementById('resultModal').addEventListener('click', (e) => {
            if (e.target.id === 'resultModal') {
                document.getElementById('resultModal').style.display = 'none';
            }
        });

    } catch (error) {
        alert("🚨 請求失敗：" + error.message);
    }
});

