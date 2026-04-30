// src/assets/api.js

// 後端 FastAPI 的 base URL
// 部署在 Zeabur 時，前後端同一網域，使用相對路徑即可
// 本地開發時，請將此改為 "http://localhost:8000"
const API_BASE = import.meta.env.VITE_API_BASE || "";

// 小工具：把後端回傳的 image_url 補上完整網址
function withFullImageUrl(data) {
    if (data && data.image_url) {
        if (!data.image_url.startsWith("http")) {
            const baseUrl = API_BASE.endsWith("/") ? API_BASE.slice(0, -1) : API_BASE;
            const path = data.image_url.startsWith("/") ? data.image_url : `/${data.image_url}`;
            data.image_url = `${baseUrl}${path}`;
        }
    }
    return data;
}

// 第一步：丟問題給後端，拿回候選文物清單
export async function searchArtifacts(question, signal = null) {
    const res = await fetch(`${API_BASE}/artifacts/search`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ question }),
        signal: signal,
    });

    if (!res.ok) {
        const text = await res.text();
        throw new Error(`後端錯誤：${res.status} ${text}`);
    }

    // { question, artifacts: [...] }
    return res.json();
}

// 第二步：使用者選好文物後，再真正要答案 + 圖片
export async function sendChat(question, artifactName, signal = null) {
    const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            question,
            artifact_name: artifactName || null,
        }),
        signal: signal,
    });

    if (!res.ok) {
        const text = await res.text();
        throw new Error(`後端錯誤：${res.status} ${text}`);
    }

    const data = await res.json(); // { answer, artifact_name, image_url, artifacts }
    return withFullImageUrl(data);
}