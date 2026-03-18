// src/logger.js

// 請將此處替換為您從 Google Apps Script 取得的 Web App URL
const GOOGLE_SHEET_WEBAPP_URL = "https://script.google.com/macros/s/AKfycbxDIBRHWbdq3pTBXeee0U1eTXM3aaYxAuUvFSmtvt3w7wvNcVUJqpG68DWL35TxNF_7/exec";

/**
 * 將互動紀錄送到 Google Sheet
 * @param {Object} data - 包含 question, answer, image_url, artifact_name, confidence_score 的物件
 */
export async function logToGoogleSheet(data) {
    if (!GOOGLE_SHEET_WEBAPP_URL) {
        console.warn("Google Sheet Web App URL 未設定，略過記錄。");
        return;
    }

    try {
        const payload = {
            question: data.question || "",
            answer: data.answer || "",
            image_url: data.image_url || "",
            artifact_name: data.artifact_name || "",
            confidence_score: data.confidence_score || "",
        };

        const res = await fetch(GOOGLE_SHEET_WEBAPP_URL, {
            method: "POST",
            mode: "no-cors",
            cache: "no-cache",
            headers: {
                "Content-Type": "text/plain",
            },
            body: JSON.stringify(payload),
        });

        console.log("互動紀錄指令已送出至 Google Sheet:", payload);
    } catch (err) {
        console.error("傳送紀錄到 Google Sheet 時發生基礎錯誤:", err);
    }
}
