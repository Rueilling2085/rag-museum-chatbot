function doPost(e) {
    try {
        // 使用您指定的試算表 ID
        var ss = SpreadsheetApp.openById("1h_pvUNWb_JlSb0mFb-u-yTaANpHOC1NqjPn7LfRSxi8");
        var sheet = ss.getSheets()[0]; // 取得第一個工作表
        var data = JSON.parse(e.postData.contents);

        // 取得當前時間
        var timestamp = new Date();

        // 將資料依序寫入新的一列
        // 欄位順序：時間, 問項, AI回答, 圖片網址, 文物名稱, 信心分數
        sheet.appendRow([
            timestamp,
            data.question || "",
            data.answer || "",
            data.image_url || "",
            data.artifact_name || "",
            data.confidence_score || ""
        ]);

        return ContentService.createTextOutput("Success").setMimeType(ContentService.MimeType.TEXT);
    } catch (err) {
        return ContentService.createTextOutput("Error: " + err.toString()).setMimeType(ContentService.MimeType.TEXT);
    }
}

// 測試用：確保腳本有權限執行
function testLog() {
    var ss = SpreadsheetApp.openById("1h_pvUNWb_JlSb0mFb-u-yTaANpHOC1NqjPn7LfRSxi8");
    var sheet = ss.getSheets()[0];
    sheet.appendRow([new Date(), "測試連線", "成功", "", "", ""]);
}
