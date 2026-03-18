// src/App.jsx
import { useState, useEffect, useRef } from "react";
import "./App.css";
import { searchArtifacts, sendChat } from "./api.js";
import { logToGoogleSheet } from "./logger.js";
import aiAvatar from "./assets/ai-avatar.png";

function App() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      type: "text",
      content:
        "您好！我是您的博物館導覽員。無論您想了解展品的用途、材質或故事，都歡迎隨時問我。我都能為您解說，並搭配相應的情境圖哦。",
      sources: [],
    },
  ]);

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  // 等待使用者確認的那一題問題
  const [pendingQuestion, setPendingQuestion] = useState(null);

  // 目前候選文物 (顯示成 4~5 個按鈕)
  const [candidates, setCandidates] = useState([]);

  // 使用者手動輸入的文物名稱
  const [manualArtifact, setManualArtifact] = useState("");

  // 目前鎖定的文物 (用於連續提問)
  const [activeArtifact, setActiveArtifact] = useState(null);

  // 每一則訊息的「附註是否展開」
  const [openSourcesMap, setOpenSourcesMap] = useState({});

  // 用於自動滾動到底部
  const messagesEndRef = useRef(null);
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // 用於取消正在進行的請求
  const abortControllerRef = useRef(null);

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading, candidates]);

  const toggleSources = (msgIndex) => {
    setOpenSourcesMap((prev) => ({
      ...prev,
      [msgIndex]: !prev[msgIndex],
    }));
  };

  // 將文字中的換行顯示成 <br />
  const renderTextWithBreaks = (text) =>
    (text || "").split("\n").map((line, i) => (
      <span key={i}>
        {i > 0 && <br />}
        {line}
      </span>
    ));

  // 移除回答開頭的問候語
  const removeGreeting = (text) => {
    if (!text) return text;
    // 移除開頭的「您好！」「您好，」等問候語
    return text.replace(/^(您好[！!，,\s]*|你好[！!，,\s]*|Hi[！!，,\s]*|Hello[！!，,\s]*)/i, '').trim();
  };

  // 幫助函式：把 AI 回答加入 messages，並紀錄到 Google Sheet
  const addAssistantAnswer = (result, questionText, confidence) => {
    // 移除回答開頭的問候語
    const cleanedAnswer = removeGreeting(result.answer || "(沒有回傳文字回答)");

    setMessages((prev) => [
      ...prev,
      {
        role: "assistant",
        type: "text",
        content: cleanedAnswer,
        // 後端給的 sources 直接掛在這一則訊息底下
        sources: result.sources || [],
        artifactName: result.artifact_name, // 加入文物名稱
      },
      ...(result.image_url
        ? [
          {
            role: "assistant",
            type: "image",
            content: result.image_url,
            artifactName: result.artifact_name,
          },
        ]
        : []),
    ]);

    setPendingQuestion(null);
    setCandidates([]);
    setManualArtifact("");

    // 紀錄到 Google Sheet
    logToGoogleSheet({
      question: questionText,
      answer: result.answer,
      image_url: result.image_url,
      artifact_name: result.artifact_name,
      confidence_score: confidence,
    });
  };

  // 停止當前請求
  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setLoading(false);
    // 不添加訊息，保持候選文物顯示，讓使用者可以重新選擇
  };

  // 送出文字問題（第一階段：只做檢索）
  const handleSend = async () => {
    if (!input.trim() || loading) return;
    const question = input.trim();
    setInput("");

    // 新增使用者訊息
    setMessages((prev) => [
      ...prev,
      { role: "user", type: "text", content: question },
    ]);

    // 清空舊的候選狀態
    setPendingQuestion(null);
    setCandidates([]);
    setManualArtifact("");

    setLoading(true);
    abortControllerRef.current = new AbortController();

    try {
      // 如果已經有鎖定的文物，直接進入回答階段
      if (activeArtifact) {
        const result = await sendChat(question, activeArtifact, abortControllerRef.current.signal);
        addAssistantAnswer(result, question, "Context-Pinned");
        return;
      }

      const searchResult = await searchArtifacts(question, abortControllerRef.current.signal);
      const artifacts = searchResult.artifacts || [];

      if (artifacts.length === 0) {
        // 找不到特定文物 → 直接讓後端用「不指定文物」回答
        const answerResult = await sendChat(question, null, abortControllerRef.current.signal);
        addAssistantAnswer(answerResult, question, "N/A");
      } else if (artifacts.length === 1) {
        // 只有一個文物 → 自動用這件
        const only = artifacts[0];
        const answerResult = await sendChat(question, only.name, abortControllerRef.current.signal);
        addAssistantAnswer(answerResult, question, only.score || 1.0);
      } else {
        // 有多件文物 → 先請使用者選
        setPendingQuestion(question);
        setCandidates(artifacts);

        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            type: "text",
            content:
              "我找到幾件可能相關的文物，請點選您指的是哪一件，或在下方輸入展品名稱：",
          },
        ]);
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        console.log('Request was cancelled');
        return;
      }
      console.error(err);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          type: "text",
          content: `抱歉，後端服務發生錯誤：${err.message}`,
        },
      ]);
    } finally {
      abortControllerRef.current = null;
      setLoading(false);
    }
  };

  // 點選某一個文物候選（第二階段：真正回答 + 生圖）
  const handleChooseArtifact = async (artifact) => {
    if (!pendingQuestion || loading) return;

    // 使用者點選文物，也當作一則訊息
    setMessages((prev) => [
      ...prev,
      {
        role: "user",
        type: "text",
        content: artifact.name,
      },
    ]);

    setLoading(true);
    const savedCandidates = candidates; // 儲存候選文物，以便取消時恢復
    setCandidates([]);
    abortControllerRef.current = new AbortController();

    try {
      const result = await sendChat(pendingQuestion, artifact.name, abortControllerRef.current.signal);
      setActiveArtifact(artifact.name); // 鎖定目前文物
      addAssistantAnswer(result, pendingQuestion, artifact.score || "Selected");
    } catch (err) {
      if (err.name === 'AbortError') {
        console.log('Request was cancelled');
        // 恢復候選文物顯示
        setCandidates(savedCandidates);
        return;
      }
      console.error(err);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          type: "text",
          content: `抱歉，在生成回答時發生錯誤：${err.message}`,
        },
      ]);
    } finally {
      abortControllerRef.current = null;
      setLoading(false);
    }
  };

  // 使用者手動輸入文物名稱後，確認送出
  const handleManualArtifactConfirm = async () => {
    if (!pendingQuestion || loading) return;
    const name = manualArtifact.trim();
    if (!name) return;

    setMessages((prev) => [
      ...prev,
      {
        role: "user",
        type: "text",
        content: name,
      },
    ]);

    setLoading(true);
    setCandidates([]);
    abortControllerRef.current = new AbortController();

    try {
      const result = await sendChat(pendingQuestion, name, abortControllerRef.current.signal);
      setActiveArtifact(name); // 鎖定目前文物
      addAssistantAnswer(result, pendingQuestion, "Manual");
    } catch (err) {
      if (err.name === 'AbortError') {
        console.log('Request was cancelled');
        return;
      }
      console.error(err);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          type: "text",
          content: `抱歉，在生成回答時發生錯誤：${err.message}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const [lightboxImage, setLightboxImage] = useState(null);
  const [zoom, setZoom] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  const startLightbox = (src) => {
    setLightboxImage(src);
    setZoom(1);
    setPosition({ x: 0, y: 0 });
  };

  const closeLightbox = () => {
    setLightboxImage(null);
    setZoom(1);
    setPosition({ x: 0, y: 0 });
    setIsDragging(false);
  };

  const handleZoomIn = (e) => {
    e.stopPropagation();
    setZoom(prev => Math.min(prev + 0.2, 3));
  };

  const handleZoomOut = (e) => {
    e.stopPropagation();
    setZoom(prev => {
      const nextZoom = Math.max(prev - 0.2, 1);
      if (nextZoom === 1) setPosition({ x: 0, y: 0 });
      return nextZoom;
    });
  };

  const handleMouseDown = (e) => {
    if (zoom <= 1) return;
    setIsDragging(true);
    setDragStart({
      x: e.clientX - position.x,
      y: e.clientY - position.y
    });
  };

  const handleMouseMove = (e) => {
    if (!isDragging || zoom <= 1) return;
    setPosition({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleTouchStart = (e) => {
    if (zoom <= 1 || !e.touches[0]) return;
    setIsDragging(true);
    const touch = e.touches[0];
    setDragStart({
      x: touch.clientX - position.x,
      y: touch.clientY - position.y
    });
  };

  const handleTouchMove = (e) => {
    if (!isDragging || zoom <= 1 || !e.touches[0]) return;
    const touch = e.touches[0];
    setPosition({
      x: touch.clientX - dragStart.x,
      y: touch.clientY - dragStart.y
    });
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <>
      <div className="iphone-wrapper">
        <div className="iphone-frame">
          <div className="iphone-screen">
            <div className="iphone-dynamic-island"></div>
            <div className="app-root">
              <header className="hero">
                <h1>博物館 AI 導覽系統</h1>
              </header>

              <main className="chat-container">
                <div className="messages-area">
                  <div className="messages">
                    {messages.map((m, idx) => (
                      <div
                        key={idx}
                        className={`msg-row ${m.role === "user" ? "msg-user" : "msg-assistant"
                          }`}
                      >
                        {/* Avatar for Assistant */}
                        {m.role === "assistant" && (
                          <div className="avatar-container assistant-avatar">
                            <img src={aiAvatar} alt="AI" className="avatar-img" />
                          </div>
                        )}

                        {/* Avatar for User */}
                        {m.role === "user" && (
                          <div className="avatar-container user-avatar">
                            <div className="avatar-icon-user">
                              <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M12 12C14.21 12 16 10.21 16 8C16 5.79 14.21 4 12 4C9.79 4 8 5.79 8 8C8 10.21 9.79 12 12 12ZM12 14C9.33 14 4 15.34 4 18V20H20V18C20 15.34 14.67 14 12 14Z" fill="currentColor" />
                              </svg>
                            </div>
                          </div>
                        )}

                        <div className="msg-content-wrapper">
                          {/* 文字訊息 */}
                          {m.type === "text" && (
                            <>
                              <div className="msg-bubble">
                                {renderTextWithBreaks(m.content)}
                                {m.role === "assistant" &&
                                  m.sources &&
                                  m.sources.length > 0 && (
                                    <div className="sources-block">
                                      <button
                                        className="sources-toggle-btn"
                                        type="button"
                                        onClick={() => toggleSources(idx)}
                                      >
                                        {openSourcesMap[idx] ? "收合附註" : "展開附註"}
                                      </button>

                                      {openSourcesMap[idx] && (
                                        <div className="sources-list">
                                          {m.sources.map((s, sIdx) => (
                                            <div key={sIdx} className="source-item">
                                              <div className="source-title">
                                                [{s.index}] {s.title}
                                              </div>
                                              <div className="source-meta">
                                                來源檔案：{s.source}
                                              </div>
                                              <div className="source-snippet">
                                                {s.snippet}
                                              </div>
                                            </div>
                                          ))}
                                        </div>
                                      )}
                                    </div>
                                  )}
                              </div>
                              {/* 只在沒有後續圖片訊息時顯示切換按鈕 */}
                              {m.role === "assistant" && activeArtifact && m.artifactName === activeArtifact &&
                                messages[idx + 1]?.type !== "image" && (
                                  <div className="msg-context-controls">
                                    <button
                                      className="switch-artifact-btn"
                                      onClick={() => setActiveArtifact(null)}
                                    >
                                      詢問其他文物
                                    </button>
                                  </div>
                                )}
                            </>
                          )}

                          {/* 圖片訊息 */}
                          {m.type === "image" && (
                            <div className="msg-image-container-with-context">
                              <div className="msg-image-wrapper">
                                {m.artifactName && (
                                  <div className="msg-image-caption">{m.artifactName}</div>
                                )}
                                <img
                                  src={m.content}
                                  alt={m.artifactName || "生成情境圖"}
                                  className="msg-image"
                                  onClick={() => startLightbox(m.content)}
                                  title="點擊放大"
                                />
                              </div>
                              {activeArtifact === m.artifactName && (
                                <div className="msg-context-controls">
                                  <button
                                    className="switch-artifact-btn"
                                    onClick={() => setActiveArtifact(null)}
                                  >
                                    詢問其他文物
                                  </button>
                                </div>
                              )}
                            </div>
                          )}

                          {/* 如果是助手的最後一則訊息且有候選文物，且不在載入中，則直接在 Bubble 內顯示 */}
                          {!loading && m.role === "assistant" && idx === messages.filter(msg => msg.role === "assistant").length + messages.filter(msg => msg.role === "user").length - 1 && candidates.length > 0 && (
                            <div className="artifact-bubble-integration">
                              {candidates.length > 3 ? (
                                <div className="artifact-dropdown-container">
                                  <select
                                    className="artifact-select"
                                    onChange={(e) => {
                                      const selected = candidates.find(c => c.name === e.target.value);
                                      if (selected) handleChooseArtifact(selected);
                                    }}
                                    defaultValue=""
                                    disabled={loading}
                                  >
                                    <option value="" disabled>點擊選擇展品...</option>
                                    {candidates.map((a) => (
                                      <option key={a.name} value={a.name}>
                                        {a.name}
                                      </option>
                                    ))}
                                  </select>
                                </div>
                              ) : (
                                <div className="artifact-list">
                                  {candidates.map((a) => (
                                    <button
                                      key={a.name}
                                      className="artifact-chip"
                                      onClick={() => handleChooseArtifact(a)}
                                      disabled={loading}
                                    >
                                      {a.name}
                                    </button>
                                  ))}
                                </div>
                              )}

                              <div className="manual-artifact">
                                <span className="manual-label">或手動輸入名稱：</span>
                                <div className="manual-input-row">
                                  <input
                                    className="manual-input"
                                    value={manualArtifact}
                                    onChange={(e) => setManualArtifact(e.target.value)}
                                    onKeyDown={(e) => {
                                      if (e.key === "Enter") {
                                        e.preventDefault();
                                        handleManualArtifactConfirm();
                                      }
                                    }}
                                    placeholder="例如：茶壺名稱"
                                    disabled={loading}
                                  />
                                  <button
                                    className={`manual-btn ${manualArtifact.trim() ? "active" : ""}`}
                                    onClick={handleManualArtifactConfirm}
                                    disabled={!manualArtifact.trim() || loading}
                                  >
                                    確認
                                  </button>
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    ))}

                    {loading && (
                      <div className="msg-row msg-assistant">
                        <div className="avatar-container assistant-avatar">
                          <img src={aiAvatar} alt="AI" className="avatar-img" />
                        </div>
                        <div className="msg-content-wrapper">
                          <div className="msg-bubble">正在思考與檢索中……</div>
                        </div>
                      </div>
                    )}
                    <div ref={messagesEndRef} />
                  </div>
                </div>

                <div className="chat-footer-wrapper">
                  <div className="input-bar">
                    <textarea
                      placeholder={activeArtifact ? `繼續詢問「${activeArtifact}」...` : "請輸入問題"}
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyDown={handleKeyDown}
                      disabled={loading}
                    />
                    {loading ? (
                      <button onClick={handleStop} className="stop-btn">
                        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <rect x="6" y="6" width="12" height="12" fill="currentColor" />
                        </svg>
                      </button>
                    ) : (
                      <button onClick={handleSend} disabled={!input.trim()} className="send-btn">
                        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" fill="currentColor" />
                        </svg>
                      </button>
                    )}
                  </div>
                </div>
              </main>

            </div>
          </div>
          <div className="iphone-home-bar"></div>
        </div>
      </div>

      {lightboxImage && (
        <div className="lightbox-overlay" onClick={closeLightbox}>
          <div className="lightbox-content" onClick={(e) => e.stopPropagation()}>
            <div
              className={`lightbox-image-container ${isDragging ? "dragging" : ""}`}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
              onTouchStart={handleTouchStart}
              onTouchMove={handleTouchMove}
              onTouchEnd={handleMouseUp}
              style={{ cursor: zoom > 1 ? (isDragging ? "grabbing" : "grab") : "default" }}
            >
              <img
                src={lightboxImage}
                alt="Enlarged"
                className="lightbox-img"
                style={{
                  transform: `scale(${zoom}) translate(${position.x / zoom}px, ${position.y / zoom}px)`,
                  transition: isDragging ? "none" : "transform 0.2s cubic-bezier(0.4, 0, 0.2, 1)"
                }}
              />
            </div>
            <button className="lightbox-close" onClick={closeLightbox}>×</button>
            <div className="lightbox-controls">
              <button className="zoom-btn" onClick={handleZoomOut} disabled={zoom <= 1}>縮小</button>
              <span className="zoom-level">{Math.round(zoom * 100)}%</span>
              <button className="zoom-btn" onClick={handleZoomIn} disabled={zoom >= 3}>放大</button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export default App;