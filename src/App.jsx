// src/App.jsx
import { useState, useEffect, useRef } from "react";
import "./App.css";
import { searchArtifacts, sendChat, sendChatStream } from "./api.js";
import { logToGoogleSheet } from "./logger.js";

// Import some beautiful featured artifacts
import img1 from "./assets/artifacts/仿官釉青瓷筆筒_2.jpg";
import img2 from "./assets/artifacts/剔紅種菊圖捧盒_1.png";
import img3 from "./assets/artifacts/汝窯 青瓷洗 「奉華」銘_1.jpg";
import img4 from "./assets/artifacts/玉佛手_1.jpg";
import img5 from "./assets/artifacts/金鑲東珠貓睛石嬪妃朝冠頂_1.jpg";
import img6 from "./assets/artifacts/畫琺瑯西洋人物懷錶_1.jpg";
import img7 from "./assets/artifacts/銀鍍金鏤空芙蓉花指甲套_3.jpg";
import img8 from "./assets/artifacts/瑪瑙花式碗_1.jpg";
import img9 from "./assets/artifacts/青瓷觚_1.jpg"; // 新增青瓷觚
import exhibitionPoster from "./assets/exhibition-poster.jpg";
import aiAvatar from "./assets/ai-avatar.png";

// 自動匯入所有的文物圖片
const ALL_ARTIFACTS_RAW = import.meta.glob('./assets/artifacts/*.{jpg,png}', { eager: true });
const ALL_ARTIFACTS = Object.entries(ALL_ARTIFACTS_RAW).map(([path, module], idx) => {
  const filename = path.split('/').pop();
  let name = filename.replace(/\.(jpeg|jpg|png)$/i, '');
  // 去除結尾類似 _1, _2 等編號 (例如 瑪瑙花式碗_1 -> 瑪瑙花式碗)
  name = name.replace(/_\d+$/, '');
  return {
    id: idx + 100,
    name: name,
    image: module.default
  };
});

const FEATURED_ARTIFACTS = [
  // 您要求置頂的 3 件經典文物
  { id: 5, name: "金鑲東珠貓睛石嬪妃朝冠頂", image: img5 },
  { id: 8, name: "瑪瑙花式碗", image: img8 },
  { id: 9, name: "青瓷觚", image: img9 },
  
  // 其餘的文物
  { id: 3, name: "汝窯 青瓷洗 「奉華」銘", image: img3 },
  { id: 1, name: "仿官釉青瓷筆筒", image: img1 },
  { id: 4, name: "玉佛手", image: img4 },
  { id: 2, name: "剔紅種菊圖捧盒", image: img2 },
  { id: 6, name: "畫琺瑯西洋人物懷錶", image: img6 },
  { id: 7, name: "銀鍍金鏤空芙蓉花指甲套", image: img7 },
];

const GUIDED_PROMPTS = [
  { icon: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#555555" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>, label: "功能用途與情境", text: "這件文物的用途是什麼？" },
  { icon: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#555555" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>, label: "材質與工藝", text: "這件文物是用什麼材質製成的？" },
  { icon: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#555555" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"></path></svg>, label: "故事與象徵", text: "這件文物背後有什麼故事或象徵意涵？" },
  { icon: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#555555" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>, label: "歷史使用者", text: "這件文物通常是誰在使用的？" },
];

// 將文字中的換行顯示成 <br /> 並且支援 **粗體** 顯示
const renderTextWithBreaks = (text) =>
  (text || "").split("\n").map((line, i) => {
    // 依據 **粗體** 或 💡 切割字串
    const parts = line.split(/(\*\*.*?\*\*|💡)/g);
    return (
      <span key={i}>
        {i > 0 && <br />}
        {parts.map((part, j) => {
          if (part.startsWith('**') && part.endsWith('**')) {
            return <strong key={j}>{part.slice(2, -2)}</strong>;
          }
          if (part === '💡') {
            return (
              <span key={j} className="inline-bulb-icon">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M9 18h6"></path>
                  <path d="M10 22h4"></path>
                  <path d="M15.09 14c.18-.98.65-1.74 1.41-2.5A4.65 4.65 0 0 0 18 8 6 6 0 0 0 6 8c0 1 .23 2.23 1.5 3.5A4.61 4.61 0 0 1 8.91 14"></path>
                </svg>
              </span>
            );
          }
          return part;
        })}
      </span>
    );
  });

// 來源附註項目的自訂元件 (有展開/收起功能)
const SourceItemCard = ({ source }) => {
  const [expanded, setExpanded] = useState(false);
  const MAX_LENGTH = 120; // 預設顯示文字量
  const snippetText = source.snippet || "";
  const isLong = snippetText.length > MAX_LENGTH;
  const displaySnippet = (!isLong || expanded) ? snippetText : snippetText.slice(0, MAX_LENGTH) + "...";

  return (
    <div className="source-item">
      <div className="source-title">
        [{source.index}] {source.title}
      </div>
      <div className="source-snippet">
        {renderTextWithBreaks(displaySnippet)}
      </div>
      {isLong && (
        <button 
          className="read-more-btn" 
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? "▴ 收起完整原文" : "▾ 展開完整原文"}
        </button>
      )}
    </div>
  );
};

function App() {
  // 視圖模式: 'landing' (展覽首頁) 或 'chat' (AI 導覽模式)
  const [viewMode, setViewMode] = useState('landing');
  // 展覽首頁的分頁: 'intro' (簡介) 或 'catalog' (相關文物)
  const [landingTab, setLandingTab] = useState('intro');

  const [messages, setMessages] = useState([
    {
      role: "assistant",
      type: "text",
      content:
        "您好！我是您的博物館導覽員👋\n\n很高興能為您服務！請問您想了解關於這件文物的什麼資訊呢？",
      sources: [],
    },
  ]);

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState("searching"); // searching, answering, imaging
  const [isPromptMenuOpen, setIsPromptMenuOpen] = useState(false);
  
  // 全部館藏的檢索視窗
  const [isCatalogOpen, setIsCatalogOpen] = useState(false);
  // 篩選全部館藏的字串
  const [catalogSearch, setCatalogSearch] = useState("");

  // 等待使用者確認的那一題問題
  const [pendingQuestion, setPendingQuestion] = useState(null);

  // 是否正在等待使用者選擇文物後，自動開啟 💡 選單
  const [waitingForPromptMenu, setWaitingForPromptMenu] = useState(false);

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

  // 移除回答開頭的問候語
  const removeGreeting = (text) => {
    if (!text) return text;
    // 移除開頭的「您好！」「您好，」等問候語
    return text.replace(/^(您好[！!，,\s]*|你好[！!，,\s]*|Hi[！!，,\s]*|Hello[！!，,\s]*)/i, '').trim();
  };

  /**
   * 核心串流處理邏輯 (方案 B)
   */
  const executeChatStream = async (questionText, artifactName) => {
    setLoading(true);
    setLoadingStatus("searching");
    abortControllerRef.current = new AbortController();

    // 先建立一個空的助手訊息，準備接收串流內容
    // 注意：這裡先不加到 messages，等收到第一個事件再說，避免空對話框
    let hasAddedMessage = false;
    let currentFullAnswer = "";

    try {
      await sendChatStream(
        questionText,
        artifactName,
        (event) => {
          if (event.type === "start") {
            // 更新候選清單與鎖定文物 (如果是自動匹配到的話)
            if (event.artifact_name) setActiveArtifact(event.artifact_name);
            if (event.artifacts) setCandidates(event.artifacts);
          } 
          else if (event.type === "sources") {
            // 收到來源後，確保訊息已建立並掛上去
            ensureAssistantMessage();
            setMessages(prev => {
              const next = [...prev];
              // 找到最新的一則文字訊息 (通常是倒數第一或第二)
              for (let i = next.length - 1; i >= 0; i--) {
                if (next[i].role === "assistant" && next[i].type === "text") {
                  next[i].sources = event.sources;
                  break;
                }
              }
              return next;
            });
          }
          else if (event.type === "text") {
            setLoadingStatus("answering");
            ensureAssistantMessage();
            currentFullAnswer += event.content;
            setMessages(prev => {
              const next = [...prev];
              for (let i = next.length - 1; i >= 0; i--) {
                if (next[i].role === "assistant" && next[i].type === "text") {
                  next[i].content = removeGreeting(currentFullAnswer);
                  break;
                }
              }
              return next;
            });
          }
          else if (event.type === "processing_image") {
            setLoadingStatus("imaging");
          }
          else if (event.type === "image") {
            setLoadingStatus("done");
            // 圖片獨立成一則訊息
            setMessages(prev => [
              ...prev,
              {
                role: "assistant",
                type: "image",
                content: event.image_url,
                artifactName: artifactName || activeArtifact,
              }
            ]);
          }
          else if (event.type === "error") {
             setMessages(prev => [...prev, { role: "assistant", type: "text", content: `錯誤: ${event.content || event.answer}` }]);
          }
        },
        abortControllerRef.current.signal
      );

      // 結束後紀錄 (這時已經有完整 answer 和 image_url 了，但為了簡單，我們可以在 done 事件後補錄，
      // 或者這裡直接略過 Google Sheet 紀錄，或稍後再補)
    } catch (err) {
      if (err.name === 'AbortError') return;
      console.error(err);
      setMessages(prev => [...prev, { role: "assistant", type: "text", content: `服務暫時無法連線: ${err.message}` }]);
    } finally {
      setLoading(false);
      abortControllerRef.current = null;
    }

    function ensureAssistantMessage() {
      if (!hasAddedMessage) {
        setMessages(prev => [
          ...prev,
          {
            role: "assistant",
            type: "text",
            content: "",
            sources: [],
            artifactName: artifactName || activeArtifact
          }
        ]);
        hasAddedMessage = true;
      }
    }
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
      // 1. 如果已經有鎖定的文物，檢查語意中是否有明確提到 *其他* 展品的名稱 (精準切換)
      if (activeArtifact) {
        const mentionedOther = ALL_ARTIFACTS.find(a => 
          a.name !== activeArtifact && question.includes(a.name)
        );

        if (mentionedOther) {
          // 使用者文字中明確提到了另一個文物，直接自動切換上下文
          setActiveArtifact(mentionedOther.name);
          await executeChatStream(question, mentionedOther.name);
          return;
        }

        // 沒提到其他文物 -> 繼續使用目前的鎖定文物回答
        await executeChatStream(question, activeArtifact);
        return;
      }

      // 2. 如果最初尚未鎖定文物，才走 RAG searchArtifacts 讓後端或使用者決定
      const searchResult = await searchArtifacts(question, abortControllerRef.current.signal);
      const artifacts = searchResult.artifacts || [];

      if (artifacts.length === 0) {
        // 找不到特定文物 → 直接讓後端用「不指定文物」回答
        await executeChatStream(question, null);
      } else if (artifacts.length === 1) {
        // 只有一個文物 → 自動用這件
        const only = artifacts[0];
        await executeChatStream(question, only.name);
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
      await executeChatStream(pendingQuestion, artifact.name);
      setPendingQuestion(null);
      setCandidates([]);
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
      await executeChatStream(pendingQuestion, name);
      setPendingQuestion(null);
      setCandidates([]);
      setManualArtifact("");
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

  // 在首頁點選文物：設定為目前鎖定文物並準備對話
  const handleSelectCatalogArtifact = (name) => {
    setActiveArtifact(name);
    // 重置對話訊息 (獨立對話模式)
    setMessages([
      {
        role: "assistant",
        type: "text",
        content: `您好！您選擇的文物是「${name}」。可點選左下角 💡 瀏覽常見問題，或直接輸入您想了解的內容，我將為您導覽介紹。`,
        sources: [],
      }
    ]);
    setViewMode('chat');
    setIsCatalogOpen(false);
  };

  const startChatting = () => {
    setViewMode('chat');
  };
  
  // 篩選全部館藏
  const filteredCatalog = ALL_ARTIFACTS.filter(a => 
    a.name.toLowerCase().includes(catalogSearch.toLowerCase())
  );

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

  const exhibitionIntro = `《紅樓夢》是在追憶，追憶著盛清時期的貴族世家，富貴繁榮的似水年華。小說裡瀰漫著，作者曹雪芹（約1716-1763）對於往事的痴迷與回望，以及一聲一聲對於盛極而衰、家族敗落的嘆息。展覽以《紅樓夢》為題，以國立故宮博物院、國家圖書館及國立臺灣大學圖書館藏品為材，以「物」來讀小說，將帶領我們看見那段消逝的年華，看見《紅樓夢》的絕美與哀戚。

展覽以三個軸線展開。一、「大雅可觀」：呈現《紅樓夢》迷人之處，在於貴族階級精緻的物質文化，在於其為作者生於江寧織造世家的生命經驗，所顯現出的富貴榮華；二、「異物奇貨」：挑揀書中來自外國的舶來品，提示作者以此襯托賈府的時尚，提示正因作者身處的時代宮廷流行「洋貨」，所掀來的仿效風潮；三、「一番夢幻」：說明書中人物依據使用、相關聯物品的描繪，塑造出不同的性格與隱喻，塑造出女性短暫卻燦爛的生命姿態，令人憐惜與不捨。

《紅樓夢》離不開「情」，親情、愛情、友情、主僕之情。情又與「人」之間的互動，緊密絣織。人又透過「物」的點綴，顯得立體而有溫度。「物」，讓小說有了畫面，成為看得見的《紅樓夢》。`;

  return (
    <>
      <div className="iphone-wrapper">
        <div className="iphone-frame">
          <div className="iphone-screen">

            <div className="app-root">
              
              {viewMode === 'landing' ? (
                /* 展覽首頁視圖 */
                <>
                  <div className="landing-view">
                    <header className="landing-header">
                      <img src={exhibitionPoster} alt="Exhibition Poster" className="landing-poster" />
                    </header>

                    <div className="landing-info">
                      <h2 className="landing-title">看得見的紅樓夢</h2>
                      <div className="landing-meta">
                        <div className="meta-item"><span className="meta-label">類型</span> 常設展</div>
                        <div className="meta-item"><span className="meta-label">時間</span> 2024/05/17 ~ 2026/05/17</div>
                        <div className="meta-item"><span className="meta-label">地點</span> 國立故宮博物院 2F 203</div>
                      </div>
                    </div>

                    <nav className="landing-tabs">
                      <button 
                        className={`tab-btn ${landingTab === 'intro' ? 'active' : ''}`}
                        onClick={() => setLandingTab('intro')}
                      >
                        簡介
                      </button>
                      <button 
                        className={`tab-btn ${landingTab === 'catalog' ? 'active' : ''}`}
                        onClick={() => setLandingTab('catalog')}
                      >
                        文物清單
                      </button>
                    </nav>

                    <div className="landing-content">
                      {landingTab === 'intro' ? (
                        <div className="intro-text">
                          {renderTextWithBreaks(exhibitionIntro)}
                        </div>
                      ) : (
                        <div className="catalog-tab-content">
                          <div className="catalog-grid inline-grid">
                            {ALL_ARTIFACTS.map(a => (
                              <button 
                                key={a.id} 
                                className={`catalog-card ${activeArtifact === a.name ? 'selected' : ''}`} 
                                onClick={() => handleSelectCatalogArtifact(a.name)}
                              >
                                <div className="catalog-img-wrapper">
                                  <img src={a.image} alt={a.name} loading="lazy" />
                                </div>
                                <span className="catalog-name">{a.name}</span>
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {landingTab === 'catalog' && (
                    <button className="floating-chat-btn" onClick={startChatting} title="開始詢問導覽員">
                      <div className="chat-btn-pulse"></div>
                      <img src={aiAvatar} alt="AI Guide" className="chat-btn-avatar" />
                    </button>
                  )}
                </>
              ) : (
                /* 原有的聊天視圖 */
                <>
                  <header className="hero">
                    <button className="back-to-landing" onClick={() => setViewMode('landing')} title="返回展覽">
                      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M15 18l-6-6 6-6" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    </button>
                    <h1>{activeArtifact ? activeArtifact : "博物館 AI 導覽系統"}</h1>
                    <button
                      className={`artifact-menu-btn ${activeArtifact ? "has-artifact" : ""}`}
                      onClick={() => {
                        setIsCatalogOpen(true);
                        setWaitingForPromptMenu(!!activeArtifact);
                      }}
                      title={activeArtifact ? `目前：${activeArtifact} （點擊切換）` : "選擇文物"}
                    >
                      {activeArtifact ? (
                        <>
                          <img
                            src={ALL_ARTIFACTS.find(a => a.name === activeArtifact)?.image || ""}
                            alt={activeArtifact}
                            className="artifact-menu-thumb"
                            onError={e => { e.target.style.display = "none"; }}
                          />
                        </>
                      ) : (
                        <>
                          <span className="artifact-menu-fallback-icon">🏛️</span>
                        </>
                      )}
                      <span className="selector-chevron">▾</span>
                    </button>
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
                            <img src={aiAvatar} alt="AI" className="custom-avatar-img" />
                          </div>
                        )}

                        {/* Avatar for User */}
                        {m.role === "user" && (
                          <div className="avatar-container user-avatar">
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className="custom-user-svg">
                              <path d="M12 12C14.21 12 16 10.21 16 8C16 5.79 14.21 4 12 4C9.79 4 8 5.79 8 8C8 10.21 9.79 12 12 12ZM12 14C9.33 14 4 15.34 4 18V20H20V18C20 15.34 14.67 14 12 14Z" fill="#333333"/>
                            </svg>
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
                                            <SourceItemCard key={sIdx} source={s} />
                                          ))}
                                        </div>
                                      )}
                                    </div>
                                  )}
                              </div>
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


                            </div>
                          )}

                          {/* 如果是助手的最後一則訊息且有候選文物，且不在載入中，則直接在 Bubble 內顯示 */}
                          {!loading && m.role === "assistant" && idx === messages.length - 1 && candidates.length > 0 && (
                            <div className="artifact-bubble-integration">
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
                          <img src={aiAvatar} alt="AI" className="custom-avatar-img" />
                        </div>
                        <div className="msg-content-wrapper">
                          <div className="msg-bubble loading-bubble">
                            <span>
                              {loadingStatus === "imaging" 
                                ? "正在為您生成情境圖" 
                                : loadingStatus === "answering" 
                                  ? "正在整理導覽內容" 
                                  : "正在為您檢索館藏資料中"}
                            </span>
                            <div className="typing-indicator">
                              <span></span>
                              <span></span>
                              <span></span>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    <div ref={messagesEndRef} />
                  </div>
                </div>

                <div className="chat-footer-wrapper">
                  {isPromptMenuOpen && (
                    <div className="prompt-menu-overlay" onClick={() => setIsPromptMenuOpen(false)}>
                      <div className="prompt-menu-content" onClick={e => e.stopPropagation()}>
                        <div className="prompt-menu-header">
                          <div className="prompt-header-title">
                            <span className="prompt-header-icon">
                              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 18h6"></path><path d="M10 22h4"></path><path d="M15.09 14c.18-.98.65-1.74 1.41-2.5A4.65 4.65 0 0 0 18 8 6 6 0 0 0 6 8c0 1 .23 2.23 1.5 3.5A4.61 4.61 0 0 1 8.91 14"></path></svg>
                            </span>
                            常見問題
                          </div>
                          <button className="prompt-menu-close" onClick={() => setIsPromptMenuOpen(false)}>×</button>
                        </div>
                        <div className="prompt-menu-list">
                          {GUIDED_PROMPTS.map((prompt, i) => (
                            <button key={i} className="prompt-menu-item" onClick={() => {
                              if (activeArtifact) {
                                setInput(prompt.text);
                              } else {
                                setInput(`關於【請在此輸入文物名稱】，${prompt.text}`);
                              }
                              setIsPromptMenuOpen(false);
                            }}>
                              <span className="prompt-icon">{prompt.icon}</span>
                              <div className="prompt-text-group">
                                <span className="prompt-label">{prompt.label}</span>
                                <span className="prompt-full">{prompt.text}</span>
                              </div>
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}
                  <div className="input-bar">
                    <button 
                      className="prompt-toggle-btn" 
                      onClick={() => {
                        if (!activeArtifact) {
                          // Guard: only add message if last message isn't already this hint
                          setMessages(prev => {
                            const lastMsg = prev[prev.length - 1];
                            if (lastMsg?.content?.includes("您還沒選擇文物")) return prev;
                            return [
                              ...prev,
                              {
                                role: "assistant",
                                type: "text",
                                content: "您還沒選擇文物喔！請先挑選一件感興趣的展品，我再來為您提供專屬的探索建議 ✨"
                              }
                            ];
                          });
                          setIsCatalogOpen(true);
                          setWaitingForPromptMenu(true);
                          return;
                        }
                        setIsPromptMenuOpen(!isPromptMenuOpen);
                      }}
                      title="常見問題"
                      disabled={loading}
                    >
                      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ pointerEvents: 'none' }}>
                        <path d="M9 18h6"></path>
                        <path d="M10 22h4"></path>
                        <path d="M15.09 14c.18-.98.65-1.74 1.41-2.5A4.65 4.65 0 0 0 18 8 6 6 0 0 0 6 8c0 1 .23 2.23 1.5 3.5A4.61 4.61 0 0 1 8.91 14"></path>
                      </svg>
                    </button>
                    <textarea
                      placeholder="請輸入您的問題..."
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

              {/* 館藏圖鑑 Modal (Bottom Sheet / Full screen) 必須放在這裡，才會被手機邊框包住 */}
              {isCatalogOpen && (
                <div className="catalog-overlay" onClick={() => setIsCatalogOpen(false)}>
                  <div className="catalog-content" onClick={e => e.stopPropagation()}>
                    <div className="catalog-header">
                      <div className="catalog-title">文物清單</div>
                      <button className="catalog-close-btn" onClick={() => setIsCatalogOpen(false)} title="關閉">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                          <line x1="18" y1="6" x2="6" y2="18"></line>
                          <line x1="6" y1="6" x2="18" y2="18"></line>
                        </svg>
                      </button>
                    </div>
                    <div className="catalog-grid inline-grid">
                      {filteredCatalog.map(a => (
                        <button key={a.id} className={`catalog-card ${activeArtifact === a.name ? 'selected' : ''}`} onClick={() => handleSelectCatalogArtifact(a.name)}>
                          <div className="catalog-img-wrapper">
                            <img src={a.image} alt={a.name} loading="lazy" />
                          </div>
                          <span className="catalog-name">{a.name}</span>
                        </button>
                      ))}
                      {filteredCatalog.length === 0 && (
                        <div className="catalog-empty">
                          沒有找到符合「{catalogSearch}」的文物。
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}
                </>
              )}
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