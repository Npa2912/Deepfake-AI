
"use client";

import {
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
} from "react";
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  Bell,
  Check,
  CheckCircle2,
  ChevronDown,
  Clock3,
  Cpu,
  FileImage,
  FileVideo,
  FolderOpen,
  Image as ImageIcon,
  Info,
  Lightbulb,
  LoaderCircle,
  Menu,
  Play,
  ScanFace,
  Search,
  ShieldCheck,
  Sun,
  Trash2,
  UploadCloud,
  X,
} from "lucide-react";

type FileMode = "image" | "video";
type Prediction = "REAL" | "FAKE";

type PredictionResult = {
  prediction: Prediction;
  confidence: number;
  prob_fake: number;
  prob_real: number;
  model: string;
  face_detection?: {
    bbox?: number[];
    detection_score?: number;
    crop_size?: number[];
  };
  face_image?: string;
  gradcam_image?: string;
  heatmap_image?: string;
  explanation?: string;
};

type ApiResponse = {
  success: boolean;
  filename?: string;
  result: PredictionResult;
};

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  "http://127.0.0.1:8000";

const MAX_SIZE = 100 * 1024 * 1024;

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024)
    return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatPercent(value?: number) {
  return typeof value === "number"
    ? `${value.toFixed(2)}%`
    : "—";
}

function normalizeImage(value?: string) {
  if (!value) return "";
  if (
    value.startsWith("data:") ||
    value.startsWith("http://") ||
    value.startsWith("https://")
  ) {
    return value;
  }
  return `data:image/png;base64,${value}`;
}

export default function Home() {
  const [mode, setMode] = useState<FileMode>("image");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] =
    useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const [duration, setDuration] = useState<number | null>(
    null
  );
  const [menuOpen, setMenuOpen] = useState(false);
  const [exampleIndex, setExampleIndex] = useState(0);

  const inputRef = useRef<HTMLInputElement>(null);
  const previewRef = useRef("");
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    return () => {
      if (previewRef.current) {
        URL.revokeObjectURL(previewRef.current);
      }
    };
  }, []);

  function clearFile() {
    if (previewRef.current) {
      URL.revokeObjectURL(previewRef.current);
      previewRef.current = "";
    }

    setFile(null);
    setPreview("");
    setResult(null);
    setError("");
    setDuration(null);

    if (inputRef.current) {
      inputRef.current.value = "";
    }
  }

  function changeMode(next: FileMode) {
    setMode(next);
    clearFile();
  }

  function handleFile(selected?: File) {
    if (!selected) return;

    const isImage =
      selected.type.startsWith("image/") ||
      /\.(jpg|jpeg|png|webp|jfif)$/i.test(
        selected.name
      );

    const isVideo =
      selected.type.startsWith("video/") ||
      /\.(mp4|mov)$/i.test(selected.name);

    if (mode === "image" && !isImage) {
      setError("Vui lòng chọn ảnh JPG, PNG hoặc WEBP.");
      return;
    }

    if (mode === "video" && !isVideo) {
      setError("Vui lòng chọn video MP4 hoặc MOV.");
      return;
    }

    if (selected.size > MAX_SIZE) {
      setError("Dung lượng tệp không được vượt quá 100 MB.");
      return;
    }

    if (previewRef.current) {
      URL.revokeObjectURL(previewRef.current);
    }

    const url = URL.createObjectURL(selected);
    previewRef.current = url;

    setFile(selected);
    setPreview(url);
    setResult(null);
    setError("");
    setDuration(null);
  }

  function onInputChange(
    event: ChangeEvent<HTMLInputElement>
  ) {
    handleFile(event.target.files?.[0]);
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    handleFile(event.dataTransfer.files?.[0]);
  }

  async function analyze() {
    if (!file || loading) return;

    if (mode === "video") {
      setError(
        "Backend hiện mới hỗ trợ phân tích ảnh. Chức năng video sẽ được tích hợp tiếp."
      );
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const started = performance.now();

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(
        `${API_URL}/api/predict`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Không thể phân tích ảnh."
        );
      }

      const payload = data as ApiResponse;

      if (!payload.result) {
        throw new Error("API không trả về kết quả hợp lệ.");
      }

      setResult(payload.result);
      setDuration(
        (performance.now() - started) / 1000
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Đã xảy ra lỗi khi kết nối máy chủ."
      );
    } finally {
      setLoading(false);
    }
  }

  const faceImage = normalizeImage(result?.face_image);

  const heatmapImage = normalizeImage(
    result?.gradcam_image || result?.heatmap_image
  );

  const fakeProbability = result?.prob_fake ?? 0;
  const realProbability = result?.prob_real ?? 0;

  const isFake = result?.prediction === "FAKE";

  const explanation = result
    ? result.explanation ||
      `ConvNeXt-Tiny V2 dự đoán ${
        isFake ? "FAKE" : "REAL"
      } với độ tin cậy ${formatPercent(
        result.confidence
      )}. ${
        heatmapImage
          ? "Bản đồ Grad-CAM thể hiện các vùng ảnh đóng góp vào dự đoán."
          : "Bản đồ Grad-CAM và phân tích dấu hiệu hình ảnh sẽ được bổ sung sau."
      }`
    : "Tải tệp và chạy phân tích để xem kết quả tại đây.";

  return (
    <div className="site">
      <header className="topbar">
        <div className="topbar-inner">
          <a className="brand" href="#">
            <div className="brand-symbol">
              <ScanFace size={29} strokeWidth={2.1} />
            </div>
            <div>
              <div className="brand-name">
                Deepfake AI
              </div>
              <div className="brand-caption">
                IMAGE &amp; VIDEO FORENSICS
              </div>
            </div>
          </a>

          <nav className="desktop-nav">
            <a href="#analysis" className="active">
              Phân tích
            </a>
            <a href="#about">Giới thiệu</a>
            <a href="#documentation">Tài liệu</a>
            <a href="#contact">Liên hệ</a>
          </nav>

          <div className="header-actions">
            <button
              className="icon-button"
              title="Giao diện sáng"
              aria-label="Giao diện sáng"
            >
              <Sun size={22} />
            </button>

            <button
              className="icon-button notification"
              title="Thông báo"
              aria-label="Thông báo"
            >
              <Bell size={21} />
              <span className="notification-dot" />
            </button>

            <div className="profile">
              <div className="avatar">N</div>
              <div className="profile-text">
                <strong>Nguyễn Phúc An</strong>
                <small>Research Project</small>
              </div>
              <ChevronDown size={16} />
            </div>

            <button
              className="mobile-menu-button"
              onClick={() => setMenuOpen(!menuOpen)}
              aria-label="Mở menu"
            >
              {menuOpen ? <X /> : <Menu />}
            </button>
          </div>
        </div>

        {menuOpen && (
          <nav className="mobile-nav">
            <a href="#analysis" onClick={() => setMenuOpen(false)}>
              Phân tích
            </a>
            <a href="#about" onClick={() => setMenuOpen(false)}>
              Giới thiệu
            </a>
            <a href="#documentation" onClick={() => setMenuOpen(false)}>
              Tài liệu
            </a>
            <a href="#contact" onClick={() => setMenuOpen(false)}>
              Liên hệ
            </a>
          </nav>
        )}
      </header>

      <main className="container" id="analysis">
        <section className="hero">
          <div className="hero-content">
            <span className="eyebrow">AI FORENSICS</span>

            <h1>
              Phát hiện ảnh và video{" "}
              <span>Deepfake</span>
            </h1>

            <p>
              Sử dụng mô hình ConvNeXt-Tiny V2 kết hợp
              phát hiện khuôn mặt YuNet để phân tích
              và dự đoán nội dung thật hoặc giả mạo,
              kèm theo bản đồ nhiệt Grad-CAM và giải
              thích kết quả.
            </p>
          </div>

          <div className="technology">
            <div className="tech-card">
              <div className="tech-icon">
                <Cpu size={26} />
              </div>
              <div>
                <strong>ConvNeXt-Tiny V2</strong>
                <small>Phân loại Real / Fake</small>
              </div>
            </div>

            <div className="tech-card">
              <div className="tech-icon">
                <ScanFace size={26} />
              </div>
              <div>
                <strong>YuNet (ONNX)</strong>
                <small>Phát hiện khuôn mặt</small>
              </div>
            </div>

            <div className="tech-card">
              <div className="tech-icon">
                <Activity size={26} />
              </div>
              <div>
                <strong>Grad-CAM</strong>
                <small>Giải thích kết quả</small>
              </div>
            </div>
          </div>
        </section>

        <section className="workspace">
          <div className="panel upload-panel">
            <div className="section-heading">
              <div className="section-heading-left">
                <span className="step">01</span>
                <h2>Tải ảnh hoặc video</h2>
              </div>
              <span className="support-text">
                JPG, PNG, WEBP, MP4, MOV · tối đa 100 MB
              </span>
            </div>

            <div className="mode-tabs">
              <button
                className={mode === "image" ? "selected" : ""}
                onClick={() => changeMode("image")}
              >
                <ImageIcon size={19} />
                Ảnh
              </button>
              <button
                className={mode === "video" ? "selected" : ""}
                onClick={() => changeMode("video")}
              >
                <Play size={19} />
                Video
              </button>
            </div>

            <div className="upload-grid">
              <div
                className={`dropzone ${
                  dragging ? "dragging" : ""
                }`}
                onDragOver={(event) => {
                  event.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
                onClick={() => inputRef.current?.click()}
              >
                <input
                  ref={inputRef}
                  type="file"
                  accept={
                    mode === "image"
                      ? ".jpg,.jpeg,.png,.webp,.jfif,image/*"
                      : ".mp4,.mov,video/mp4,video/quicktime"
                  }
                  hidden
                  onChange={onInputChange}
                />

                <div className="dropzone-content">
                  <div className="upload-icon">
                    <UploadCloud size={39} strokeWidth={1.7} />
                  </div>
                  <strong>Kéo thả tệp tin vào đây</strong>
                  <span>hoặc nhấn để chọn tệp tin</span>

                  <button
                    type="button"
                    className="choose-button"
                    onClick={(event) => {
                      event.stopPropagation();
                      inputRef.current?.click();
                    }}
                  >
                    <FolderOpen size={19} />
                    Chọn tệp tin
                  </button>

                  <small>
                    {mode === "image"
                      ? "JPG, PNG, WEBP (tối đa 100 MB)"
                      : "MP4, MOV (tối đa 100 MB)"}
                  </small>
                </div>
              </div>

              <div className="preview-card">
                <div className="preview-heading">
                  <strong>Xem trước</strong>
                  <button
                    onClick={clearFile}
                    disabled={!file}
                    className="remove-button"
                  >
                    <Trash2 size={15} />
                    Xóa
                  </button>
                </div>

                <div className="preview-image">
                  {preview ? (
                    mode === "image" ? (
                      <img
                        src={preview}
                        alt="Ảnh xem trước"
                      />
                    ) : (
                      <video
                        ref={videoRef}
                        src={preview}
                        controls
                        playsInline
                      />
                    )
                  ) : (
                    <div className="empty-preview">
                      {mode === "image" ? (
                        <FileImage size={39} />
                      ) : (
                        <FileVideo size={39} />
                      )}
                      <span>Chưa chọn tệp</span>
                    </div>
                  )}
                </div>

                {file ? (
                  <div className="file-details">
                    <strong title={file.name}>
                      {file.name}
                    </strong>
                    <span>{formatBytes(file.size)}</span>
                  </div>
                ) : (
                  <div className="file-details">
                    <strong>Chưa có tệp</strong>
                    <span>Ảnh hoặc video sẽ hiển thị tại đây</span>
                  </div>
                )}
              </div>
            </div>

            {error && (
              <div className="error-message">
                <AlertTriangle size={17} />
                {error}
              </div>
            )}

            <button
              className="analyze-button"
              onClick={analyze}
              disabled={!file || loading}
            >
              {loading ? (
                <>
                  <LoaderCircle
                    className="spinner"
                    size={23}
                  />
                  Đang phân tích...
                </>
              ) : (
                <>
                  <Search size={24} />
                  Phân tích Deepfake
                  <ArrowRight size={21} />
                </>
              )}
            </button>
          </div>

          <div className="panel result-panel">
            <div className="section-heading">
              <div className="section-heading-left">
                <span className="step">02</span>
                <h2>Kết quả phân tích</h2>
              </div>

              <div className="result-meta">
                <span
                  className={`status ${
                    result ? "completed" : ""
                  }`}
                >
                  {result ? (
                    <CheckCircle2 size={15} />
                  ) : (
                    <Clock3 size={15} />
                  )}
                  {result ? "Hoàn thành" : "Chờ phân tích"}
                </span>

                {duration !== null && (
                  <span className="time">
                    <Clock3 size={15} />
                    Thời gian: {duration.toFixed(2)} giây
                  </span>
                )}
              </div>
            </div>

            <div className="prediction-grid">
              <div
                className={`prediction-card ${
                  result
                    ? isFake
                      ? "prediction-fake"
                      : "prediction-real"
                    : ""
                }`}
              >
                <div className="prediction-icon">
                  {result ? (
                    isFake ? (
                      <AlertTriangle size={31} />
                    ) : (
                      <ShieldCheck size={31} />
                    )
                  ) : (
                    <ScanFace size={31} />
                  )}
                </div>

                <div className="prediction-text">
                  <span>Dự đoán của mô hình</span>
                  <strong>
                    {result ? result.prediction : "—"}
                  </strong>
                  <p>
                    Độ tin cậy:{" "}
                    <b>
                      {result
                        ? formatPercent(result.confidence)
                        : "—"}
                    </b>
                  </p>
                </div>
              </div>

              <div className="probability-card">
                <div className="probability-item">
                  <div className="probability-label">
                    <span>Xác suất FAKE</span>
                    <strong>
                      {result
                        ? formatPercent(fakeProbability)
                        : "—"}
                    </strong>
                  </div>
                  <div className="progress-track">
                    <div
                      className="progress-fill fake-fill"
                      style={{
                        width: `${
                          result ? fakeProbability : 0
                        }%`,
                      }}
                    />
                  </div>
                </div>

                <div className="probability-item">
                  <div className="probability-label">
                    <span>Xác suất REAL</span>
                    <strong>
                      {result
                        ? formatPercent(realProbability)
                        : "—"}
                    </strong>
                  </div>
                  <div className="progress-track">
                    <div
                      className="progress-fill real-fill"
                      style={{
                        width: `${
                          result ? realProbability : 0
                        }%`,
                      }}
                    />
                  </div>
                </div>
              </div>
            </div>

            <div className="visual-grid">
              <div className="visual-card">
                <h3>Ảnh khuôn mặt đã crop</h3>
                <div className="visual-content">
                  {faceImage ? (
                    <img
                      src={faceImage}
                      alt="Khuôn mặt được YuNet crop"
                    />
                  ) : (
                    <div className="visual-placeholder">
                      <ScanFace size={28} />
                      <span>Chưa có ảnh</span>
                    </div>
                  )}
                </div>
                <div className="visual-caption">
                  {faceImage ? "224 × 224" : ""}
                </div>
              </div>

              <div className="visual-card">
                <h3>Grad-CAM Heatmap</h3>
                <div className="visual-content">
                  {heatmapImage ? (
                    <img
                      src={heatmapImage}
                      alt="Grad-CAM Heatmap"
                    />
                  ) : (
                    <div className="visual-placeholder">
                      <Activity size={28} />
                      <span>
                        {result
                          ? "Chờ tích hợp Grad-CAM"
                          : "Chưa có bản đồ nhiệt"}
                      </span>
                    </div>
                  )}
                </div>
                <div className="visual-caption">
                  {heatmapImage ? "224 × 224" : ""}
                </div>
              </div>

              <div className="visual-card model-card">
                <h3>Thông tin mô hình</h3>
                <div className="model-info">
                  <div>
                    <span>Mô hình</span>
                    <strong>
                      {result?.model || "ConvNeXt-Tiny V2"}
                    </strong>
                  </div>
                  <div>
                    <span>Phát hiện khuôn mặt</span>
                    <strong>YuNet (ONNX)</strong>
                  </div>
                  <div>
                    <span>Điểm phát hiện</span>
                    <strong>
                      {result?.face_detection?.detection_score ??
                        "—"}
                    </strong>
                  </div>
                  <div>
                    <span>Vùng khuôn mặt</span>
                    <strong>
                      {result?.face_detection?.bbox
                        ? `[${result.face_detection.bbox.join(
                            ", "
                          )}]`
                        : "—"}
                    </strong>
                  </div>
                  <div>
                    <span>Kích thước crop</span>
                    <strong>224 × 224</strong>
                  </div>
                  <div>
                    <span>Số lớp đầu ra</span>
                    <strong>2 (FAKE / REAL)</strong>
                  </div>
                  <div>
                    <span>Tiền xử lý</span>
                    <strong>ImageNet Normalize</strong>
                  </div>
                </div>
              </div>
            </div>

            <div className="explanation-section">
              <div className="explanation-title">
                <span className="step">03</span>
                <h2>Giải thích kết quả</h2>
              </div>

              <div className="explanation-box">
                <div className="explanation-icon">
                  <Lightbulb size={29} />
                </div>
                <p>{explanation}</p>
              </div>
            </div>
          </div>
        </section>

        <section className="examples-panel">
          <div className="examples-heading">
            <div className="examples-title">
              <div className="examples-icon">
                <ImageIcon size={21} />
              </div>
              <h2>Ví dụ phân tích</h2>
              <span>
                Một số ví dụ ảnh và video được phân tích
                bởi mô hình
              </span>
            </div>

            <div className="carousel-controls">
              <button
                onClick={() =>
                  setExampleIndex(Math.max(0, exampleIndex - 1))
                }
                disabled={exampleIndex === 0}
                aria-label="Trước"
              >
                <ArrowLeft size={19} />
              </button>
              <button
                onClick={() => setExampleIndex(exampleIndex + 1)}
                aria-label="Tiếp"
              >
                <ArrowRight size={19} />
              </button>
            </div>
          </div>

          <div className="examples-empty">
            <FileImage size={27} />
            <div>
              <strong>Thư viện ví dụ phân tích</strong>
              <p>
                Các ảnh và video mẫu sẽ được bổ sung từ
                dữ liệu kiểm thử thực tế, không sử dụng
                hình ảnh AI minh họa.
              </p>
            </div>
          </div>
        </section>

        <footer className="footer">
          <div>
            <strong>Deepfake AI</strong>
            <span>Image &amp; Video Forensics</span>
          </div>
          <p>
            Research Project · ConvNeXt-Tiny V2 · YuNet ·
            Grad-CAM
          </p>
        </footer>
      </main>

      <style jsx global>{`
        :root {
          --green: #164c35;
          --green-dark: #103e2b;
          --green-light: #eaf3ed;
          --text: #101820;
          --muted: #667580;
          --border: #dce4df;
          --surface: #ffffff;
          --background: #f5f7f5;
          --red: #dc3838;
        }

        * {
          box-sizing: border-box;
        }

        html {
          scroll-behavior: smooth;
        }

        body {
          margin: 0;
          background: var(--background);
          color: var(--text);
          font-family: Arial, Helvetica, sans-serif;
          -webkit-font-smoothing: antialiased;
        }

        button {
          font: inherit;
          cursor: pointer;
        }

        button:disabled {
          cursor: not-allowed;
        }

        .site {
          min-height: 100vh;
        }

        .topbar {
          height: 76px;
          background: white;
          border-bottom: 1px solid var(--border);
          position: sticky;
          top: 0;
          z-index: 50;
        }

        .topbar-inner {
          height: 100%;
          max-width: 1660px;
          padding: 0 38px;
          margin: auto;
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 24px;
        }

        .brand {
          display: flex;
          align-items: center;
          gap: 13px;
          color: inherit;
          text-decoration: none;
          flex-shrink: 0;
        }

        .brand-symbol {
          width: 42px;
          height: 42px;
          color: var(--green);
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .brand-name {
          font-size: 24px;
          font-weight: 800;
          letter-spacing: -0.9px;
        }

        .brand-caption {
          font-size: 10px;
          letter-spacing: 1.25px;
          margin-top: 4px;
          color: #465666;
        }

        .desktop-nav {
          display: flex;
          align-items: center;
          gap: 45px;
          height: 100%;
        }

        .desktop-nav a {
          height: 100%;
          display: flex;
          align-items: center;
          position: relative;
          text-decoration: none;
          color: #354252;
          font-size: 15px;
          white-space: nowrap;
        }

        .desktop-nav a.active {
          color: var(--green);
          font-weight: 700;
        }

        .desktop-nav a.active::after {
          content: "";
          position: absolute;
          height: 3px;
          background: var(--green);
          left: 0;
          right: 0;
          bottom: 13px;
          border-radius: 3px;
        }

        .header-actions {
          display: flex;
          align-items: center;
          gap: 20px;
        }

        .icon-button {
          width: 33px;
          height: 35px;
          border: 0;
          background: transparent;
          color: var(--green-dark);
          display: grid;
          place-items: center;
          position: relative;
        }

        .notification-dot {
          width: 9px;
          height: 9px;
          border-radius: 50%;
          background: #15803d;
          position: absolute;
          top: 2px;
          right: 1px;
          border: 2px solid white;
        }

        .profile {
          display: flex;
          align-items: center;
          gap: 11px;
          margin-left: 5px;
        }

        .avatar {
          width: 40px;
          height: 40px;
          border-radius: 50%;
          background: var(--green);
          color: white;
          font-weight: 700;
          font-size: 20px;
          display: grid;
          place-items: center;
        }

        .profile-text {
          display: flex;
          flex-direction: column;
          gap: 3px;
          white-space: nowrap;
        }

        .profile-text strong {
          font-size: 13px;
        }

        .profile-text small {
          font-size: 11px;
          color: var(--muted);
        }

        .mobile-menu-button,
        .mobile-nav {
          display: none;
        }

        .container {
          max-width: 1540px;
          margin: auto;
          padding: 25px 30px 30px;
        }

        .hero {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 28px;
          margin-bottom: 23px;
        }

        .hero-content {
          max-width: 740px;
        }

        .eyebrow {
          display: inline-block;
          padding: 6px 12px;
          border-radius: 8px;
          background: #e2ebe5;
          color: var(--green-dark);
          font-size: 12px;
          font-weight: 700;
          letter-spacing: 0.5px;
          margin-bottom: 13px;
        }

        .hero h1 {
          font-size: clamp(29px, 2.5vw, 43px);
          line-height: 1.16;
          letter-spacing: -1.3px;
          margin: 0 0 12px;
          font-weight: 800;
        }

        .hero h1 span {
          color: var(--green);
        }

        .hero p {
          font-size: 14px;
          color: #44515e;
          line-height: 1.75;
          margin: 0;
          max-width: 690px;
        }

        .technology {
          display: flex;
          gap: 12px;
          flex-shrink: 0;
        }

        .tech-card {
          min-width: 175px;
          display: flex;
          align-items: center;
          gap: 12px;
          border: 1px solid var(--border);
          background: white;
          padding: 12px;
          border-radius: 12px;
        }

        .tech-icon {
          color: var(--green);
          display: grid;
          place-items: center;
          flex-shrink: 0;
        }

        .tech-card strong {
          font-size: 12px;
          display: block;
          white-space: nowrap;
        }

        .tech-card small {
          font-size: 11px;
          color: var(--muted);
          display: block;
          margin-top: 6px;
          white-space: nowrap;
        }

        .workspace {
          display: grid;
          grid-template-columns: 0.95fr 1.05fr;
          align-items: stretch;
          gap: 16px;
        }

        .panel,
        .examples-panel {
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: 15px;
          box-shadow: 0 2px 1px rgba(0, 0, 0, 0.035);
        }

        .panel {
          padding: 19px;
          min-width: 0;
        }

        .section-heading,
        .section-heading-left,
        .result-meta {
          display: flex;
          align-items: center;
        }

        .section-heading {
          justify-content: space-between;
          gap: 12px;
          margin-bottom: 19px;
        }

        .section-heading-left {
          gap: 12px;
        }

        .step {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          background: var(--green);
          color: white;
          min-width: 39px;
          height: 30px;
          border-radius: 9px;
          font-size: 13px;
          font-weight: 800;
        }

        .section-heading h2,
        .explanation-title h2 {
          font-size: 17px;
          margin: 0;
          font-weight: 800;
          white-space: nowrap;
        }

        .support-text {
          font-size: 10px;
          color: var(--muted);
          text-align: right;
        }

        .mode-tabs {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 10px;
          margin-bottom: 15px;
        }

        .mode-tabs button {
          height: 47px;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 12px;
          background: #f7f9f7;
          border: 1px solid var(--border);
          border-radius: 8px;
          font-weight: 700;
          color: #18231e;
        }

        .mode-tabs button.selected {
          background: var(--green);
          color: white;
          border-color: var(--green);
        }

        .upload-grid {
          display: grid;
          grid-template-columns: 1.15fr 0.85fr;
          gap: 12px;
        }

        .dropzone {
          min-height: 306px;
          border: 1.5px dashed #9fb9a8;
          border-radius: 11px;
          display: flex;
          align-items: center;
          justify-content: center;
          text-align: center;
          background: #fbfcfb;
          cursor: pointer;
          padding: 15px;
        }

        .dropzone.dragging {
          border-color: var(--green);
          background: #edf6ef;
        }

        .dropzone-content {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 13px;
        }

        .upload-icon {
          color: var(--green);
          margin-bottom: 4px;
        }

        .dropzone-content strong {
          font-size: 15px;
        }

        .dropzone-content > span {
          color: var(--muted);
          font-size: 13px;
        }

        .choose-button {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 11px;
          background: var(--green);
          color: white;
          border: 0;
          padding: 13px 21px;
          min-width: 190px;
          border-radius: 8px;
          font-weight: 700;
          margin: 3px 0;
        }

        .dropzone-content small {
          color: var(--muted);
          font-size: 11px;
        }

        .preview-card {
          border: 1px solid var(--border);
          border-radius: 11px;
          padding: 12px;
          min-width: 0;
          display: flex;
          flex-direction: column;
        }

        .preview-heading {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 11px;
        }

        .preview-heading strong {
          font-size: 13px;
        }

        .remove-button {
          background: transparent;
          border: 0;
          color: #d83b3b;
          font-size: 12px;
          display: flex;
          align-items: center;
          gap: 4px;
        }

        .remove-button:disabled {
          opacity: 0.4;
        }

        .preview-image {
          height: 245px;
          background: #f1f4f2;
          border-radius: 9px;
          overflow: hidden;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .preview-image img,
        .preview-image video {
          width: 100%;
          height: 100%;
          object-fit: contain;
        }

        .empty-preview {
          color: #9aa9a0;
          display: flex;
          align-items: center;
          flex-direction: column;
          gap: 12px;
          font-size: 12px;
        }

        .file-details {
          margin-top: 12px;
          display: flex;
          flex-direction: column;
          gap: 5px;
          min-width: 0;
        }

        .file-details strong {
          font-size: 12px;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .file-details span {
          font-size: 11px;
          color: var(--muted);
        }

        .analyze-button {
          width: 100%;
          height: 58px;
          background: var(--green);
          border: 1px solid var(--green);
          color: white;
          border-radius: 9px;
          margin-top: 17px;
          font-size: 17px;
          font-weight: 800;
          display: flex;
          justify-content: center;
          align-items: center;
          gap: 15px;
        }

        .analyze-button:hover:not(:disabled),
        .choose-button:hover {
          background: var(--green-dark);
        }

        .analyze-button:disabled {
          opacity: 0.65;
        }

        .error-message {
          display: flex;
          gap: 8px;
          align-items: center;
          padding: 12px;
          margin-top: 13px;
          border: 1px solid #fecaca;
          border-radius: 8px;
          background: #fff4f4;
          color: #b91c1c;
          font-size: 12px;
          overflow-wrap: anywhere;
        }

        .spinner {
          animation: spin 1s linear infinite;
        }

        @keyframes spin {
          to {
            transform: rotate(360deg);
          }
        }

        .result-meta {
          gap: 12px;
        }

        .status {
          display: flex;
          align-items: center;
          gap: 6px;
          border-radius: 20px;
          background: #f0f2f0;
          color: #536257;
          padding: 7px 10px;
          font-size: 11px;
          white-space: nowrap;
        }

        .status.completed {
          background: #e7f3e9;
          color: #14743d;
        }

        .time {
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 11px;
          color: var(--muted);
          white-space: nowrap;
        }

        .prediction-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 12px;
          margin-bottom: 14px;
        }

        .prediction-card,
        .probability-card {
          border: 1px solid var(--border);
          border-radius: 11px;
          min-height: 132px;
          padding: 16px;
        }

        .prediction-card {
          display: flex;
          align-items: center;
          gap: 15px;
          background: #f9faf9;
        }

        .prediction-card.prediction-fake {
          background: #fff2f2;
          border-color: #fecaca;
        }

        .prediction-card.prediction-real {
          background: #eff8f1;
          border-color: #c7e8d0;
        }

        .prediction-icon {
          width: 61px;
          height: 61px;
          border-radius: 50%;
          background: #e8eee9;
          color: var(--green);
          display: grid;
          place-items: center;
          flex-shrink: 0;
        }

        .prediction-fake .prediction-icon {
          background: #fee2e2;
          color: var(--red);
        }

        .prediction-real .prediction-icon {
          background: #d8f0df;
          color: #14743d;
        }

        .prediction-text > span {
          color: var(--muted);
          font-size: 12px;
        }

        .prediction-text > strong {
          display: block;
          font-size: 36px;
          letter-spacing: -1px;
          margin: 5px 0;
        }

        .prediction-fake .prediction-text > strong {
          color: #d62f2f;
        }

        .prediction-real .prediction-text > strong {
          color: #16723c;
        }

        .prediction-text p {
          font-size: 12px;
          margin: 0;
        }

        .prediction-text b {
          font-size: 15px;
        }

        .probability-card {
          display: flex;
          flex-direction: column;
          justify-content: center;
          gap: 19px;
        }

        .probability-label {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 10px;
          font-size: 12px;
          margin-bottom: 10px;
        }

        .probability-label strong {
          font-size: 13px;
        }

        .progress-track {
          height: 11px;
          background: #e4e8e5;
          border-radius: 10px;
          overflow: hidden;
        }

        .progress-fill {
          height: 100%;
          border-radius: 10px;
          transition: width 0.4s ease;
        }

        .fake-fill {
          background: #df4242;
        }

        .real-fill {
          background: #23854e;
        }

        .visual-grid {
          display: grid;
          grid-template-columns: 1fr 1fr 1.23fr;
          gap: 11px;
        }

        .visual-card {
          border: 1px solid var(--border);
          border-radius: 10px;
          padding: 11px;
          min-width: 0;
        }

        .visual-card h3 {
          font-size: 12px;
          margin: 0 0 10px;
          font-weight: 800;
        }

        .visual-content {
          width: 100%;
          height: 157px;
          border-radius: 8px;
          background: #f2f5f3;
          overflow: hidden;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .visual-content img {
          height: 100%;
          width: 100%;
          object-fit: contain;
        }

        .visual-placeholder {
          color: #8e9c94;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 9px;
          text-align: center;
          font-size: 11px;
          padding: 8px;
        }

        .visual-caption {
          height: 18px;
          margin-top: 6px;
          color: var(--muted);
          font-size: 11px;
          text-align: center;
        }

        .model-info {
          display: flex;
          flex-direction: column;
          gap: 11px;
        }

        .model-info > div {
          display: flex;
          justify-content: space-between;
          gap: 9px;
          align-items: flex-start;
          font-size: 10px;
        }

        .model-info span {
          color: var(--muted);
        }

        .model-info strong {
          font-size: 10px;
          font-weight: 700;
          text-align: right;
          overflow-wrap: anywhere;
        }

        .explanation-section {
          margin-top: 15px;
        }

        .explanation-title {
          display: flex;
          align-items: center;
          gap: 12px;
          margin-bottom: 12px;
        }

        .explanation-box {
          display: flex;
          align-items: center;
          gap: 13px;
          background: #edf5ef;
          border-radius: 10px;
          padding: 12px;
        }

        .explanation-icon {
          color: #168043;
          width: 46px;
          display: grid;
          place-items: center;
          flex-shrink: 0;
        }

        .explanation-box p {
          margin: 0;
          border-left: 3px solid var(--green);
          padding-left: 14px;
          font-size: 12px;
          line-height: 1.7;
        }

        .examples-panel {
          margin-top: 18px;
          padding: 17px 19px;
        }

        .examples-heading,
        .examples-title,
        .carousel-controls {
          display: flex;
          align-items: center;
        }

        .examples-heading {
          justify-content: space-between;
          gap: 16px;
        }

        .examples-title {
          gap: 13px;
          flex-wrap: wrap;
        }

        .examples-icon {
          background: var(--green);
          color: white;
          width: 42px;
          height: 39px;
          border-radius: 9px;
          display: grid;
          place-items: center;
        }

        .examples-title h2 {
          font-size: 18px;
          margin: 0;
        }

        .examples-title > span {
          font-size: 12px;
          color: var(--muted);
          margin-left: 7px;
        }

        .carousel-controls {
          gap: 8px;
        }

        .carousel-controls button {
          width: 37px;
          height: 37px;
          border: 1px solid var(--border);
          background: white;
          border-radius: 9px;
          display: grid;
          place-items: center;
        }

        .carousel-controls button:disabled {
          opacity: 0.35;
        }

        .examples-empty {
          min-height: 105px;
          margin-top: 15px;
          border: 1px dashed var(--border);
          border-radius: 10px;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 14px;
          color: #68786e;
          padding: 15px;
        }

        .examples-empty strong {
          color: #32443a;
          font-size: 13px;
        }

        .examples-empty p {
          font-size: 12px;
          margin: 6px 0 0;
          line-height: 1.5;
        }

        .footer {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 20px;
          padding: 25px 0 5px;
          color: var(--muted);
          font-size: 11px;
        }

        .footer > div {
          display: flex;
          gap: 13px;
          align-items: center;
        }

        .footer strong {
          color: var(--green);
          font-size: 13px;
        }

        .footer p {
          margin: 0;
        }

        @media (max-width: 1400px) {
          .technology {
            gap: 7px;
          }

          .tech-card {
            min-width: 0;
            padding: 10px;
          }

          .tech-icon {
            display: none;
          }

          .hero-content {
            max-width: 650px;
          }

          .container {
            padding-left: 22px;
            padding-right: 22px;
          }
        }

        @media (max-width: 1100px) {
          .hero {
            flex-direction: column;
            align-items: flex-start;
          }

          .technology {
            width: 100%;
          }

          .tech-card {
            flex: 1;
          }

          .workspace {
            grid-template-columns: 1fr;
          }

          .visual-content {
            height: 190px;
          }
        }

        @media (max-width: 800px) {
          .topbar-inner {
            padding: 0 16px;
          }

          .desktop-nav,
          .profile,
          .header-actions > .icon-button {
            display: none;
          }

          .mobile-menu-button {
            display: grid;
            place-items: center;
            border: 0;
            background: transparent;
            color: var(--green);
          }

          .mobile-nav {
            display: flex;
            flex-direction: column;
            background: white;
            padding: 14px 20px;
            border-bottom: 1px solid var(--border);
            gap: 17px;
          }

          .mobile-nav a {
            color: var(--green);
            text-decoration: none;
            font-weight: 600;
          }

          .container {
            padding: 20px 13px;
          }

          .hero h1 {
            font-size: 30px;
          }

          .technology {
            flex-wrap: wrap;
          }

          .tech-card {
            flex: 1 1 150px;
          }

          .panel {
            padding: 14px;
          }

          .support-text {
            display: none;
          }

          .upload-grid,
          .prediction-grid {
            grid-template-columns: 1fr;
          }

          .dropzone {
            min-height: 220px;
          }

          .preview-image {
            height: 230px;
          }

          .visual-grid {
            grid-template-columns: 1fr 1fr;
          }

          .model-card {
            grid-column: 1 / -1;
          }

          .visual-content {
            height: 170px;
          }

          .examples-title > span {
            display: none;
          }

          .footer {
            flex-direction: column;
            align-items: flex-start;
          }
        }

        @media (max-width: 420px) {
          .brand-name {
            font-size: 20px;
          }

          .brand-caption {
            font-size: 8px;
          }

          .section-heading h2 {
            font-size: 15px;
          }

          .result-meta .time {
            display: none;
          }

          .prediction-text > strong {
            font-size: 29px;
          }

          .visual-grid {
            gap: 7px;
          }

          .visual-card {
            padding: 8px;
          }

          .visual-content {
            height: 140px;
          }
        }
      `}</style>
    </div>
  );
}
