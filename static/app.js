const modelSelect = document.getElementById("modelSelect");
const videoSelect = document.getElementById("videoSelect");
const imageList = document.getElementById("imageList");
const loadModelBtn = document.getElementById("loadModelBtn");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const reportBtn = document.getElementById("reportBtn");
const downloadImageBtn = document.getElementById("downloadImageBtn");
const languageSelect = document.getElementById("languageSelect");
const runState = document.getElementById("runState");
const toast = document.getElementById("toast");
const chartTooltip = document.getElementById("chartTooltip");
const chart = document.getElementById("chart3d");
const ctx = chart.getContext("2d");
const qualityRadarCanvas = document.getElementById("qualityRadarChart");
const qualityRadarCtx = qualityRadarCanvas.getContext("2d");
const modelRadarCanvas = document.getElementById("modelRadarChart");
const modelRadarCtx = modelRadarCanvas.getContext("2d");
const gradeChart = document.getElementById("gradeChart");
const gradeChartCtx = gradeChart.getContext("2d");
const violinChart = document.getElementById("violinChart");
const violinChartCtx = violinChart.getContext("2d");
const modal = document.getElementById("analysisModal");
const modalClose = document.getElementById("modalClose");
const modalRadar = document.getElementById("modalRadar");
const modalRadarCtx = modalRadar.getContext("2d");
const binInspector = document.getElementById("binInspector");
const binSampleImage = document.getElementById("binSampleImage");
const binDrawCanvas = document.getElementById("binDrawCanvas");
const binDrawCtx = binDrawCanvas.getContext("2d");
const binSampleTitle = document.getElementById("binSampleTitle");
const binSampleMeta = document.getElementById("binSampleMeta");
const binDrawStats = document.getElementById("binDrawStats");
const binResetBtn = document.getElementById("binResetBtn");

const sizeOrder = ["<0.15%", "0.15-0.25%", "0.25-0.5%", "0.5-1%", "1-2%", "2-4%", "4-8%", ">=8%"];
const sizeClassOrder = ["XXS", "XS", "S", "M", "L", "XL", "XXL", "XXXL"];
const confOrder = ["0.00-0.30", "0.30-0.50", "0.50-0.70", "0.70-0.85", "0.85-1.01"];
let latestStatus = null;
let latestAssets = { models: [], videos: [], images: [] };
let currentLang = localStorage.getItem("assessment_lang") || "zh";
let matrixCells = [];
let currentBinSample = null;
let drawingRect = null;
let isDrawingBinRect = false;
let gradeHoverRegions = [];
let violinHoverRegions = [];
let resizeFrame = null;

const CHART_HEIGHTS = {
  matrix: 460,
  radar: 320,
  grade: 440,
  violin: 520,
};

const I18N = {
  zh: {
    app_title: "YOLO 推論評估",
    app_subtitle: "量化影片/影像品質、bbox 尺寸、輪廓清晰度與模型感知信心的關係",
    run_setup: "執行設定", model_source: "模型 / 來源", tflite_model: "TFLite 模型", load: "載入",
    image_source: "影像來源", video_file: "影片檔", webcam: "Webcam", image_files: "圖片檔",
    video: "影片", webcam_index: "Webcam index", image_manager: "圖片檔案管理", images_path: "路徑: images/",
    download_url: "公開圖片 URL", download_image: "下載到 images/", image_tracking_note: "圖片是離散樣本；不評估 tracking 品質。",
    conf_threshold: "Conf 門檻", frame_stride: "Frame stride", start_analysis: "開始分析", stop: "停止",
    report_name: "報告名稱", tags: "Tags", export_report: "匯出報告",
    object_samples: "物件樣本", avg_conf: "平均 Conf", avg_bbox_px: "平均 BBox %", sharpness: "清晰度", recommended: "建議比例",
    live_view: "即時感知畫面", matrix_title: "BBox Frame 佔比 / Confidence 矩陣", matrix_subtitle: "列: conf · 欄: bbox 面積佔整張 frame 的比例",
    quality_radar: "影像品質雷達圖", quality_radar_sub: "整體畫面品質", model_radar: "模型感知雷達圖", model_radar_sub: "偵測信心與 bbox 品質",
    bbox_metrics: "目前 BBox 指標", person_only: "只統計 person detections", track: "Track", conf: "Conf", approx_size: "Frame %",
    actual_size: "實際 W x H", sharp: "Sharp", edge: "Edge", brightness: "Brightness", contrast: "Contrast", grade: "評級",
    classic_samples: "經典樣本", classic_samples_sub: "每個階段的代表性 bbox 與條件",
    bin_inspector: "Bin 互動抽樣", bin_inspector_sub: "點選矩陣 cell 後，可在原始 frame 上拖拉矩形理解 bbox 佔比",
    reset_canvas: "清除畫板", selected_bin: "選取 bin", drag_hint: "拖拉矩形後顯示 frame 佔比與像素 W,H",
    grade_distribution: "A/B/C/D 評級分佈", grade_distribution_sub: "即時模型感知評級數量與比例",
    violin_distribution: "指標分佈圖", violin_distribution_sub: "目前樣本的 mean、std 與百分位分佈",
    grade_tip: "Video/Webcam: 40% Conf + 30% 清晰度 + 20% Tracking 穩定 + 10% BBox size。Images: 45% Conf + 40% 清晰度 + 15% BBox size。A>=75, B>=60, C>=45, D<45。",
    classic_samples_purpose: "Classic Samples 是代表性品質案例：低輪廓、小目標、推薦品質。它不是特定 ID 穩定性抽樣；tracking 穩定性請看 Product Spec / tracking 指標。",
    no_bbox: "目前沒有 bbox 樣本", downloaded: "圖片已下載", analysis_started: "分析已開始", stop_requested: "已要求停止", loaded: "已載入",
    quality_labels: ["清晰", "曝光", "對比", "Lux", "穩定", "邊緣"],
    model_labels: ["Conf均值", "Conf覆蓋", "BBox輪廓", "Tracking", "BBox尺寸", "低模糊", "邊緣覆蓋", "A/B覆蓋"],
    quality_tips: [
      "🔎 整體畫面銳利程度。越高越好。",
      "☀️ 亮度接近合理中間值。越高越好。",
      "◐ 明暗差與紋理可分辨度。越高越好。",
      "💡 相對照度 proxy，非校正 lux。越高通常越好。",
      "🎥 動態模糊越少越穩。越高越好。",
      "〰️ 畫面邊緣結構密度。越高代表輪廓資訊越多。",
    ],
    model_tips: [
      "🎯 偵測 confidence 平均值。越高越好。",
      "📈 conf >= 0.50 的樣本比例。越高代表可用區間越大。",
      "👤 bbox 內人物輪廓清晰度。越高越好。",
      "🧭 追蹤穩定度：ID age、連續度、中心點軌跡平滑度。越高越好。",
      "📐 bbox 尺寸覆蓋，太小會降低感知。越高越好。",
      "🧊 低模糊 bbox 比例。越高越好。",
      "〰️ bbox 內有效邊緣覆蓋。越高越好。",
      "✅ A/B 等級樣本比例。越高越接近可交付標準。",
    ],
  },
  en: {
    app_title: "YOLO Inference Assessment",
    app_subtitle: "Quantify image quality, bbox size, contour clarity, and model confidence.",
    run_setup: "Run Setup", model_source: "Model / Source", tflite_model: "TFLite model", load: "Load",
    image_source: "Image source", video_file: "Video file", webcam: "Webcam", image_files: "Image files",
    video: "Video", webcam_index: "Webcam index", image_manager: "Images File Manager", images_path: "Path: images/",
    download_url: "Public image URL", download_image: "Download to images/", image_tracking_note: "Image files are discrete samples; tracking quality is not evaluated.",
    conf_threshold: "Conf threshold", frame_stride: "Frame stride", start_analysis: "Start Analysis", stop: "Stop",
    report_name: "Report name", tags: "Tags", export_report: "Export Report",
    object_samples: "Object Samples", avg_conf: "Avg Conf", avg_bbox_px: "Avg BBox %", sharpness: "Sharpness", recommended: "Recommended",
    live_view: "Live Perception View", matrix_title: "BBox Frame Ratio / Confidence Matrix", matrix_subtitle: "Rows: conf · Columns: bbox area as percent of the whole frame",
    quality_radar: "Image Quality Radar", quality_radar_sub: "Whole-frame quality", model_radar: "Model Perception Radar", model_radar_sub: "Detection confidence and bbox quality",
    bbox_metrics: "Current BBox Metrics", person_only: "Person detections only", track: "Track", conf: "Conf", approx_size: "Frame %",
    actual_size: "Actual W x H", sharp: "Sharp", edge: "Edge", brightness: "Brightness", contrast: "Contrast", grade: "Grade",
    classic_samples: "Classic Samples", classic_samples_sub: "Representative bbox examples and conditions",
    bin_inspector: "Bin Sample Inspector", bin_inspector_sub: "Click a matrix cell, then drag a rectangle on the sampled frame to understand bbox scale",
    reset_canvas: "Reset Canvas", selected_bin: "Selected bin", drag_hint: "Drag a rectangle to show frame ratio and pixel W,H",
    grade_distribution: "A/B/C/D Grade Distribution", grade_distribution_sub: "Live perception grade count and ratio",
    violin_distribution: "Metric Distribution", violin_distribution_sub: "Mean, std and percentile spread from current samples",
    grade_tip: "Video/Webcam: 40% Conf + 30% clarity + 20% tracking stability + 10% bbox size. Images: 45% Conf + 40% clarity + 15% bbox size. A>=75, B>=60, C>=45, D<45.",
    classic_samples_purpose: "Classic Samples are representative quality cases: low contour, small target, and recommended quality. They are not per-ID stability samples; use Product Spec / tracking indicators for track stability.",
    no_bbox: "No bbox sample is available yet", downloaded: "Image downloaded", analysis_started: "Analysis started", stop_requested: "Stop requested", loaded: "Loaded",
    quality_labels: ["Clarity", "Exposure", "Contrast", "Lux", "Stability", "Edges"],
    model_labels: ["Conf mean", "Conf cover", "BBox contour", "Tracking", "BBox size", "Low blur", "Edge cover", "A/B cover"],
    quality_tips: [
      "🔎 Overall frame sharpness. Higher is better.",
      "☀️ Exposure close to a usable mid range. Higher is better.",
      "◐ Texture and luminance separation. Higher is better.",
      "💡 Relative lux proxy, not calibrated lux. Higher is usually better.",
      "🎥 Less motion blur and steadier frames. Higher is better.",
      "〰️ Structural edge density. Higher means more contour information.",
    ],
    model_tips: [
      "🎯 Average detection confidence. Higher is better.",
      "📈 Share of samples with conf >= 0.50. Higher means wider usable range.",
      "👤 Person contour clarity inside bbox. Higher is better.",
      "🧭 Tracking stability from ID age, continuity, and center-path smoothness. Higher is better.",
      "📐 BBox size coverage. Tiny targets reduce perception. Higher is better.",
      "🧊 Share of low-blur bbox samples. Higher is better.",
      "〰️ Useful edge coverage inside bbox. Higher is better.",
      "✅ Share of A/B grade samples. Higher is closer to delivery standard.",
    ],
  },
  ja: {
    app_title: "YOLO 推論評価", app_subtitle: "画質、bbox サイズ、輪郭明瞭度、モデル信頼度を定量化します。",
    run_setup: "実行設定", model_source: "モデル / ソース", tflite_model: "TFLite モデル", load: "読込",
    image_source: "画像ソース", video_file: "動画ファイル", webcam: "Webcam", image_files: "画像ファイル",
    video: "動画", webcam_index: "Webcam index", image_manager: "画像ファイル管理", images_path: "パス: images/",
    download_url: "公開画像 URL", download_image: "images/ へ保存", image_tracking_note: "画像は離散サンプルのため tracking 品質は評価しません。",
    conf_threshold: "Conf 閾値", frame_stride: "Frame stride", start_analysis: "分析開始", stop: "停止",
    report_name: "レポート名", tags: "タグ", export_report: "レポート出力",
    object_samples: "物体サンプル", avg_conf: "平均 Conf", avg_bbox_px: "平均 BBox %", sharpness: "明瞭度", recommended: "推奨比率",
    live_view: "ライブ認識ビュー", matrix_title: "BBox Frame 比率 / Confidence 行列", matrix_subtitle: "行: conf · 列: bbox 面積の frame 全体比率",
    quality_radar: "画質レーダー", quality_radar_sub: "フレーム全体の品質", model_radar: "モデル認識レーダー", model_radar_sub: "検出信頼度と bbox 品質",
    bbox_metrics: "現在の BBox 指標", person_only: "person detections のみ", track: "Track", conf: "Conf", approx_size: "Frame %",
    actual_size: "実 W x H", sharp: "Sharp", edge: "Edge", brightness: "Brightness", contrast: "Contrast", grade: "評価",
    classic_samples: "代表サンプル", classic_samples_sub: "各段階の代表 bbox と条件",
    bin_inspector: "Bin サンプル確認", bin_inspector_sub: "行列 cell をクリックし、frame 上で矩形をドラッグして bbox 比率を確認します",
    reset_canvas: "キャンバス消去", selected_bin: "選択 bin", drag_hint: "矩形をドラッグすると frame 比率と pixel W,H を表示します",
    grade_distribution: "A/B/C/D 評価分布", grade_distribution_sub: "リアルタイム認識評価の数と比率",
    violin_distribution: "指標分布", violin_distribution_sub: "現在サンプルの mean、std、百分位分布",
    grade_tip: "Video/Webcam: 40% Conf + 30% 明瞭度 + 20% tracking 安定性 + 10% bbox サイズ。Images: 45% Conf + 40% 明瞭度 + 15% bbox サイズ。A>=75, B>=60, C>=45, D<45。",
    classic_samples_purpose: "Classic Samples は低輪郭、小対象、推奨品質の代表例です。特定 ID の安定性抽出ではありません。tracking 安定性は Product Spec / tracking 指標を確認してください。",
    no_bbox: "bbox サンプルがありません", downloaded: "画像を保存しました", analysis_started: "分析を開始しました", stop_requested: "停止要求済み", loaded: "読込完了",
    quality_labels: ["明瞭", "露出", "対比", "Lux", "安定", "輪郭"],
    model_labels: ["Conf平均", "Conf範囲", "BBox輪郭", "Tracking", "BBoxサイズ", "低ぼけ", "輪郭範囲", "A/B範囲"],
    quality_tips: [
      "🔎 フレーム全体の鮮明度。高いほど良い。",
      "☀️ 適切な中間露出に近い度合い。高いほど良い。",
      "◐ 明暗差と質感の分離。高いほど良い。",
      "💡 相対 lux proxy。校正済み lux ではありません。通常は高いほど良い。",
      "🎥 動きぼけが少ない安定度。高いほど良い。",
      "〰️ エッジ構造密度。高いほど輪郭情報が多い。",
    ],
    model_tips: [
      "🎯 検出 confidence 平均。高いほど良い。",
      "📈 conf >= 0.50 のサンプル割合。高いほど利用可能範囲が広い。",
      "👤 bbox 内人物輪郭の明瞭度。高いほど良い。",
      "🧭 ID age、連続度、中心点軌跡の滑らかさによる tracking 安定性。高いほど良い。",
      "📐 bbox サイズ範囲。小さすぎる対象は認識を下げます。高いほど良い。",
      "🧊 低ぼけ bbox の割合。高いほど良い。",
      "〰️ bbox 内の有効エッジ範囲。高いほど良い。",
      "✅ A/B 評価サンプル割合。高いほど納品基準に近い。",
    ],
  },
  ko: {
    app_title: "YOLO 추론 평가", app_subtitle: "영상 품질, bbox 크기, 윤곽 선명도, 모델 신뢰도를 정량화합니다.",
    run_setup: "실행 설정", model_source: "모델 / 소스", tflite_model: "TFLite 모델", load: "로드",
    image_source: "이미지 소스", video_file: "비디오 파일", webcam: "Webcam", image_files: "이미지 파일",
    video: "비디오", webcam_index: "Webcam index", image_manager: "이미지 파일 관리자", images_path: "경로: images/",
    download_url: "공개 이미지 URL", download_image: "images/에 다운로드", image_tracking_note: "이미지는 개별 샘플이므로 tracking 품질은 평가하지 않습니다.",
    conf_threshold: "Conf 기준", frame_stride: "Frame stride", start_analysis: "분석 시작", stop: "정지",
    report_name: "보고서 이름", tags: "태그", export_report: "보고서 출력",
    object_samples: "객체 샘플", avg_conf: "평균 Conf", avg_bbox_px: "평균 BBox %", sharpness: "선명도", recommended: "권장 비율",
    live_view: "실시간 인식 화면", matrix_title: "BBox Frame 비율 / Confidence 매트릭스", matrix_subtitle: "행: conf · 열: bbox 면적의 전체 frame 대비 비율",
    quality_radar: "이미지 품질 레이더", quality_radar_sub: "전체 프레임 품질", model_radar: "모델 인식 레이더", model_radar_sub: "검출 신뢰도와 bbox 품질",
    bbox_metrics: "현재 BBox 지표", person_only: "person detections만", track: "Track", conf: "Conf", approx_size: "Frame %",
    actual_size: "실제 W x H", sharp: "Sharp", edge: "Edge", brightness: "Brightness", contrast: "Contrast", grade: "등급",
    classic_samples: "대표 샘플", classic_samples_sub: "단계별 대표 bbox 및 조건",
    bin_inspector: "Bin 샘플 검사", bin_inspector_sub: "매트릭스 cell을 클릭한 뒤 frame 위에서 사각형을 드래그해 bbox 비율을 확인합니다",
    reset_canvas: "캔버스 초기화", selected_bin: "선택 bin", drag_hint: "사각형을 드래그하면 frame 비율과 pixel W,H를 표시합니다",
    grade_distribution: "A/B/C/D 등급 분포", grade_distribution_sub: "실시간 인식 등급 수와 비율",
    violin_distribution: "지표 분포", violin_distribution_sub: "현재 샘플의 mean, std 및 분위 분포",
    grade_tip: "Video/Webcam: 40% Conf + 30% 선명도 + 20% tracking 안정성 + 10% bbox 크기. Images: 45% Conf + 40% 선명도 + 15% bbox 크기. A>=75, B>=60, C>=45, D<45.",
    classic_samples_purpose: "Classic Samples는 낮은 윤곽, 작은 대상, 권장 품질의 대표 사례입니다. 특정 ID 안정성 샘플이 아니며 tracking 안정성은 Product Spec / tracking 지표를 보세요.",
    no_bbox: "bbox 샘플이 없습니다", downloaded: "이미지 다운로드 완료", analysis_started: "분석 시작됨", stop_requested: "정지 요청됨", loaded: "로드됨",
    quality_labels: ["선명", "노출", "대비", "Lux", "안정", "엣지"],
    model_labels: ["Conf평균", "Conf범위", "BBox윤곽", "Tracking", "BBox크기", "저블러", "엣지범위", "A/B범위"],
    quality_tips: [
      "🔎 전체 프레임 선명도. 높을수록 좋습니다.",
      "☀️ 적정 중간 노출에 가까운 정도. 높을수록 좋습니다.",
      "◐ 명암과 질감 분리도. 높을수록 좋습니다.",
      "💡 상대 lux proxy이며 보정 lux가 아닙니다. 보통 높을수록 좋습니다.",
      "🎥 동적 블러가 적은 안정도. 높을수록 좋습니다.",
      "〰️ 엣지 구조 밀도. 높을수록 윤곽 정보가 많습니다.",
    ],
    model_tips: [
      "🎯 검출 confidence 평균. 높을수록 좋습니다.",
      "📈 conf >= 0.50 샘플 비율. 높을수록 사용 가능 범위가 넓습니다.",
      "👤 bbox 내부 사람 윤곽 선명도. 높을수록 좋습니다.",
      "🧭 ID age, 연속도, 중심점 궤적 smoothness 기반 tracking 안정성. 높을수록 좋습니다.",
      "📐 bbox 크기 범위. 너무 작은 대상은 인식을 낮춥니다. 높을수록 좋습니다.",
      "🧊 저블러 bbox 샘플 비율. 높을수록 좋습니다.",
      "〰️ bbox 내부 유효 엣지 범위. 높을수록 좋습니다.",
      "✅ A/B 등급 샘플 비율. 높을수록 납품 기준에 가깝습니다.",
    ],
  },
};

const GRADE_EXPLAINS = {
  A: {
    zh: "A: 可交付區間。高 confidence、bbox 清晰，且 video/webcam 中 tracking 穩定。",
    en: "A: Delivery-ready range. High confidence, clear bbox, and stable tracking for video/webcam.",
  },
  B: {
    zh: "B: 可用區間。大多數情況可用，但仍需注意小 bbox、模糊或 tracking 波動。",
    en: "B: Usable range. Generally acceptable, with some risk from small bbox, blur, or tracking drift.",
  },
  C: {
    zh: "C: 邊界區間。模型可偵測，但品質或追蹤穩定度不足以作為強規格。",
    en: "C: Borderline range. Detected, but quality or tracking stability is weak for a strong spec.",
  },
  D: {
    zh: "D: 不建議區間。低 confidence、低清晰度、小 bbox 或 tracking 不穩定。",
    en: "D: Not recommended. Low confidence, poor clarity, tiny bbox, or unstable tracking.",
  },
};

const METRIC_TIPS = {
  conf: { zh: "Confidence：模型對 person 類別的信心。越高越好。", en: "Confidence: model confidence for person detections. Higher is better." },
  bbox_area_pct: { zh: "BBox frame %：bbox 面積佔整張 frame 的比例。太小通常會降低感知。", en: "BBox frame %: bbox area relative to full frame. Tiny targets usually reduce perception." },
  sharpness_score: { zh: "BBox sharpness：主要評分依據之一，綜合多種邊緣與紋理清晰度。越高越好。", en: "BBox sharpness: one primary grade driver from edge and texture clarity. Higher is better." },
  contour_clarity_score: { zh: "Contour clarity：bbox 內物件輪廓可分辨度，輔助判斷邊界品質。越高越好。", en: "Contour clarity: object boundary separability inside bbox. Higher is better." },
  track_stability_score: { zh: "Track stability：ID 存活、連續性與中心點軌跡平滑度。只適用 video/webcam。", en: "Track stability: ID age, continuity, and center-path smoothness. Video/webcam only." },
  track_smoothness_score: { zh: "Track smoothness：中心點速度/加速度是否平滑；突然折線或跳點會降低。", en: "Track smoothness: center velocity/acceleration smoothness; sudden bends or jumps reduce it." },
  track_continuity_score: { zh: "Track continuity：同一 ID 在 frame 之間是否連續。間隔變大會降低。", en: "Track continuity: whether the same ID remains continuous across frames. Large gaps reduce it." },
  frame_lux_proxy: { zh: "Lux proxy：亮度與對比推估的相對照度，非校正實體 lux。", en: "Lux proxy: relative brightness/contrast light estimate, not calibrated physical lux." },
};

function t(key) {
  return (I18N[currentLang] && I18N[currentLang][key]) || I18N.en[key] || key;
}

function localizedTip(map, fallback = "") {
  return (map && (map[currentLang] || map.en)) || fallback;
}

function applyLanguage() {
  document.documentElement.lang = currentLang === "zh" ? "zh-Hant" : currentLang;
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    node.textContent = t(node.dataset.i18n);
  });
  renderRadarLegends();
  if (latestStatus) {
    updateRadarPanels(latestStatus);
    drawMatrixChart(latestStatus.area_conf_matrix || []);
    drawGradeChart(latestStatus.grade_distribution || []);
    drawViolinChart(latestStatus.metric_distribution || []);
  }
}

function renderRadarLegends() {
  renderLegend("qualityRadarLegend", t("quality_labels"), t("quality_tips"));
  renderLegend("modelRadarLegend", t("model_labels"), t("model_tips"));
}

function renderLegend(elementId, labels, tips) {
  const container = document.getElementById(elementId);
  if (!container) return;
  container.innerHTML = "";
  labels.forEach((label, index) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "radar-chip";
    chip.setAttribute("data-tooltip", tips[index] || "");
    chip.setAttribute("aria-label", `${label}: ${tips[index] || ""}`);
    chip.textContent = label;
    container.appendChild(chip);
  });
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("show");
  window.setTimeout(() => toast.classList.remove("show"), 3600);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok || data.ok === false) {
    throw new Error(data.error || "Request failed");
  }
  return data;
}

function fillSelect(select, items) {
  select.innerHTML = "";
  for (const item of items) {
    const option = document.createElement("option");
    option.value = item;
    option.textContent = item;
    select.appendChild(option);
  }
}

async function loadAssets() {
  const assets = await api("/api/assets");
  latestAssets = assets;
  fillSelect(modelSelect, assets.models);
  fillSelect(videoSelect, assets.videos);
  renderImageList(assets.images || []);
}

function renderImageList(images) {
  imageList.innerHTML = "";
  if (!images.length) {
    const empty = document.createElement("p");
    empty.className = "hint";
    empty.textContent = "images/";
    imageList.appendChild(empty);
    return;
  }
  for (const name of images) {
    const label = document.createElement("label");
    label.className = "check-row";
    label.innerHTML = `<input type="checkbox" value="${name}" checked><span>${name}</span>`;
    imageList.appendChild(label);
  }
}

function selectedSource() {
  const sourceType = document.querySelector("input[name='sourceType']:checked").value;
  if (sourceType === "webcam") {
    return { type: "webcam", index: Number(document.getElementById("webcamIndex").value || 0) };
  }
  if (sourceType === "images") {
    const images = Array.from(imageList.querySelectorAll("input:checked")).map((input) => input.value);
    return { type: "images", images };
  }
  return { type: "video", name: videoSelect.value };
}

function updateSourcePanels() {
  const sourceType = document.querySelector("input[name='sourceType']:checked").value;
  document.getElementById("videoPanel").classList.toggle("hidden", sourceType !== "video");
  document.getElementById("webcamPanel").classList.toggle("hidden", sourceType !== "webcam");
  document.getElementById("imagesPanel").classList.toggle("hidden", sourceType !== "images");
}

function setRunState(status, error = false) {
  runState.textContent = status;
  runState.classList.toggle("running", status === "Running");
  runState.classList.toggle("error", error);
}

function formatNumber(value, digits = 0) {
  const number = Number(value || 0);
  return number.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

function resizeCanvasToContainer(canvas, height, minWidth = 320) {
  const parent = canvas.parentElement;
  if (!parent) return;
  const measuredWidth = Math.floor(parent.clientWidth || 0);
  const width = measuredWidth > 0 ? measuredWidth : minWidth;
  const targetHeight = Math.max(260, Math.floor(height));
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== targetHeight) canvas.height = targetHeight;
  canvas.style.width = "100%";
  canvas.style.height = `${targetHeight}px`;
}

function resizeAllCanvases() {
  resizeCanvasToContainer(chart, CHART_HEIGHTS.matrix, 360);
  resizeCanvasToContainer(qualityRadarCanvas, CHART_HEIGHTS.radar, 300);
  resizeCanvasToContainer(modelRadarCanvas, CHART_HEIGHTS.radar, 300);
  resizeCanvasToContainer(gradeChart, CHART_HEIGHTS.grade, 320);
  resizeCanvasToContainer(violinChart, CHART_HEIGHTS.violin, 420);
}

function redrawChartsFromState() {
  const data = latestStatus || {};
  drawMatrixChart(data.area_conf_matrix || []);
  updateRadarPanels(data);
  drawGradeChart(data.grade_distribution || []);
  drawViolinChart(data.metric_distribution || []);
  syncBinCanvas();
}

function requestChartResize() {
  if (resizeFrame) return;
  resizeFrame = window.requestAnimationFrame(() => {
    resizeFrame = null;
    resizeAllCanvases();
    redrawChartsFromState();
  });
}

function setupResizableRows() {
  document.querySelectorAll(".resizable-row").forEach((row) => {
    if (row.querySelector(".split-resizer")) return;
    const panels = Array.from(row.children).filter((child) => !child.classList.contains("split-resizer"));
    if (panels.length < 2) return;
    const splitId = row.dataset.splitId || "default";
    const saved = Number(localStorage.getItem(`assessment_split_${splitId}`) || 50);
    const applySplit = (pct) => {
      const left = Math.max(24, Math.min(76, Number(pct || 50)));
      row.style.gridTemplateColumns = `minmax(280px, ${left}fr) 14px minmax(280px, ${100 - left}fr)`;
      localStorage.setItem(`assessment_split_${splitId}`, String(left));
    };
    const handle = document.createElement("button");
    handle.type = "button";
    handle.className = "split-resizer";
    handle.setAttribute("aria-label", "Resize columns");
    handle.setAttribute("title", "Drag to resize columns");
    panels[0].after(handle);
    applySplit(saved);

    const updateFromPointer = (event) => {
      const rect = row.getBoundingClientRect();
      const usable = Math.max(1, rect.width - 14);
      const minPx = Math.min(300, usable * 0.24);
      const leftPx = Math.max(minPx, Math.min(usable - minPx, event.clientX - rect.left));
      applySplit((leftPx / usable) * 100);
      requestChartResize();
    };

    handle.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      handle.setPointerCapture(event.pointerId);
      handle.classList.add("active");
      document.body.classList.add("resizing");
      updateFromPointer(event);
    });
    handle.addEventListener("pointermove", (event) => {
      if (!handle.classList.contains("active")) return;
      updateFromPointer(event);
    });
    const stopResize = (event) => {
      if (!handle.classList.contains("active")) return;
      handle.classList.remove("active");
      document.body.classList.remove("resizing");
      if (handle.hasPointerCapture(event.pointerId)) handle.releasePointerCapture(event.pointerId);
      requestChartResize();
    };
    handle.addEventListener("pointerup", stopResize);
    handle.addEventListener("pointercancel", stopResize);
  });
}

function drawMatrixChart(matrix) {
  const w = chart.width;
  const h = chart.height;
  matrixCells = [];
  ctx.clearRect(0, 0, w, h);
  const grad = ctx.createLinearGradient(0, 0, w, h);
  grad.addColorStop(0, "#07111f");
  grad.addColorStop(0.55, "#0f1f35");
  grad.addColorStop(1, "#16213a");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);

  const counts = new Map();
  let maxCount = 1;
  for (const item of matrix) {
    const key = `${item.area_bin}|${item.conf_bin}`;
    counts.set(key, Number(item.count || 0));
    maxCount = Math.max(maxCount, Number(item.count || 0));
  }

  const left = 118;
  const top = 44;
  const cellW = Math.floor((w - left - 28) / sizeOrder.length);
  const cellH = 56;
  ctx.font = "12px Arial";
  ctx.fillStyle = "#dbeafe";
  ctx.fillText(t("conf"), 24, 26);
  ctx.fillText(t("matrix_subtitle"), left + 95, 26);

  for (let row = 0; row < confOrder.length; row++) {
    const y = top + row * cellH;
    ctx.fillStyle = "#9fb4d8";
    ctx.fillText(confOrder[confOrder.length - 1 - row], 18, y + 34);
    for (let col = 0; col < sizeOrder.length; col++) {
      const x = left + col * cellW;
      const sizeLabel = sizeOrder[col];
      const confLabel = confOrder[confOrder.length - 1 - row];
      const count = counts.get(`${sizeLabel}|${confLabel}`) || 0;
      const intensity = count / maxCount;
      const cell = {
        x,
        y,
        width: cellW - 12,
        height: cellH - 10,
        area_bin: sizeLabel,
        conf_bin: confLabel,
        count,
      };
      matrixCells.push(cell);

      ctx.fillStyle = count ? `rgba(59, 130, 246, ${0.16 + intensity * 0.42})` : "rgba(148, 163, 184, 0.055)";
      ctx.strokeStyle = count ? "rgba(147, 197, 253, 0.38)" : "rgba(148, 163, 184, 0.12)";
      ctx.lineWidth = 1;
      roundRect(ctx, x, y, cellW - 12, cellH - 10, 8);
      ctx.fill();
      ctx.stroke();

      if (count > 0) {
        const radius = 7 + intensity * 17;
        const cx = x + cellW / 2 - 6;
        const cy = y + cellH / 2 - 5;
        const glow = ctx.createRadialGradient(cx, cy, 2, cx, cy, radius + 18);
        glow.addColorStop(0, "rgba(217,119,6,0.96)");
        glow.addColorStop(0.45, "rgba(59,130,246,0.72)");
        glow.addColorStop(1, "rgba(59,130,246,0)");
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(cx, cy, radius + 18, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#f8fafc";
        ctx.font = "700 13px Arial";
        ctx.textAlign = "center";
        ctx.fillText(String(count), cx, cy + 4);
        ctx.textAlign = "left";
      }
    }
  }

  ctx.fillStyle = "#cbd5e1";
  ctx.font = "12px Arial";
  for (let col = 0; col < sizeOrder.length; col++) {
    const x = left + col * cellW + 6;
    ctx.fillStyle = "#f8fafc";
    ctx.font = "700 12px Arial";
    ctx.fillText(sizeClassOrder[col], x, top + confOrder.length * cellH + 24);
    ctx.fillStyle = "#9fb0cc";
    ctx.font = "11px Arial";
    ctx.fillText(sizeOrder[col], x, top + confOrder.length * cellH + 40);
  }

  if (matrix.length === 0) {
    ctx.fillStyle = "rgba(226, 232, 240, 0.86)";
    ctx.font = "700 18px Arial";
    ctx.textAlign = "center";
    ctx.fillText(t("no_bbox"), w / 2, h / 2 - 8);
    ctx.font = "13px Arial";
    ctx.fillText(t("start_analysis"), w / 2, h / 2 + 18);
    ctx.textAlign = "left";
  }
}

function matrixCellFromEvent(event) {
  const rect = chart.getBoundingClientRect();
  const scaleX = chart.width / rect.width;
  const scaleY = chart.height / rect.height;
  const x = (event.clientX - rect.left) * scaleX;
  const y = (event.clientY - rect.top) * scaleY;
  return matrixCells.find((cell) => (
    x >= cell.x && x <= cell.x + cell.width && y >= cell.y && y <= cell.y + cell.height
  ));
}

function sizeClassFromBin(bin) {
  const index = sizeOrder.indexOf(bin);
  return index >= 0 ? sizeClassOrder[index] : "";
}

async function loadBinSample(cell) {
  if (!cell || cell.count <= 0) {
    showToast("No sample for selected bin");
    return;
  }
  try {
    const data = await api("/api/bin_sample", {
      method: "POST",
      body: JSON.stringify({ area_bin: cell.area_bin, conf_bin: cell.conf_bin }),
    });
    currentBinSample = data.sample;
    drawingRect = null;
    binInspector.classList.remove("hidden");
    binSampleTitle.textContent = `${t("selected_bin")}: ${sizeClassFromBin(cell.area_bin)} ${cell.area_bin} / ${cell.conf_bin}`;
    binSampleMeta.textContent =
      `frame ${currentBinSample.frame_index} · track ${currentBinSample.track_id ?? "-"} · bbox ${formatNumber(currentBinSample.bbox_area_pct, 2)}% · ${formatNumber(currentBinSample.bbox_w, 0)} x ${formatNumber(currentBinSample.bbox_h, 0)} · track stability ${currentBinSample.track_stability_score == null ? "-" : formatNumber(currentBinSample.track_stability_score, 1)}`;
    binDrawStats.textContent = t("drag_hint");
    binSampleImage.onload = syncBinCanvas;
    binSampleImage.src = currentBinSample.image || "";
  } catch (error) {
    showToast(error.message);
  }
}

function syncBinCanvas() {
  if (!binSampleImage.complete || !binSampleImage.naturalWidth) return;
  const rect = binSampleImage.getBoundingClientRect();
  const parentRect = binSampleImage.parentElement.getBoundingClientRect();
  binDrawCanvas.style.width = `${rect.width}px`;
  binDrawCanvas.style.height = `${rect.height}px`;
  binDrawCanvas.style.left = `${rect.left - parentRect.left}px`;
  binDrawCanvas.style.top = `${rect.top - parentRect.top}px`;
  binDrawCanvas.width = Math.max(1, Math.round(rect.width));
  binDrawCanvas.height = Math.max(1, Math.round(rect.height));
  redrawBinCanvas();
}

function drawKnownSampleBbox() {
  if (!currentBinSample || !currentBinSample.bbox) return;
  const scaleX = binDrawCanvas.width / Number(currentBinSample.frame_w || 1);
  const scaleY = binDrawCanvas.height / Number(currentBinSample.frame_h || 1);
  const [x1, y1, x2, y2] = currentBinSample.bbox.map(Number);
  const bx = x1 * scaleX;
  const by = y1 * scaleY;
  const bw = (x2 - x1) * scaleX;
  const bh = (y2 - y1) * scaleY;
  binDrawCtx.fillStyle = "rgba(220, 38, 38, 0.50)";
  binDrawCtx.strokeStyle = "rgba(248, 113, 113, 1)";
  binDrawCtx.lineWidth = 3;
  binDrawCtx.fillRect(bx, by, bw, bh);
  binDrawCtx.strokeRect(bx, by, bw, bh);
}

function drawTrackTrail() {
  const trail = currentBinSample && currentBinSample.track_trail;
  if (!trail || trail.length < 2) return;
  const frameW = Number(currentBinSample.frame_w || 1);
  const frameH = Number(currentBinSample.frame_h || 1);
  const diag = Math.hypot(frameW, frameH);
  const scaleX = binDrawCanvas.width / frameW;
  const scaleY = binDrawCanvas.height / frameH;
  binDrawCtx.strokeStyle = "rgba(34, 211, 238, 0.95)";
  binDrawCtx.fillStyle = "rgba(34, 211, 238, 0.95)";
  binDrawCtx.lineWidth = 3;
  binDrawCtx.beginPath();
  trail.forEach((p, index) => {
    const x = Number(p.cx || 0) * diag * scaleX;
    const y = Number(p.cy || 0) * diag * scaleY;
    if (index === 0) binDrawCtx.moveTo(x, y);
    else binDrawCtx.lineTo(x, y);
  });
  binDrawCtx.stroke();
  trail.forEach((p, index) => {
    const x = Number(p.cx || 0) * diag * scaleX;
    const y = Number(p.cy || 0) * diag * scaleY;
    binDrawCtx.beginPath();
    binDrawCtx.arc(x, y, index === trail.length - 1 ? 5 : 3, 0, Math.PI * 2);
    binDrawCtx.fill();
  });
}

function redrawBinCanvas() {
  binDrawCtx.clearRect(0, 0, binDrawCanvas.width, binDrawCanvas.height);
  drawTrackTrail();
  drawKnownSampleBbox();
  if (!drawingRect) return;
  binDrawCtx.fillStyle = "rgba(37, 99, 235, 0.18)";
  binDrawCtx.strokeStyle = "rgba(219, 234, 254, 0.95)";
  binDrawCtx.lineWidth = 2;
  binDrawCtx.fillRect(drawingRect.x, drawingRect.y, drawingRect.w, drawingRect.h);
  binDrawCtx.strokeRect(drawingRect.x, drawingRect.y, drawingRect.w, drawingRect.h);
}

function canvasPoint(event) {
  const rect = binDrawCanvas.getBoundingClientRect();
  const clientX = event.touches ? event.touches[0].clientX : event.clientX;
  const clientY = event.touches ? event.touches[0].clientY : event.clientY;
  return {
    x: Math.max(0, Math.min(binDrawCanvas.width, clientX - rect.left)),
    y: Math.max(0, Math.min(binDrawCanvas.height, clientY - rect.top)),
  };
}

function updateDrawStats(rect) {
  if (!currentBinSample || !rect) {
    binDrawStats.textContent = t("drag_hint");
    return;
  }
  const scaleX = Number(currentBinSample.frame_w || 1) / Math.max(1, binDrawCanvas.width);
  const scaleY = Number(currentBinSample.frame_h || 1) / Math.max(1, binDrawCanvas.height);
  const pxW = Math.abs(rect.w) * scaleX;
  const pxH = Math.abs(rect.h) * scaleY;
  const pct = (pxW * pxH) / Math.max(1, Number(currentBinSample.frame_w || 1) * Number(currentBinSample.frame_h || 1)) * 100;
  binDrawStats.textContent = `Drawn bbox: ${formatNumber(pct, 3)}% frame · ${formatNumber(pxW, 0)} x ${formatNumber(pxH, 0)} px`;
}

function startBinDraw(event) {
  if (!currentBinSample) return;
  event.preventDefault();
  const p = canvasPoint(event);
  isDrawingBinRect = true;
  drawingRect = { x: p.x, y: p.y, w: 0, h: 0, startX: p.x, startY: p.y };
  redrawBinCanvas();
}

function moveBinDraw(event) {
  if (!isDrawingBinRect || !drawingRect) return;
  event.preventDefault();
  const p = canvasPoint(event);
  drawingRect.x = Math.min(drawingRect.startX, p.x);
  drawingRect.y = Math.min(drawingRect.startY, p.y);
  drawingRect.w = Math.abs(p.x - drawingRect.startX);
  drawingRect.h = Math.abs(p.y - drawingRect.startY);
  redrawBinCanvas();
  updateDrawStats(drawingRect);
}

function endBinDraw() {
  if (!isDrawingBinRect) return;
  isDrawingBinRect = false;
  updateDrawStats(drawingRect);
}

function resetBinCanvas() {
  drawingRect = null;
  redrawBinCanvas();
  updateDrawStats(null);
}

function roundRect(context, x, y, width, height, radius) {
  context.beginPath();
  context.moveTo(x + radius, y);
  context.lineTo(x + width - radius, y);
  context.quadraticCurveTo(x + width, y, x + width, y + radius);
  context.lineTo(x + width, y + height - radius);
  context.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  context.lineTo(x + radius, y + height);
  context.quadraticCurveTo(x, y + height, x, y + height - radius);
  context.lineTo(x, y + radius);
  context.quadraticCurveTo(x, y, x + radius, y);
}

function updateTable(objects) {
  const body = document.getElementById("bboxRows");
  body.innerHTML = "";
  for (const obj of objects) {
    const grade = String(obj.perception_grade || "D").toLowerCase();
    const row = document.createElement("tr");
    row.tabIndex = 0;
    row.className = "clickable-row";
    row.innerHTML = `
      <td>${obj.track_id ?? ""}</td>
      <td>${formatNumber(obj.conf, 2)}</td>
      <td>${sizeClassFromBin(obj.size_bin || obj.area_bin)} ${obj.size_bin || obj.area_bin || ""} (${formatNumber(obj.bbox_area_pct, 2)}%)</td>
      <td>${formatNumber(obj.bbox_w, 0)} x ${formatNumber(obj.bbox_h, 0)}</td>
      <td>${formatNumber(obj.sharpness_score, 1)}</td>
      <td>${obj.track_stability_score == null ? "-" : formatNumber(obj.track_stability_score, 1)}</td>
      <td>${formatNumber(obj.edge_density, 4)}</td>
      <td>${formatNumber(obj.brightness, 1)}</td>
      <td>${formatNumber(obj.contrast, 1)}</td>
      <td><span class="grade grade-${grade}" tabindex="0" data-tooltip="${t("grade_tip")}">${obj.perception_grade || "D"}</span></td>
    `;
    row.addEventListener("click", () => openAnalysis(obj));
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") openAnalysis(obj);
    });
    body.appendChild(row);
  }
}

function updateSamples(examples) {
  const grid = document.getElementById("sampleGrid");
  grid.innerHTML = "";
  for (const item of examples) {
    const card = document.createElement("article");
    card.className = "sample";
    card.innerHTML = `
      <img src="${item.context || item.image || ""}" alt="${item.stage}">
      <div class="body">
        <h3>${item.stage}</h3>
        <p>${item.condition}</p>
        <p>frame ${item.frame_index} · conf ${formatNumber(item.conf, 2)} · sharp ${formatNumber(item.sharpness_score, 1)}</p>
      </div>
    `;
    card.addEventListener("click", () => openAnalysis(item));
    grid.appendChild(card);
  }
}

function metricValue(value, digits = 1) {
  return value == null ? "-" : formatNumber(value, digits);
}

function metricZeroNote(item) {
  const zeroLike = [
    item.sharpness_score,
    item.contour_clarity_score,
    item.laplacian_var,
    item.edge_density,
    item.brightness,
    item.contrast,
  ].filter((value) => Number(value || 0) === 0).length;
  if (zeroLike >= 4) {
    return currentLang === "zh"
      ? "多數 bbox 影像處理指標為 0，通常代表該 sample 沒有完整 crop/debug metrics、bbox crop 為空、或此圖是舊報告/舊快取樣本。請重新跑分析取得完整指標。"
      : "Most bbox image-processing metrics are 0. This usually means the sample lacks full crop/debug metrics, the bbox crop was empty, or it came from an older cached sample. Re-run analysis for complete metrics.";
  }
  return currentLang === "zh"
    ? "評分主要看第一層 Grade Drivers；第二層影像處理指標用來解釋清晰度來源。某單一指標為 0 不一定代表失敗，可能只是邊緣/紋理極少或該來源不適用。"
    : "Use Grade Drivers as the primary scoring layer. Image-processing metrics explain where clarity comes from. A single zero does not always mean failure; it can mean little texture/edge information or a non-applicable source.";
}

function drawRadar(context, canvas, values, labels, options = {}) {
  const w = canvas.width;
  const h = canvas.height;
  context.clearRect(0, 0, w, h);
  const dark = options.dark !== false;
  context.fillStyle = dark ? "#0b1220" : "#ffffff";
  context.fillRect(0, 0, w, h);
  const cx = w / 2;
  const cy = h / 2 + 8;
  const radius = Math.min(w, h) * 0.34;
  const count = labels.length;

  context.strokeStyle = dark ? "rgba(148,163,184,0.25)" : "#dbeafe";
  context.fillStyle = dark ? "#94a3b8" : "#475569";
  context.font = "12px Arial";
  for (let ring = 1; ring <= 4; ring++) {
    context.beginPath();
    for (let i = 0; i < count; i++) {
      const angle = -Math.PI / 2 + (Math.PI * 2 * i) / count;
      const r = (radius * ring) / 4;
      const x = cx + Math.cos(angle) * r;
      const y = cy + Math.sin(angle) * r;
      if (i === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    }
    context.closePath();
    context.stroke();
  }

  for (let i = 0; i < count; i++) {
    const angle = -Math.PI / 2 + (Math.PI * 2 * i) / count;
    context.beginPath();
    context.moveTo(cx, cy);
    context.lineTo(cx + Math.cos(angle) * radius, cy + Math.sin(angle) * radius);
    context.stroke();
    const lx = cx + Math.cos(angle) * (radius + 34);
    const ly = cy + Math.sin(angle) * (radius + 24);
    context.textAlign = lx < cx - 10 ? "right" : lx > cx + 10 ? "left" : "center";
    context.fillText(labels[i], lx, ly);
  }

  context.beginPath();
  values.forEach((value, i) => {
    const angle = -Math.PI / 2 + (Math.PI * 2 * i) / count;
    const r = radius * Math.max(0, Math.min(100, Number(value || 0))) / 100;
    const x = cx + Math.cos(angle) * r;
    const y = cy + Math.sin(angle) * r;
    if (i === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.closePath();
  context.fillStyle = "rgba(59,130,246,0.34)";
  context.strokeStyle = "#d97706";
  context.lineWidth = 2;
  context.fill();
  context.stroke();
  context.lineWidth = 1;
  context.textAlign = "left";
}

function radarLabels() {
  return ["Conf", "Contour", "Size", "Frame", "Exposure", "Contrast", "Lux", "Motion"];
}

function radarValuesFromObject(item) {
  const metrics = item.metrics || item;
  return [
    Number(item.conf || 0) * 100,
    Number(item.sharpness_score || 0),
    Math.min(Number(item.bbox_area_pct || 0) / 2, 1) * 100,
    Number(metrics.frame_sharpness || item.frame_sharpness || 0),
    Number(metrics.frame_exposure_score || item.frame_exposure_score || 0),
    Math.min(Number(metrics.frame_contrast || item.frame_contrast || 0) / 70, 1) * 100,
    Number(metrics.frame_lux_proxy || item.frame_lux_proxy || 0),
    Math.max(0, 100 - Number(metrics.frame_motion_blur || item.frame_motion_blur || 0)),
  ];
}

function updateRadarPanels(data) {
  const quality = data.quality_radar || {};
  const model = data.model_radar || {};
  drawRadar(
    qualityRadarCtx,
    qualityRadarCanvas,
    [
      quality.frame_clarity,
      quality.exposure,
      quality.contrast,
      quality.lux_proxy,
      quality.motion_stability,
      quality.edge_structure,
    ],
    t("quality_labels"),
    { dark: true }
  );
  drawRadar(
    modelRadarCtx,
    modelRadarCanvas,
    [
      model.confidence_mean,
      model.confidence_coverage,
      model.bbox_contour,
      model.tracking_stability,
      model.bbox_size,
      model.low_blur_coverage,
      model.edge_coverage,
      model.ab_grade_coverage,
    ],
    t("model_labels"),
    { dark: true }
  );
}

function showChartTooltip(event, html) {
  if (!html) {
    chartTooltip.style.display = "none";
    return;
  }
  chartTooltip.innerHTML = html;
  chartTooltip.style.display = "block";
  chartTooltip.style.left = `${Math.min(window.innerWidth - 340, event.clientX + 14)}px`;
  chartTooltip.style.top = `${Math.min(window.innerHeight - 160, event.clientY + 14)}px`;
}

function hideChartTooltip() {
  chartTooltip.style.display = "none";
}

function canvasHit(canvas, regions, event) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const x = (event.clientX - rect.left) * scaleX;
  const y = (event.clientY - rect.top) * scaleY;
  return regions.find((region) => x >= region.x && x <= region.x + region.w && y >= region.y && y <= region.y + region.h);
}

function drawGradeChart(grades) {
  const w = gradeChart.width;
  const h = gradeChart.height;
  gradeChartCtx.clearRect(0, 0, w, h);
  gradeHoverRegions = [];
  gradeChartCtx.fillStyle = "#0b1220";
  gradeChartCtx.fillRect(0, 0, w, h);
  const rows = grades && grades.length ? grades : ["A", "B", "C", "D"].map((grade) => ({ grade, count: 0, ratio: 0 }));
  const maxCount = Math.max(1, ...rows.map((row) => Number(row.count || 0)));
  const colors = { A: "#16a34a", B: "#2563eb", C: "#d97706", D: "#dc2626" };
  const left = 68;
  const bottom = h - 54;
  const top = 34;
  const slot = (w - left - 40) / rows.length;
  gradeChartCtx.strokeStyle = "rgba(148,163,184,0.24)";
  gradeChartCtx.beginPath();
  gradeChartCtx.moveTo(left, top);
  gradeChartCtx.lineTo(left, bottom);
  gradeChartCtx.lineTo(w - 22, bottom);
  gradeChartCtx.stroke();
  rows.forEach((row, index) => {
    const barH = (Number(row.count || 0) / maxCount) * (bottom - top - 18);
    const x = left + index * slot + slot * 0.24;
    const y = bottom - barH;
    const bw = slot * 0.52;
    gradeHoverRegions.push({
      x: x - 8,
      y: Math.min(y, bottom - 2) - 34,
      w: bw + 16,
      h: Math.max(38, barH + 74),
      html: `<strong>${row.grade}</strong><br>${localizedTip(GRADE_EXPLAINS[row.grade])}<br>${row.count || 0} samples · ${formatNumber((row.ratio || 0) * 100, 1)}%`,
    });
    gradeChartCtx.fillStyle = colors[row.grade] || "#64748b";
    roundRect(gradeChartCtx, x, y, bw, barH || 2, 8);
    gradeChartCtx.fill();
    gradeChartCtx.fillStyle = "#e5eefc";
    gradeChartCtx.font = "700 14px Arial";
    gradeChartCtx.textAlign = "center";
    gradeChartCtx.fillText(`${row.count || 0}`, x + bw / 2, y - 9);
    gradeChartCtx.fillStyle = "#9fb0cc";
    gradeChartCtx.font = "12px Arial";
    gradeChartCtx.fillText(`${formatNumber((row.ratio || 0) * 100, 1)}%`, x + bw / 2, y - 25);
    gradeChartCtx.fillStyle = "#f8fafc";
    gradeChartCtx.font = "800 20px Arial";
    gradeChartCtx.fillText(row.grade, x + bw / 2, bottom + 30);
  });
  if (!rows.some((row) => Number(row.count || 0) > 0)) {
    gradeChartCtx.fillStyle = "rgba(226, 232, 240, 0.86)";
    gradeChartCtx.font = "700 18px Arial";
    gradeChartCtx.fillText(t("no_bbox"), w / 2, h / 2);
  }
  gradeChartCtx.textAlign = "left";
}

function drawViolinChart(distribution) {
  const w = violinChart.width;
  const h = violinChart.height;
  violinChartCtx.clearRect(0, 0, w, h);
  violinHoverRegions = [];
  violinChartCtx.fillStyle = "#0b1220";
  violinChartCtx.fillRect(0, 0, w, h);
  const rows = (distribution || []).filter((row) => ["conf", "bbox_area_pct", "sharpness_score", "contour_clarity_score", "track_stability_score", "frame_lux_proxy"].includes(row.key));
  const left = 48;
  const top = 28;
  const bottom = h - 98;
  const plotH = bottom - top;
  violinChartCtx.strokeStyle = "rgba(148,163,184,0.24)";
  violinChartCtx.beginPath();
  violinChartCtx.moveTo(left, top);
  violinChartCtx.lineTo(left, bottom);
  violinChartCtx.lineTo(w - 20, bottom);
  violinChartCtx.stroke();
  if (!rows.length) {
    violinChartCtx.fillStyle = "rgba(226, 232, 240, 0.86)";
    violinChartCtx.font = "700 18px Arial";
    violinChartCtx.textAlign = "center";
    violinChartCtx.fillText(t("no_bbox"), w / 2, h / 2);
    violinChartCtx.textAlign = "left";
    return;
  }
  const slot = (w - left - 22) / rows.length;
  const scaleValue = (value) => {
    const v = Math.max(0, Math.min(100, Number(value || 0)));
    return bottom - (v / 100) * plotH;
  };
  rows.forEach((row, index) => {
    const scale = row.key === "conf" ? 100 : 1;
    const mean = Number(row.mean || 0) * scale;
    const p10 = Number(row.p10 || 0) * scale;
    const p50 = Number(row.p50 || 0) * scale;
    const p90 = Number(row.p90 || 0) * scale;
    const rawStd = Number(row.std || 0) * scale;
    const std = Math.max(2, Math.min(32, rawStd));
    const cx = left + index * slot + slot / 2;
    const meanY = scaleValue(mean);
    const p10Y = scaleValue(p10);
    const p50Y = scaleValue(p50);
    const p90Y = scaleValue(p90);
    const half = 10 + Math.min(30, std * 0.9);
    violinHoverRegions.push({
      x: cx - slot / 2 + 2,
      y: top,
      w: slot - 4,
      h: h - top,
      html: `<strong>${row.label}</strong><br>${localizedTip(METRIC_TIPS[row.key], row.label)}<br>mean ${formatNumber(mean, 2)} · std ${formatNumber(rawStd, 2)}<br>P10 ${formatNumber(p10, 2)} · P50 ${formatNumber(p50, 2)} · P90 ${formatNumber(p90, 2)}`,
    });
    violinChartCtx.fillStyle = "rgba(59,130,246,0.42)";
    violinChartCtx.strokeStyle = "rgba(147,197,253,0.72)";
    violinChartCtx.beginPath();
    violinChartCtx.moveTo(cx, p10Y);
    violinChartCtx.bezierCurveTo(cx + half, p10Y + 8, cx + half, p90Y - 8, cx, p90Y);
    violinChartCtx.bezierCurveTo(cx - half, p90Y - 8, cx - half, p10Y + 8, cx, p10Y);
    violinChartCtx.closePath();
    violinChartCtx.fill();
    violinChartCtx.stroke();
    violinChartCtx.strokeStyle = "#f59e0b";
    violinChartCtx.lineWidth = 2;
    violinChartCtx.beginPath();
    violinChartCtx.moveTo(cx - half * 0.72, p50Y);
    violinChartCtx.lineTo(cx + half * 0.72, p50Y);
    violinChartCtx.stroke();
    violinChartCtx.fillStyle = "#f8fafc";
    violinChartCtx.beginPath();
    violinChartCtx.arc(cx, meanY, 4, 0, Math.PI * 2);
    violinChartCtx.fill();
    violinChartCtx.fillStyle = "#cbd5e1";
    violinChartCtx.font = "700 12px Arial";
    violinChartCtx.textAlign = "center";
    const label = row.label.replace("BBox frame area %", "BBox %").replace("Contour clarity", "Contour").replace("Track stability", "Track");
    const words = label.split(" ");
    violinChartCtx.fillText(words[0] || label, cx, bottom + 24);
    if (words[1]) violinChartCtx.fillText(words.slice(1).join(" "), cx, bottom + 40);
    violinChartCtx.font = "11px Arial";
    violinChartCtx.fillText(`μ ${formatNumber(mean, 1)}`, cx, bottom + 58);
    violinChartCtx.lineWidth = 1;
  });
  violinChartCtx.textAlign = "left";
}

function updateRecommendations(items) {
  const box = document.getElementById("recommendations");
  box.innerHTML = "";
  for (const item of items || []) {
    const p = document.createElement("p");
    p.textContent = localizeRecommendation(item);
    box.appendChild(p);
  }
}

function openAnalysis(item) {
  const metrics = item.metrics || item;
  document.getElementById("modalTitle").textContent = `Track ${item.track_id ?? "-"} · Grade ${item.perception_grade || "-"}`;
  document.getElementById("modalSubtitle").textContent =
    `${item.size_bin || item.area_bin || "bbox"} · ${formatNumber(item.bbox_area_pct, 2)}% frame · actual ${formatNumber(item.bbox_w, 0)} x ${formatNumber(item.bbox_h, 0)} · conf ${formatNumber(item.conf, 2)}`;
  document.getElementById("modalImage").src = item.context || item.image || "";
  drawRadar(modalRadarCtx, modalRadar, radarValuesFromObject(item), radarLabels(), { dark: true });
  const debug = item.debug_images || {};
  const debugStrip = document.getElementById("debugStrip");
  debugStrip.innerHTML = "";
  for (const [name, src] of Object.entries(debug)) {
    if (!src) continue;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "debug-thumb";
    button.innerHTML = `<img src="${src}" alt="${name}"><span>${name}</span>`;
    button.addEventListener("click", () => {
      document.getElementById("modalImage").src = src;
    });
    debugStrip.appendChild(button);
  }
  document.getElementById("modalMetrics").innerHTML = `
    <div class="metric-section">${currentLang === "zh" ? "1. 最終評分主因" : "1. Grade Drivers"}</div>
    <div><span>Confidence</span><strong>${metricValue(item.conf, 2)}</strong></div>
    <div><span>BBox sharpness</span><strong>${metricValue(item.sharpness_score, 1)}</strong></div>
    <div><span>Track stability</span><strong>${metricValue(item.track_stability_score, 1)}</strong></div>
    <div><span>BBox frame area</span><strong>${metricValue(item.bbox_area_pct, 2)}%</strong></div>
    <div class="metric-note">${t("grade_tip")}</div>
    <div class="metric-section">${currentLang === "zh" ? "2. BBox 影像處理細節" : "2. BBox Image Processing Details"}</div>
    <div><span>Contour clarity score</span><strong>${metricValue(item.contour_clarity_score, 1)}</strong></div>
    <div><span>Laplacian variance</span><strong>${metricValue(item.laplacian_var, 1)}</strong></div>
    <div><span>Tenengrad score</span><strong>${metricValue(item.tenengrad_score, 1)}</strong></div>
    <div><span>Edge contrast score</span><strong>${metricValue(item.edge_contrast_score, 1)}</strong></div>
    <div><span>Edge density score</span><strong>${metricValue(item.edge_density_score, 1)}</strong></div>
    <div><span>Edge density</span><strong>${metricValue(item.edge_density, 4)}</strong></div>
    <div><span>BBox brightness</span><strong>${metricValue(item.brightness, 1)}</strong></div>
    <div><span>BBox contrast</span><strong>${metricValue(item.contrast, 1)}</strong></div>
    <div class="metric-section">${currentLang === "zh" ? "3. Tracking 與整體畫面背景" : "3. Tracking And Frame Context"}</div>
    <div><span>Track smoothness</span><strong>${metricValue(item.track_smoothness_score, 1)}</strong></div>
    <div><span>Track continuity</span><strong>${metricValue(item.track_continuity_score, 1)}</strong></div>
    <div><span>Frame clarity</span><strong>${metricValue(metrics.frame_sharpness, 1)}</strong></div>
    <div><span>Frame exposure</span><strong>${metricValue(metrics.frame_exposure_score, 1)}</strong></div>
    <div><span>Dynamic blur</span><strong>${metricValue(metrics.frame_motion_blur, 1)}</strong></div>
    <div><span>Lux proxy</span><strong>${metricValue(metrics.frame_lux_proxy, 1)}</strong></div>
    <div class="metric-note">${metricZeroNote(item)}</div>
  `;
  modal.classList.add("show");
  modal.setAttribute("aria-hidden", "false");
}

function closeModal() {
  modal.classList.remove("show");
  modal.setAttribute("aria-hidden", "true");
}

async function refreshStatus() {
  try {
    const data = await api("/api/status");
    latestStatus = data;
    setRunState(data.error ? "Error" : data.running ? "Running" : "Idle", Boolean(data.error));
    document.getElementById("mSamples").textContent = formatNumber(data.object_samples, 0);
    document.getElementById("mConf").textContent = formatNumber(data.summary.avg_conf, 2);
    document.getElementById("mArea").textContent = `${formatNumber(data.summary.avg_bbox_area_pct, 2)}%`;
    document.getElementById("mSharp").textContent = formatNumber(data.summary.avg_sharpness, 1);
    document.getElementById("mRecommended").textContent = `${formatNumber(data.summary.recommended_ratio * 100, 1)}%`;
    const frameCount = data.frame_count ? ` / ${formatNumber(data.frame_count, 0)}` : "";
    document.getElementById("progressText").textContent = `Frame ${formatNumber(data.frame_index, 0)}${frameCount}`;
    updateTable(data.current_objects || []);
    updateSamples(data.examples || []);
    drawMatrixChart(data.area_conf_matrix || []);
    updateRadarPanels(data);
    drawGradeChart(data.grade_distribution || []);
    drawViolinChart(data.metric_distribution || []);
    updateRecommendations(data.recommendations || []);
    updateTextList("trackingRecommendations", data.tracking_recommendations || []);
    if (data.error) showToast(data.error);
  } catch (error) {
    setRunState("Error", true);
  }
}

function updateTextList(id, items) {
  const box = document.getElementById(id);
  box.innerHTML = "";
  for (const item of items || []) {
    const p = document.createElement("p");
    p.textContent = localizeRecommendation(item);
    box.appendChild(p);
  }
}

function localizeRecommendation(text) {
  if (currentLang === "zh") return text;
  const sourceType = latestStatus && latestStatus.source && latestStatus.source.type;
  const samples = latestStatus ? Number(latestStatus.object_samples || 0) : 0;
  if (sourceType === "images") {
    return {
      en: "Image files are discrete samples. Detection and image quality are evaluated, but tracking continuity and ByteTrack stability are not.",
      ja: "画像ファイルは離散サンプルです。検出と画質のみ評価し、tracking の連続性や ByteTrack 安定性は評価しません。",
      ko: "이미지 파일은 개별 샘플입니다. 검출과 이미지 품질만 평가하며 tracking 연속성 및 ByteTrack 안정성은 평가하지 않습니다.",
    }[currentLang];
  }
  if (samples < 20) {
    return {
      en: "Fewer than 20 person bbox samples were collected. Use a longer clip or a scene with more people before drawing customer-facing conclusions.",
      ja: "person bbox サンプルが 20 未満です。顧客向け結論には、より長い動画または人物が多いシーンを推奨します。",
      ko: "person bbox 샘플이 20개 미만입니다. 고객용 결론에는 더 긴 영상 또는 사람이 많은 장면을 권장합니다.",
    }[currentLang];
  }
  return {
    en: "Review the matrix and radar charts to define the quality range where this model is reliable.",
    ja: "行列とレーダー図を確認し、このモデルが安定する品質範囲を定義してください。",
    ko: "매트릭스와 레이더 차트를 확인하여 이 모델이 안정적인 품질 범위를 정의하세요.",
  }[currentLang] || text;
}

loadModelBtn.addEventListener("click", async () => {
  try {
    await api("/api/load_model", { method: "POST", body: JSON.stringify({ model: modelSelect.value }) });
    showToast(`${t("loaded")} ${modelSelect.value}`);
  } catch (error) {
    showToast(error.message);
  }
});

startBtn.addEventListener("click", async () => {
  try {
    await api("/api/start", {
      method: "POST",
      body: JSON.stringify({
        model: modelSelect.value,
        source: selectedSource(),
        conf_threshold: Number(document.getElementById("confThreshold").value || 0.25),
        sample_stride: Number(document.getElementById("sampleStride").value || 1),
      }),
    });
    showToast(t("analysis_started"));
  } catch (error) {
    showToast(error.message);
  }
});

stopBtn.addEventListener("click", async () => {
  try {
    await api("/api/stop", { method: "POST", body: "{}" });
    showToast(t("stop_requested"));
  } catch (error) {
    showToast(error.message);
  }
});

reportBtn.addEventListener("click", async () => {
  try {
    const data = await api("/api/report", {
      method: "POST",
      body: JSON.stringify({
        name: document.getElementById("reportName").value,
        tags: document.getElementById("reportTags").value,
        language: currentLang,
      }),
    });
    const downloadLink = document.createElement("a");
    downloadLink.href = `/reports/${data.html}?download=1`;
    downloadLink.download = data.html;
    document.body.appendChild(downloadLink);
    downloadLink.click();
    downloadLink.remove();
    const downloadText = currentLang === "zh"
      ? "已觸發瀏覽器下載；也已同步保存到 Docker 掛載的 Downloads。"
      : "Browser download started; also saved to the Docker-mounted Downloads folder.";
    document.getElementById("reportResult").innerHTML =
      `<a href="/reports/${data.html}" target="_blank" rel="noreferrer">${data.html}</a><br><span>${downloadText}</span>`;
    showToast("Report exported");
  } catch (error) {
    showToast(error.message);
  }
});

downloadImageBtn.addEventListener("click", async () => {
  try {
    const url = document.getElementById("datasetUrl").value;
    const data = await api("/api/images/download", {
      method: "POST",
      body: JSON.stringify({ url }),
    });
    showToast(`${t("downloaded")}: ${data.file}`);
    await loadAssets();
  } catch (error) {
    showToast(error.message);
  }
});

languageSelect.addEventListener("change", () => {
  currentLang = languageSelect.value;
  localStorage.setItem("assessment_lang", currentLang);
  applyLanguage();
});

document.querySelectorAll("input[name='sourceType']").forEach((input) => {
  input.addEventListener("change", updateSourcePanels);
});

chart.addEventListener("click", (event) => loadBinSample(matrixCellFromEvent(event)));
chart.addEventListener("mousemove", (event) => {
  const cell = matrixCellFromEvent(event);
  chart.style.cursor = cell && cell.count > 0 ? "pointer" : "default";
});
binDrawCanvas.addEventListener("mousedown", startBinDraw);
binDrawCanvas.addEventListener("mousemove", moveBinDraw);
window.addEventListener("mouseup", endBinDraw);
binDrawCanvas.addEventListener("touchstart", startBinDraw, { passive: false });
binDrawCanvas.addEventListener("touchmove", moveBinDraw, { passive: false });
window.addEventListener("touchend", endBinDraw);
binResetBtn.addEventListener("click", resetBinCanvas);
window.addEventListener("resize", requestChartResize);
gradeChart.addEventListener("mousemove", (event) => {
  const hit = canvasHit(gradeChart, gradeHoverRegions, event);
  gradeChart.style.cursor = hit ? "help" : "default";
  showChartTooltip(event, hit && hit.html);
});
gradeChart.addEventListener("mouseleave", hideChartTooltip);
violinChart.addEventListener("mousemove", (event) => {
  const hit = canvasHit(violinChart, violinHoverRegions, event);
  violinChart.style.cursor = hit ? "help" : "default";
  showChartTooltip(event, hit && hit.html);
});
violinChart.addEventListener("mouseleave", hideChartTooltip);

modalClose.addEventListener("click", closeModal);
modal.addEventListener("click", (event) => {
  if (event.target === modal) closeModal();
});
window.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeModal();
});

document.querySelectorAll(".metric").forEach((card) => {
  card.tabIndex = 0;
  card.addEventListener("click", () => {
    const item = (latestStatus && latestStatus.current_objects && latestStatus.current_objects[0])
      || (latestStatus && latestStatus.examples && latestStatus.examples[0]);
    if (item) openAnalysis(item);
    else showToast("No bbox sample is available yet");
  });
});

setupResizableRows();
resizeAllCanvases();
drawMatrixChart([]);
updateRadarPanels({});
drawGradeChart([]);
drawViolinChart([]);

loadAssets()
  .then(() => {
    languageSelect.value = currentLang;
    updateSourcePanels();
    requestChartResize();
    applyLanguage();
    return refreshStatus();
  })
  .catch((error) => showToast(error.message));

window.setInterval(refreshStatus, 1000);
