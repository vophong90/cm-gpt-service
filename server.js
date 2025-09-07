import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import rateLimit from 'express-rate-limit';
import OpenAI from 'openai';

/** ====== Config ====== */
const app = express();
const PORT = process.env.PORT || 3000;

// CORS: cho 1 hoặc nhiều origin
const ALLOWED = new Set(
  (process.env.ORIGINS || process.env.ORIGIN || '')
    .split(',')
    .map(s => s.trim())
    .filter(Boolean)
);
// Cho phép GitHub Pages của bạn ngay cả khi quên set env (tối thiểu)
if (ALLOWED.size === 0) ALLOWED.add('https://vophong90.github.io');

// Bảo vệ endpoint bằng token riêng (không phải OpenAI key)
const APP_TOKEN = process.env.APP_TOKEN || '';

// Model & chế độ auto
// MODEL: tên model mặc định (gợi ý đặt 'gpt-5')
// MODEL_MODE: auto | full | mini (mặc định auto)
const DEFAULT_MODEL = process.env.MODEL || 'gpt-5';
const MODEL_MODE = (process.env.MODEL_MODE || 'auto').toLowerCase();

// OpenAI client
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

/** ====== Middlewares ====== */
app.use(express.json({ limit: '8mb' }));

app.use(
  cors({
    origin(origin, cb) {
      const ok = !origin || [...ALLOWED].some(o => origin === o || origin.endsWith('.github.io'));
      cb(null, ok);
    },
    methods: ['POST', 'OPTIONS', 'GET'],
    allowedHeaders: ['Content-Type', 'Authorization'],
    maxAge: 86400
  })
);

// Preflight
app.options('*', cors());

// Rate limit theo IP cho /api/*
app.use(
  '/api/',
  rateLimit({
    windowMs: 60 * 1000,
    max: 30,
    standardHeaders: true,
    legacyHeaders: false
  })
);

// Bearer token (nếu cấu hình)
app.use('/api/', (req, res, next) => {
  if (!APP_TOKEN) return next();
  const auth = req.headers.authorization || '';
  const token = auth.startsWith('Bearer ') ? auth.slice(7) : '';
  if (token !== APP_TOKEN) return res.status(401).json({ error: 'Unauthorized' });
  next();
});

/** ====== Helpers ====== */

// Chọn model theo endpoint + chế độ
function pickModel(kind /* 'suggest' | 'evaluate' */, overrideModel) {
  if (overrideModel) return overrideModel; // cho phép client override nếu cần

  const full = 'gpt-5';
  const mini = 'gpt-5-mini';

  if (MODEL_MODE === 'full') return full;
  if (MODEL_MODE === 'mini') return mini;

  // auto
  if (kind === 'suggest') return mini;     // gợi ý → ưu tiên nhanh
  if (kind === 'evaluate') return full;    // đánh giá → ưu tiên chất lượng
  return DEFAULT_MODEL; // dự phòng
}

// Tách list từng dòng thành mảng item
function linesToItems(text) {
  return String(text)
    .split(/\r?\n/)
    .map(s => s.trim())
    .filter(s => s && !/^#+\s*/.test(s))
    .map(s => s.replace(/^\d+[\).]\s*/, '').replace(/^[-*•]\s*/, ''));
}

// Gọi Responses API với fallback nếu server không chấp nhận tham số GPT-5 mới
async function createResponseWithFallback(params, { stripVerbosity = false, stripReasoning = false } = {}) {
  try {
    return await client.responses.create(params);
  } catch (e) {
    const msg = (e?.message || '').toLowerCase();
    const isUnknownParam =
      msg.includes('unknown') ||
      msg.includes('unrecognized') ||
      msg.includes('invalid parameter') ||
      msg.includes('unsupported');

    // Fallback 1: bỏ verbosity
    if (!stripVerbosity && params?.text?.verbosity) {
      const p2 = { ...params, text: { ...(params.text || {}) } };
      delete p2.text.verbosity;
      return await createResponseWithFallback(p2, { stripVerbosity: true, stripReasoning });
    }
    // Fallback 2: bỏ reasoning.effort
    if (!stripReasoning && params?.reasoning?.effort) {
      const p2 = { ...params };
      delete p2.reasoning;
      return await createResponseWithFallback(p2, { stripVerbosity: true, stripReasoning: true });
    }
    throw e;
  }
}

/** ====== Endpoints ====== */

// Health & whoami
app.get('/healthz', (_req, res) => res.status(200).send('ok'));
app.get('/whoami', (_req, res) =>
  res.json({
    ok: true,
    allowed_origins: [...ALLOWED],
    model: DEFAULT_MODEL,
    model_mode: MODEL_MODE
  })
);

// ==== /api/suggest ====  (gợi ý CLO — bản có Bloom & YHCT+YHHĐ)
app.post('/api/suggest', async (req, res) => {
  try {
    const {
      plo,
      ploText = '',
      course = {},
      level: levelFromBody,            // I | R | M | A (cũ)
      linkLevel: linkLevelFromBody, 
      bloomVerbs = [],          // [{ verb, level }, ...] từ webapp (CSV Bloom người dùng tải)
      count = 6,
      model                    // optional override
    } = req.body || {};

    // Ánh xạ mức liên kết → bậc Bloom phù hợp
    const LEVEL2BLOOM = {
      I: ['Remember','Understand'],
      R: ['Apply','Analyze'],
      M: ['Analyze','Evaluate'],
      A: ['Evaluate','Create']
    };
    const LEVEL = linkLevelFromBody ?? levelFromBody ?? 'I';
    const linkLevel = String(LEVEL).toUpperCase();
    const targetBloomLevels = LEVEL2BLOOM[linkLevel] || LEVEL2BLOOM.I;

    // Chuẩn hoá & lọc động từ từ client
    const norm = (x) => {
      if (!x) return null;
      if (typeof x === 'string') return { verb: x.trim(), level: '' };
      const v = String(x.verb || '').trim();
      const l = String(x.level || '').trim();
      return v ? { verb: v, level: l } : null;
    };

    const seen = new Set();
    // Ưu tiên: chỉ lấy verbs có level thuộc targetBloomLevels; nếu rỗng thì fallback lấy tất cả
    let filtered = (bloomVerbs || [])
      .map(norm)
      .filter(Boolean);

    const byTarget = filtered.filter(x => targetBloomLevels.includes(x.level));
    if (byTarget.length > 0) filtered = byTarget;

    // Loại trùng theo động từ (không phân biệt hoa thường), cắt bớt cho gọn prompt
    const uniq = [];
    for (const it of filtered) {
      const k = it.verb.toLowerCase();
      if (!seen.has(k)) { seen.add(k); uniq.push(it); }
    }
    const MAX_VERBS = 180; // tránh prompt quá dài
    const verbsForPrompt = uniq.slice(0, MAX_VERBS)
      .map(it => `${it.verb} (${it.level || '—'})`)
      .join(', ');

    // Fallback nếu danh sách rỗng hoàn toàn
    const fallbackVerbs = ['Mô tả','Giải thích','Vận dụng','Phân tích','Đánh giá','Xây dựng'];
    const VERB_LIST = verbsForPrompt || fallbackVerbs.join(', ');

    // Thông tin học phần
    const courseLabel = (course.label || course.id || '').trim();
    const courseFull  = (course.fullname || '').trim();
    const credits     = (course.tong ?? course.credits ?? '').toString();

    // Prompt theo đặc tả của bạn (YHCT + YHĐ, đo lường, 1 ý, 25–30 từ, đúng COUNT dòng)
    const prompt = `
Bạn là chuyên gia xây dựng chuẩn đầu ra học phần y khoa tích hợp Y học hiện đại (YHHĐ) và Y học cổ truyền (YHCT).

Nhiệm vụ: đề xuất ${count} CLO bằng tiếng Việt cho học phần "${courseLabel} — ${courseFull}" (TC: ${credits}).

Mỗi CLO đúng chuẩn:
- Bắt đầu bằng một ĐỘNG TỪ Bloom thuộc mức ${linkLevel} (chỉ chọn trong danh sách SAU; không lặp lại động từ nếu có thể).
- Phù hợp và thể hiện rõ PLO ${plo}: ${ploText}
- Bám sát đặc thù học phần này (thuật ngữ, quy trình, ca lâm sàng, đối tượng người bệnh…).
- TÍNH ĐO LƯỜNG: nêu điều kiện/tiêu chí kiểm tra (ví dụ: theo hướng dẫn quốc gia, ≥80% tình huống chuẩn, trong 10 phút, dùng thước đo X, theo phác đồ Y…).
- Với chương trình YHCT + YHHĐ: nếu thích hợp, thể hiện TÍCH HỢP (so sánh, phối hợp, chỉ định/chống chỉ định của cả hai hệ).

Ràng buộc phong cách:
- Mỗi CLO chỉ 1 ý (không nối “và”), tối đa 25–30 từ, viết súc tích.
- Không giải thích thêm, không bullet, không đánh số bằng dấu chấm; mỗi CLO trên MỘT DÒNG dạng: CLOx: <nội dung>.

DANH SÁCH ĐỘNG TỪ CHO PHÉP (chỉ dùng các từ này, ưu tiên phù hợp mức ${linkLevel}): ${VERB_LIST}

Trả về đúng ${count} dòng, lần lượt: CLO1:, CLO2:, ..., CLO${count}:.
`.trim();

    const modelName = pickModel('suggest', model);

    const rsp = await createResponseWithFallback({
      model: modelName,
      input: prompt,
      text: { verbosity: 'low' },         // GPT-5 param (có fallback)
      reasoning: { effort: 'minimal' },   // GPT-5 param (có fallback)
      max_output_tokens: 900,
      temperature: 0.4
    });

    const raw = rsp.output_text || '';
    // Tách thành từng dòng, bỏ bullet/số thứ tự nếu có
    const items = linesToItems(raw)
      .filter(line => /^CLO\s*\d+\s*:/.test(line)) // giữ các dòng bắt đầu bằng CLOx:
      .slice(0, count);

    // Nếu model không obey số lượng, cố cân chỉnh lại (lấy thêm dòng text thường)
    while (items.length < count) {
      const extras = linesToItems(raw).filter(s => !/^CLO\s*\d+\s*:/.test(s));
      if (!extras.length) break;
      items.push(`CLO${items.length+1}: ${extras.shift()}`);
    }
    res.json({ items: items.slice(0, count), raw, model: modelName });
  } catch (err) {
    console.error('[suggest_failed]', err?.message || err);
    res.status(500).json({ error: 'suggest_failed' });
  }
});

// ==== /api/evaluate ====  (đánh giá CLO ↔ PLO — bản nâng cao YHCT+YHHD & Bloom)
app.post('/api/evaluate', async (req, res) => {
  try {
    const {
      plo,                 // "PLO1"
      ploText = '',        // nội dung PLO
      cloText = '',        // nội dung CLO cần đánh giá
      level: levelFromBody,            // I | R | M | A (cũ)
      linkLevel: linkLevelFromBody, 
      course = {},         // { id, label, fullname, tong }
      bloomVerbs = [],     // [{verb, level}] từ webapp (CSV Bloom)
      model                // optional override
    } = req.body || {};

    // Ánh xạ mức liên kết → bậc Bloom tham chiếu
    const LEVEL2BLOOM = {
      I: ['Remember','Understand'],
      R: ['Apply','Analyze'],
      M: ['Analyze','Evaluate'],
      A: ['Evaluate','Create']
    };
    const LEVEL = linkLevelFromBody ?? levelFromBody ?? 'I';
    const linkLevel = String(LEVEL).toUpperCase();
    const targetBloomLevels = LEVEL2BLOOM[linkLevel] || LEVEL2BLOOM.I;

    // Lọc động từ theo bậc Bloom mong muốn (nếu không có, fallback toàn bộ)
    const norm = (x) => {
      if (!x) return null;
      if (typeof x === 'string') return { verb: x.trim(), level: '' };
      const v = String(x.verb || '').trim();
      const l = String(x.level || '').trim();
      return v ? { verb: v, level: l } : null;
    };

    const allVerbs = (bloomVerbs || []).map(norm).filter(Boolean);
    const matched = allVerbs.filter(x => targetBloomLevels.includes(x.level));
    const pool = (matched.length ? matched : allVerbs);

    // Loại trùng & giới hạn độ dài danh sách
    const seen = new Set();
    const uniq = [];
    for (const it of pool) {
      const k = it.verb.toLowerCase();
      if (!seen.has(k)) { seen.add(k); uniq.push(it); }
    }
    const MAX_VERBS = 150;
    const verbsForPrompt = uniq.slice(0, MAX_VERBS)
      .map(it => `${it.verb} (${it.level || '—'})`)
      .join(', ');

    const fallbackVerbs = ['Mô tả','Giải thích','Vận dụng','Phân tích','Đánh giá','Xây dựng'];
    const VERB_LIST = verbsForPrompt || fallbackVerbs.join(', ');

    // Thông tin học phần
    const courseLabel = (course.label || course.id || '').trim();
    const courseFull  = (course.fullname || '').trim();
    const credits     = (course.tong ?? course.credits ?? '').toString();

    // Prompt đánh giá nâng cao (có rubric & gợi ý chỉnh)
    const prompt = `
Bạn là chuyên gia xây dựng & thẩm định CLO cho chương trình bác sĩ tích hợp Y học hiện đại (YHHD) và Y học cổ truyền (YHCT).

BỐI CẢNH
- Học phần: "${courseLabel} — ${courseFull}" (TC: ${credits})
- PLO liên quan: ${plo}: ${ploText}
- Mức liên kết PLO–COURSE: ${linkLevel} (I/R/M/A)
- Danh sách ĐỘNG TỪ Bloom cho phép (ưu tiên đúng bậc ${linkLevel}): ${VERB_LIST}

NHIỆM VỤ
ĐÁNH GIÁ nội dung CLO sau (tiếng Việt):
«${cloText}»

YÊU CẦU ĐẦU RA — TRẢ KẾT QUẢ NGẮN GỌN, TIẾNG VIỆT, THEO ĐÚNG CÁC MỤC:
1) **Phán quyết** (Cao/Vừa/Thấp) về mức phù hợp PLO & học phần + 1 câu giải thích.
2) **Rubric (0–4 điểm từng tiêu chí; tổng quy ra /100)**:
   - PLO alignment: mức thể hiện đúng ý PLO (0–4)
   - Bloom & động từ: CLO mở đầu bằng động từ trong danh sách? bậc đúng mức ${linkLevel}? (0–4)
   - Tính đo lường: có tiêu chí/chuẩn/nguỡng/thời gian/công cụ đo? (0–4)
   - Phù hợp học phần: có thuật ngữ, quy trình, đối tượng người bệnh của học phần? (0–4)
   - Tích hợp YHCT+YHHD (nếu phù hợp): so sánh/phối hợp/chỉ định–chống chỉ định? (0–4)
   Viết dạng: Điểm: a,b,c,d,e; Tổng: X/20 → Y/100.
3) **Điểm mạnh** (2–4 gạch đầu dòng ngắn).
4) **Vấn đề & cách sửa** (2–4 gạch đầu dòng, hành động cụ thể).
5) **Đề xuất phiên bản CLO đã chỉnh** (1 dòng, ≤ 25–30 từ, đúng 1 ý, bắt đầu bằng một ĐỘNG TỪ trong danh sách; có tiêu chí đo lường; nếu phù hợp, thể hiện tích hợp YHCT+YHHD). Định dạng: CLO*: <nội dung>.
6) **Cảnh báo rủi ro** (nếu có: mơ hồ, 2 ý trong 1 CLO, không đo lường, sai bậc Bloom…).

QUY TẮC
- Ngắn gọn, rõ, không kèm giải thích ngoài các mục trên.
- Không thêm bullet cho tiêu đề; dùng bullet ngắn gọn trong mục 3) và 4).
- Luôn giữ tiếng Việt và đúng ngữ cảnh YHCT + YHHD.
`.trim();

    const modelName = pickModel('evaluate', model);

    const rsp = await createResponseWithFallback({
      model: modelName,
      input: prompt,
      // GPT-5: cân bằng chất lượng
      text: { verbosity: 'medium' },       // có fallback trong helper
      reasoning: { effort: 'medium' },     // có fallback trong helper
      max_output_tokens: 600,
      temperature: 0.2
    });

    const text = rsp.output_text || '';
    res.json({ text, model: modelName });
  } catch (err) {
    console.error('[evaluate_failed]', err?.message || err);
    res.status(500).json({ error: 'evaluate_failed' });
  }
});

// ==== /api/raw ==== (tuỳ chọn) — prompt thô để debug
app.post('/api/raw', async (req, res) => {
  try {
    const { prompt = '', max_tokens = 600, model } = req.body || {};
    const modelName = model || DEFAULT_MODEL;

    const rsp = await createResponseWithFallback({
      model: modelName,
      input: prompt,
      text: { verbosity: 'low' },
      max_output_tokens: max_tokens
    });

    res.json({ text: rsp.output_text || '', model: modelName });
  } catch (err) {
    console.error('[raw_failed]', err?.message || err);
    res.status(500).json({ error: 'raw_failed' });
  }
});

app.listen(PORT, () => {
  console.log(`cm-gpt-service listening on :${PORT}`, { MODEL_MODE, DEFAULT_MODEL, ALLOWED: [...ALLOWED] });
});
