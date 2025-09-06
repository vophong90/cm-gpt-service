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
app.get('/health', (_req, res) => res.status(200).send('ok'));
app.get('/whoami', (_req, res) =>
  res.json({
    ok: true,
    allowed_origins: [...ALLOWED],
    model: DEFAULT_MODEL,
    model_mode: MODEL_MODE
  })
);

// ==== /api/suggest ====  (gợi ý CLO)
app.post('/api/suggest', async (req, res) => {
  try {
    const {
      plo,
      ploText = '',
      course = {},
      level = 'I',
      bloomVerbs = [],
      count = 6,
      model // optional override
    } = req.body || {};

    const verbs = (bloomVerbs || [])
      .slice(0, 120)
      .map(v => `${v.verb}(${v.level})`)
      .join(', ');

    const prompt = `Bạn là chuyên gia xây dựng CLO.
Yêu cầu: đề xuất ${count} CLO ngắn gọn bằng tiếng Việt cho học phần "${course.label || ''} — ${course.fullname || ''}" (tín chỉ: ${course.tong ?? ''}).
Mỗi CLO có dạng: CLOx: <Động từ Bloom> <mô tả cụ thể, có đo lường>.
PLO liên quan: ${plo} — ${ploText}
Mức liên kết PLO–COURSE: ${level} (I/R/M/A). Ưu tiên dùng các động từ sau: ${verbs}.
Trả về mỗi CLO trên **một dòng**. Không cần giải thích thêm.`;

    const modelName = pickModel('suggest', model);

    const rsp = await createResponseWithFallback({
      model: modelName,
      input: prompt,
      // GPT-5: ngắn gọn + nhanh
      text: { verbosity: 'low' },            // ← GPT-5 param (có fallback)
      reasoning: { effort: 'minimal' },      // ← GPT-5 param (có fallback)
      max_output_tokens: 800,
      temperature: 0.5
    });

    const text = rsp.output_text || '';
    const items = linesToItems(text).slice(0, count);
    res.json({ items, raw: text, model: modelName });
  } catch (err) {
    console.error('[suggest_failed]', err?.message || err);
    res.status(500).json({ error: 'suggest_failed' });
  }
});

// ==== /api/evaluate ====  (đánh giá CLO ↔ PLO)
app.post('/api/evaluate', async (req, res) => {
  try {
    const { plo, ploText = '', cloText = '', model } = req.body || {};

    const prompt = `Đánh giá mức phù hợp của CLO với PLO (ngắn gọn ≤ 120 từ, bullet nếu cần).
- PLO (${plo}): ${ploText}
- CLO: ${cloText}
Hãy nêu: mức phù hợp (cao/vừa/thấp), điểm mạnh, điểm cần chỉnh sửa, gợi ý động từ Bloom/thước đo.`;

    const modelName = pickModel('evaluate', model);

    const rsp = await createResponseWithFallback({
      model: modelName,
      input: prompt,
      // GPT-5: cân bằng chất lượng
      text: { verbosity: 'medium' },       // ← GPT-5 param (có fallback)
      reasoning: { effort: 'medium' },     // ← GPT-5 param (có fallback)
      max_output_tokens: 360,
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
