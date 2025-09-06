import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import rateLimit from 'express-rate-limit';
import OpenAI from 'openai';

/** ====== Config ====== */
const app = express();
const PORT = process.env.PORT || 3000;

// Cho phép đúng origin GitHub Pages của bạn
const ALLOWED_ORIGIN = process.env.ORIGIN || 'https://<username>.github.io/<repo>';
// Bảo vệ đơn giản bằng token riêng (không phải OpenAI key)
const APP_TOKEN = process.env.APP_TOKEN || ''; // đặt trên Render

// Model OpenAI (chỉnh theo nhu cầu)
const MODEL = process.env.MODEL || 'gpt-4o-mini';

// OpenAI client (key lưu ở Render)
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

/** ====== Middlewares ====== */
app.use(express.json({ limit: '8mb' }));

app.use(cors({
  origin(origin, cb) {
    // Cho phép direct requests từ Pages và local dev
    const ok =
      !origin ||
      origin === ALLOWED_ORIGIN ||
      origin.endsWith('.github.io');
    cb(null, ok);
  },
  methods: ['POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  maxAge: 86400
}));

// Preflight
app.options('*', cors());

// Rate limit theo IP
app.use('/api/', rateLimit({
  windowMs: 60 * 1000,
  max: 30, // 30 req/phút/IP
  standardHeaders: true,
  legacyHeaders: false
}));

// Auth đơn giản bằng Bearer APP_TOKEN
app.use('/api/', (req, res, next) => {
  if (!APP_TOKEN) return next(); // cho phép nếu bạn chưa bật token
  const auth = req.headers.authorization || '';
  const token = auth.startsWith('Bearer ') ? auth.slice(7) : '';
  if (token !== APP_TOKEN) return res.status(401).json({ error: 'Unauthorized' });
  next();
});

/** ====== Helpers ====== */
function linesToItems(text) {
  // Cố gắng trích danh sách CLO từ text dạng gạch đầu dòng/dòng mới
  return String(text)
    .split(/\r?\n/)
    .map(s => s.trim())
    .filter(s => s && !/^#+\s/.test(s))
    .map(s => s.replace(/^\d+[\).]\s*/, '').replace(/^[-*•]\s*/, ''));
}

/** ====== Endpoints ====== */

// Gợi ý CLO (đầu vào có cấu trúc)
app.post('/api/suggest', async (req, res) => {
  try {
    const { plo, ploText = '', course = {}, level = 'I', bloomVerbs = [], count = 6 } = req.body || {};
    const verbs = (bloomVerbs || [])
      .slice(0, 100)
      .map(v => `${v.verb}(${v.level})`)
      .join(', ');

    const prompt = `Bạn là chuyên gia xây dựng CLO.
Yêu cầu: đề xuất ${count} CLO ngắn gọn bằng tiếng Việt cho học phần "${course.label || ''} — ${course.fullname || ''}" (tín chỉ: ${course.tong ?? ''}).
Mỗi CLO có dạng: CLOx: <Động từ Bloom> <mô tả cụ thể, có đo lường>.
PLO liên quan: ${plo} — ${ploText}
Mức liên kết PLO–COURSE: ${level} (I/R/M/A). Ưu tiên dùng các động từ sau: ${verbs}.
Trả về mỗi CLO trên **một dòng**. Không cần giải thích thêm.`;

    const rsp = await client.responses.create({
      model: MODEL,
      input: prompt,
      max_output_tokens: 800,
      temperature: 0.5
    }); // Responses API → output_text
    const text = rsp.output_text || '';
    const items = linesToItems(text).slice(0, count);

    res.json({ items, raw: text });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'suggest_failed' });
  }
});

// Đánh giá CLO ↔ PLO
app.post('/api/evaluate', async (req, res) => {
  try {
    const { plo, ploText = '', cloText = '' } = req.body || {};
    const prompt = `Đánh giá mức phù hợp của CLO với PLO (ngắn gọn ≤ 120 từ, bullet nếu cần).
- PLO (${plo}): ${ploText}
- CLO: ${cloText}
Hãy nêu: mức phù hợp (cao/vừa/thấp), điểm mạnh, điểm cần chỉnh sửa, gợi ý động từ Bloom/thước đo.`;

    const rsp = await client.responses.create({
      model: MODEL,
      input: prompt,
      max_output_tokens: 300,
      temperature: 0.2
    });
    const text = rsp.output_text || '';
    res.json({ text });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'evaluate_failed' });
  }
});

// (Tuỳ chọn) prompt thô
app.post('/api/raw', async (req, res) => {
  try {
    const { prompt = '', max_tokens = 600 } = req.body || {};
    const rsp = await client.responses.create({
      model: MODEL,
      input: prompt,
      max_output_tokens: max_tokens
    });
    res.json({ text: rsp.output_text || '' });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'raw_failed' });
  }
});

app.get('/health', (req, res) => res.status(200).send('ok'));

app.listen(PORT, () => {
  console.log(`cm-gpt-service listening on :${PORT}`);
});
