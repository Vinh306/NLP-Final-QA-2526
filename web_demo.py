"""
web_demo.py — Demo web cho 4 cấu hình QA hệ thống quy chế TDTU
Chạy trên Colab: exec(open('web_demo.py').read())
"""

import gradio as gr
import os, time, threading, numpy as np
from datetime import datetime

CONFIG_META = {
    "A": {"label": "A — LLM gốc",         "color": "#4A90D9",
          "desc": "Qwen2.5-3B-Instruct · Không RAG · Không fine-tune", "has_rag": False},
    "B": {"label": "B — LLM gốc + RAG",   "color": "#27AE60",
          "desc": "Qwen2.5-3B-Instruct · FAISS RAG · Không fine-tune", "has_rag": True},
    "C": {"label": "C — Fine-tuned",       "color": "#E67E22",
          "desc": "Qwen2.5-3B · QLoRA fine-tune ~400 cặp · Không RAG", "has_rag": False},
    "D": {"label": "D — Fine-tuned + RAG", "color": "#8E44AD",
          "desc": "Qwen2.5-3B · QLoRA fine-tune · FAISS RAG",          "has_rag": True},
}

# ─────────────────────────────────────────────────────────────────
# RAG CONTEXT HELPER
# ─────────────────────────────────────────────────────────────────
def get_contexts_with_scores(question, top_k=5):
    """Lấy top-k chunks + similarity scores từ FAISS index."""
    try:
        q_emb = rag.embedder.encode(
            [question], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        scores, indices = rag.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(rag.chunks):
                results.append((rag.chunks[idx], float(score)))
        return results
    except Exception:
        return []


def render_context_html(contexts_with_scores, config_color):
    """Render HTML collapsible cho phần tài liệu tham chiếu + điểm số."""
    if not contexts_with_scores:
        return ""
    items = []
    for i, (chunk, score) in enumerate(contexts_with_scores):
        pct   = min(int(score * 100), 100)
        bar_c = "#27AE60" if score >= 0.70 else "#E67E22" if score >= 0.50 else "#E74C3C"
        preview = (chunk[:220] + "…") if len(chunk) > 220 else chunk
        items.append(f"""
        <div style="margin-bottom:8px;padding:9px 11px;
                    background:#fafafa;border-radius:7px;
                    border-left:3px solid {bar_c};font-size:12px;">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
            <b style="color:#444;white-space:nowrap;">Đoạn {i+1}</b>
            <div style="flex:1;background:#e4e4e4;border-radius:10px;height:5px;">
              <div style="width:{pct}%;background:{bar_c};border-radius:10px;height:5px;"></div>
            </div>
            <b style="color:{bar_c};white-space:nowrap;">{score:.3f}</b>
          </div>
          <div style="color:#333;line-height:1.55;">{preview}</div>
        </div>""")

    return (
        f'<details style="margin-top:4px;border:1px solid #e0e0e0;'
        f'border-radius:8px;overflow:hidden;">'
        f'<summary style="cursor:pointer;padding:7px 12px;'
        f'background:#f4f4f4;font-size:12px;font-weight:700;color:{config_color};">'
        f'Tài liệu tham chiếu (top-{len(contexts_with_scores)})'
        f'</summary>'
        f'<div style="padding:10px 10px 4px;">{"".join(items)}</div>'
        f'</details>'
    )


# RESPOND — chạy song song, stream từng kết quả khi xong
def respond(question,
            history_A, history_B, history_C, history_D,
            run_A, run_B, run_C, run_D):

    if not question.strip():
        yield history_A, history_B, history_C, history_D, "", "", ""
        return

    enabled  = {"A": run_A, "B": run_B, "C": run_C, "D": run_D}
    fns      = {"A": generate_A, "B": generate_B, "C": generate_C, "D": generate_D}
    cur      = {"A": list(history_A), "B": list(history_B),
                "C": list(history_C), "D": list(history_D)}
    ctx_html = {"B": "", "D": ""}

    # ① Fetch RAG contexts ngay (nhanh ~0.1s) — hiển thị trong lúc model chạy
    for key in ["B", "D"]:
        if enabled[key]:
            ctxs = get_contexts_with_scores(question, top_k=getattr(rag, "top_k", 5))
            ctx_html[key] = render_context_html(ctxs, CONFIG_META[key]["color"])

    # ② Thêm placeholder "đang xử lý" cho config được bật
    for key in "ABCD":
        if enabled[key]:
            cur[key] = cur[key] + [[question, "⏳ Đang xử lý…"]]

    # Yield ngay: UI phản hồi + contexts hiện ra liền
    yield (cur["A"], cur["B"], cur["C"], cur["D"], ctx_html["B"], ctx_html["D"], "")

    # ③ Chạy tất cả model trên thread riêng
    results = {}
    lock    = threading.Lock()
    done_ev = threading.Event()

    def run_model(key):
        t0 = time.time()
        try:
            ans = fns[key](question)
        except Exception as e:
            ans = f"[Lỗi: {e}]"
        with lock:
            results[key] = (ans, round(time.time() - t0, 2))
        done_ev.set()

    active  = {k for k in "ABCD" if enabled[k]}
    threads = {k: threading.Thread(target=run_model, args=(k,), daemon=True) for k in active}
    for t in threads.values():
        t.start()

    # ④ Poll mỗi 0.4s — yield ngay khi có model nào hoàn thành
    yielded   = set()
    lat_parts = []

    while len(yielded) < len(active):
        done_ev.wait(timeout=0.4)
        done_ev.clear()

        with lock:
            snap = dict(results)

        changed = False
        for key, (ans, elapsed) in snap.items():
            if key not in yielded:
                yielded.add(key)
                changed = True
                cur[key][-1][1] = ans
                c = CONFIG_META[key]["color"]
                lat_parts.append(
                    f'<span style="color:{c};font-weight:700;">Config {key}</span>'
                    f': {elapsed}s'
                )

        if changed:
            lat_html = (
                '<div style="padding:7px 12px;background:#f0f0f0;border-radius:8px;'
                'font-size:12px;display:flex;gap:14px;flex-wrap:wrap;align-items:center;">'
                + " &nbsp;|&nbsp; ".join(lat_parts)
                + f'<span style="color:#aaa;margin-left:auto;">'
                  f'⏱ {datetime.now().strftime("%H:%M:%S")}</span></div>'
            )
            yield (cur["A"], cur["B"], cur["C"], cur["D"], ctx_html["B"], ctx_html["D"], lat_html)

    for t in threads.values():
        t.join()


def clear_all():
    return [], [], [], [], "", "", ""



# CSS
CSS = """
body { font-family: 'Be Vietnam Pro', 'Segoe UI', sans-serif !important; }
.gradio-container { max-width: 1600px !important; margin: 0 auto; }

/* Header */
#app-header {
    background: linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
    padding: 18px 26px 14px; border-radius: 13px;
    margin-bottom: 12px; color: white;
}
#app-header h1 { margin:0 0 3px; font-size:21px; font-weight:700; color: white !important;}
#app-header p  { margin:0; opacity:0.6; font-size:12px; color: white !important;}

/* FIX CHÍNH: Buộc 4 panel nằm cùng 1 hàng, không wrap xuống ── */
.four-col-row {
    display: flex !important;
    flex-wrap: nowrap !important;
    gap: 10px !important;
    align-items: flex-start !important;
    overflow-x: auto !important;
}
/* Mỗi column chiếm đúng 25%, không co lại */
.four-col-row > .gr-column,
.four-col-row > div[class*="column"],
.four-col-row > div {
    flex: 1 1 0 !important;
    min-width: 220px !important;
    max-width: 25% !important;
    width: 25% !important;
    overflow: hidden !important;
}

/* Cards */
.config-card {
    border-radius: 11px !important;
    border: 2px solid transparent !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
    height: 100% !important;
}
.config-A { border-color: #4A90D9 !important; }
.config-B { border-color: #27AE60 !important; }
.config-C { border-color: #E67E22 !important; }
.config-D { border-color: #8E44AD !important; }

/* Chatbot bên trong card */
.chatbot-wrap { min-height: 0 !important; flex: 1 !important; }
.chatbot-wrap > div { padding-bottom: 0 !important; margin-bottom: 0 !important; }

/* Row context (bên dưới 4 panel) */
.ctx-row {
    display: flex !important;
    flex-wrap: nowrap !important;
    gap: 10px !important;
    margin-top: 6px !important;
}
.ctx-row > div {
    flex: 1 1 0 !important;
    min-width: 0 !important;
    width: 25% !important;
    max-width: 25% !important;
}

/* Input */
#question-row {
    background: #f5f5f5; border-radius: 10px;
    padding: 11px 13px; margin-top: 8px;
}
#send-btn {
    background: linear-gradient(135deg,#4A90D9,#8E44AD) !important;
    color: white !important; font-weight: 700 !important;
}
#send-btn:hover { opacity: 0.85 !important; }

/* Latency */
#latency-bar { min-height: 30px; margin-bottom: 4px; }

/* Toggles */
.cb-row label { font-size: 13px !important; }

/* Suggested */
.sq-btn {
    font-size: 11px !important; padding: 3px 9px !important;
    border-radius: 16px !important; white-space: nowrap !important;
}
"""

SUGGESTED_QUESTIONS = [
    "Sinh viên bị cảnh báo học vụ khi nào?",
    "Điều kiện nhận học bổng khuyến khích là gì?",
    "Vi phạm quy chế thi cử bị xử lý ra sao?",
    "Điều kiện xét tốt nghiệp là gì?",
    "Số tín chỉ tối thiểu mỗi học kỳ là bao nhiêu?",
    "Quy định về điểm rèn luyện như thế nào?",
]


# BUILD UI
def build_demo():
    with gr.Blocks(css=CSS, title="NLP Demo — QA TDTU") as demo:

        # Header
        gr.HTML("""
        <div id="app-header">
            <h1>Hệ thống Hỏi-Đáp Quy Chế TDTU</h1>
            <p>So sánh 4 cấu hình · Qwen2.5-3B-Instruct · QLoRA Fine-tune · FAISS RAG</p>
        </div>""")

        # Toggles
        with gr.Row():
            gr.HTML("<b style='line-height:32px;font-size:13px;'>Chọn config:</b>")
            cb_A = gr.Checkbox(value=True, label="A — LLM gốc", elem_classes="cb-row")
            cb_B = gr.Checkbox(value=True, label="B — LLM + RAG", elem_classes="cb-row")
            cb_C = gr.Checkbox(value=True, label="C — Fine-tuned", elem_classes="cb-row")
            cb_D = gr.Checkbox(value=True, label="D — Fine-tuned + RAG", elem_classes="cb-row")

        # Latency bar
        latency_out = gr.HTML("", elem_id="latency-bar")

        # 4 Panels (cùng 1 hàng, không wrap)
        panels   = {}
        ctx_outs = {}

        # Dùng elem_classes="four-col-row" để buộc flex nowrap
        with gr.Row(equal_height=True, elem_classes="four-col-row"):
            for key in "ABCD":
                m = CONFIG_META[key]
                with gr.Column(scale=1, min_width=220,
                               elem_classes=f"config-card config-{key}"):

                    gr.HTML(
                        f'<div style="background:linear-gradient(90deg,{m["color"]},{m["color"]}bb);'
                        f'color:white;padding:8px 13px;font-weight:700;font-size:13px;">'
                        f'Config {m["label"]}</div>'
                        f'<div style="font-size:11px;color:#666;padding:4px 13px 5px;'
                        f'background:#f9f9f9;border-bottom:1px solid #eee;'
                        f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
                        f'{m["desc"]}</div>'
                    )

                    panels[key] = gr.Chatbot(
                        label="", height=340,
                        show_label=False,
                        bubble_full_width=False,
                        elem_classes="chatbot-wrap",
                        show_copy_button=False,
                    )

        #  Context row — tách ra NGOÀI panel row để không làm lệch chiều cao 
        # Placeholder rỗng cho cột A và C, context thật cho B và D
        with gr.Row(elem_classes="ctx-row"):
            gr.HTML("")                                          # A — không có RAG
            ctx_outs["B"] = gr.HTML("", elem_id="ctx-B")       # B — có RAG
            gr.HTML("")                                          # C — không có RAG
            ctx_outs["D"] = gr.HTML("", elem_id="ctx-D")       # D — có RAG

        # Input
        with gr.Row(elem_id="question-row"):
            with gr.Column(scale=6):
                question_box = gr.Textbox(
                    placeholder="Nhập câu hỏi về quy chế TDTU… (Enter để gửi, Shift+Enter xuống dòng)",
                    label="", show_label=False, lines=2, max_lines=5,
                )
            with gr.Column(scale=1, min_width=130):
                send_btn  = gr.Button("Gửi ▶", variant="primary", elem_id="send-btn")
                clear_btn = gr.Button("🗑 Xóa", variant="secondary")

        # Suggest
        gr.HTML('<p style="font-size:11px;color:#bbb;margin:7px 0 3px;">💡 Câu hỏi gợi ý:</p>')
        with gr.Row():
            sq_btns = [gr.Button(q, size="sm", elem_classes="sq-btn") for q in SUGGESTED_QUESTIONS]

        # Wiring
        submit_in  = [question_box, panels["A"], panels["B"], panels["C"], panels["D"], cb_A, cb_B, cb_C, cb_D]
        submit_out = [panels["A"], panels["B"], panels["C"], panels["D"], ctx_outs["B"], ctx_outs["D"], latency_out]

        send_btn.click(fn=respond, inputs=submit_in, outputs=submit_out)
        question_box.submit(fn=respond, inputs=submit_in, outputs=submit_out)
        clear_btn.click(fn=clear_all, outputs=submit_out)

        for sq_btn, q in zip(sq_btns, SUGGESTED_QUESTIONS):
            sq_btn.click(fn=lambda x=q: x, outputs=question_box)

    return demo


# LAUNCH
if __name__ == "__main__" or "get_ipython" in dir():
    demo = build_demo()
    demo.launch(
        share=True,
        debug=False,
        show_error=True,
        server_name="0.0.0.0",
        server_port=7860,
        quiet=True,
    )