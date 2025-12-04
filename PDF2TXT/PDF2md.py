# -*- coding: utf-8 -*-
import os
import time
import traceback
from multiprocessing import Process, Queue

from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod

# Writer that does not write image files
class NoImageWriter(FileBasedDataWriter):
    def write(self, path: str, data: bytes) -> None:
        ext = os.path.splitext(path)[1].lower()
        if ext in {'.jpg', '.jpeg', '.png', '.gif'}:
            return
        super().write(path, data)

input_root = r'path your pdf'
output_root = r'path your md'
log_position_path = os.path.join(output_root, 'log_position.txt')
error_log_path = os.path.join(output_root, 'error_log.txt')
TIMEOUT_SEC = 60  # Timeout in seconds

os.makedirs(output_root, exist_ok=True)

# List PDFs
all_pdfs = [
    f for f in os.listdir(input_root)
    if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(input_root, f))
]
all_pdfs.sort()
print(f"Detected {len(all_pdfs)} PDF files.")

START_IDX = 1
END_IDX = len(all_pdfs)

reader = FileBasedDataReader("")
n_total = len(all_pdfs)
start = max(START_IDX, 1)
end = min(END_IDX, n_total)

def _worker(fname: str, input_root: str, output_root: str, result_q: Queue):
    """
    Child process: process a single PDF and send the result or exception
    back to the main process via a queue.
    """
    t0 = time.time()
    try:
        pdf_path = os.path.join(input_root, fname)
        base_name = os.path.splitext(fname)[0]

        pdf_bytes = FileBasedDataReader("").read(pdf_path)
        if not pdf_bytes:
            raise RuntimeError("File read, but byte length is 0")

        ds = PymuDocDataset(pdf_bytes)
        is_ocr = (ds.classify() == SupportedPdfParseMethod.OCR)
        infer_res = ds.apply(doc_analyze, ocr=is_ocr)

        pipe = (infer_res.pipe_ocr_mode if is_ocr else infer_res.pipe_txt_mode)(NoImageWriter(output_root))
        md_writer = FileBasedDataWriter(output_root)
        pipe.dump_md(md_writer, f"{base_name}.md", "")

        cost = time.time() - t0
        result_q.put({
            "ok": True,
            "msg": f"[√] Processed: {fname} -> {base_name}.md (elapsed {cost:.1f}s)"
        })
    except Exception as e:
        tb = traceback.format_exc()
        result_q.put({"ok": False, "err": f"{fname} (error: {e})", "trace": tb})

if __name__ == "__main__":
    for idx in range(start, end + 1):
        fname = all_pdfs[idx - 1]
        print(f"[>] Processing ({idx}/{n_total}): {fname}", flush=True)

        q = Queue()
        p = Process(target=_worker, args=(fname, input_root, output_root, q))
        p.start()
        p.join(timeout=TIMEOUT_SEC)

        if p.is_alive():
            # Timeout: force kill the child process and log it
            p.terminate()
            p.join()
            print(f"[!] Skip `{fname}`, reason: timeout (>{TIMEOUT_SEC}s)", flush=True)
            try:
                with open(error_log_path, 'a', encoding='utf-8') as logf:
                    logf.write(f"{fname} (timeout)\n")
            except Exception as log_e:
                print(f"[!] Failed to write error log: {log_e}", flush=True)
        else:
            # Child process finished within the time limit; get result from queue
            try:
                res = q.get_nowait()
            except Exception:
                res = {"ok": False, "err": f"{fname} (unknown worker state)"}

            if res.get("ok"):
                print(res["msg"], flush=True)
            else:
                print(f"[!] Skip `{fname}`, reason: {res.get('err', 'unknown error')}", flush=True)
                try:
                    with open(error_log_path, 'a', encoding='utf-8') as logf:
                        logf.write(res.get("err", f"{fname} (error)") + "\n")
                except Exception as log_e:
                    print(f"[!] Failed to write error log: {log_e}", flush=True)
                # (Optional) Print traceback for debugging
                if "trace" in res:
                    print(res["trace"], flush=True)

        # Record current processing position in real time
        try:
            with open(log_position_path, 'w', encoding='utf-8') as posf:
                posf.write(str(idx))
        except Exception as log_e:
            print(f"[!] Failed to write progress log: {log_e}", flush=True)

    print(f"[ℹ] All MD files have been saved to: {output_root}")
    print(f"[ℹ] Failed entries have been recorded in: {error_log_path}")