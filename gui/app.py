# gui/app.py - 虚拟人脸识别 GUI
"""
图形界面：推理/检测、训练，支持模型选择与进度展示。
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import sys
import glob

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _scan_checkpoints(save_dir):
    """扫描 save_dir 下的 .pth 文件，返回 [(显示名, 绝对路径), ...]"""
    if not save_dir or not os.path.isdir(save_dir):
        return []
    paths = sorted(glob.glob(os.path.join(save_dir, "*.pth")), key=os.path.getmtime, reverse=True)
    return [(os.path.basename(p), p) for p in paths]


def _run_inference(config, video_path, checkpoint_path, callback):
    """后台执行推理，callback(success, label=None, confidence=None, error=None)"""
    try:
        from scripts.infer import infer
        label_str, confidence = infer(config, video_path, checkpoint_path=checkpoint_path)
        callback(success=True, label=label_str, confidence=confidence)
    except Exception as e:
        callback(success=False, error=str(e))


def _run_train(config, root, progress_callback, done_callback):
    """后台执行训练，每 epoch 调用 progress_callback，结束后调用 done_callback"""
    def _progress(epoch, total_epochs, train_loss, train_acc, val_loss, val_acc, last_checkpoint, save_dir):
        if progress_callback and root.winfo_exists():
            root.after(0, lambda: progress_callback(
                epoch, total_epochs, train_loss, train_acc, val_loss, val_acc,
                last_checkpoint, save_dir
            ))

    def _run():
        try:
            from scripts.train import train
            result = train(config, progress_callback=_progress)
            if done_callback and root.winfo_exists():
                root.after(0, lambda: done_callback(success=True, result=result))
        except Exception as e:
            if done_callback and root.winfo_exists():
                root.after(0, lambda err=str(e): done_callback(success=False, error=err))

    threading.Thread(target=_run, daemon=True).start()


def run_gui(config):
    root = tk.Tk()
    root.title("虚拟人脸识别 - Video Deepfake Detection")
    root.geometry("640x580")
    root.resizable(True, True)
    root.configure(bg="#1a1a2e")

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TFrame", background="#1a1a2e")
    style.configure("Title.TLabel", background="#1a1a2e", foreground="#e94560", font=("Microsoft YaHei UI", 16, "bold"))
    style.configure("TLabel", background="#1a1a2e", foreground="#eaeaea", font=("Microsoft YaHei UI", 10))
    style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=(10, 5))
    style.configure("Accent.TButton", background="#e94560", foreground="white", font=("Microsoft YaHei UI", 11, "bold"))
    style.map("Accent.TButton", background=[("active", "#c73e54")])
    style.configure("TNotebook", background="#1a1a2e")
    style.configure("TNotebook.Tab", background="#16213e", foreground="#eaeaea", padding=(12, 6))
    style.map("TNotebook.Tab", background=[("selected", "#e94560")])
    style.configure("Horizontal.TProgressbar", troughcolor="#16213e", background="#e94560", thickness=12)

    main = ttk.Frame(root, padding=20)
    main.pack(fill=tk.BOTH, expand=True)

    title_label = ttk.Label(main, text="虚拟人脸识别", style="Title.TLabel")
    title_label.pack(pady=(0, 4))
    subtitle = ttk.Label(main, text="Video Deepfake / 视频真伪检测", style="TLabel")
    subtitle.pack(pady=(0, 16))

    notebook = ttk.Notebook(main)
    notebook.pack(fill=tk.BOTH, expand=True)

    _save_dir = config.get("model", {}).get("save_dir", "models/checkpoints")
    save_dir = _save_dir if os.path.isabs(_save_dir) else os.path.normpath(os.path.join(_PROJECT_ROOT, _save_dir))
    checkpoints = _scan_checkpoints(save_dir)
    if not checkpoints and config.get("model", {}).get("checkpoint_path"):
        cp = config["model"]["checkpoint_path"]
        if os.path.isabs(cp):
            abs_cp = cp
        else:
            abs_cp = os.path.normpath(os.path.join(_PROJECT_ROOT, cp))
        if os.path.isfile(abs_cp):
            checkpoints = [(os.path.basename(abs_cp), abs_cp)]

    # ==================== 推理 Tab ====================
    infer_frame = ttk.Frame(notebook, padding=10)
    notebook.add(infer_frame, text="  推理 / 检测  ")

    # 模型选择
    model_row = ttk.Frame(infer_frame)
    model_row.pack(fill=tk.X, pady=(0, 8))
    ttk.Label(model_row, text="模型文件:", style="TLabel").pack(side=tk.LEFT, padx=(0, 8))
    model_var = tk.StringVar()
    model_combo = ttk.Combobox(model_row, textvariable=model_var, width=42, state="readonly", font=("Consolas", 10))
    model_combo.pack(side=tk.LEFT, padx=(0, 8), fill=tk.X, expand=True)
    if checkpoints:
        model_combo["values"] = [name for name, _ in checkpoints]
        model_combo.current(0)
    else:
        model_combo["values"] = ["(无可用模型)"]
        model_combo.current(0)

    def browse_model():
        p = filedialog.askopenfilename(title="选择模型权重", filetypes=[("PyTorch 权重", "*.pth"), ("所有", "*.*")])
        if p:
            model_var.set(os.path.basename(p))
            model_combo["values"] = list(model_combo["values"]) + [os.path.basename(p)]
            model_path_map[os.path.basename(p)] = p

    model_path_map = {name: path for name, path in checkpoints}
    ttk.Button(model_row, text="浏览...", command=browse_model).pack(side=tk.LEFT)

    def get_selected_checkpoint():
        name = model_var.get()
        if name and name != "(无可用模型)":
            return model_path_map.get(name) or os.path.join(save_dir, name)
        return None

    # 视频选择
    file_row = ttk.Frame(infer_frame)
    file_row.pack(fill=tk.X, pady=(0, 8))
    ttk.Label(file_row, text="视频文件:", style="TLabel").pack(side=tk.LEFT, padx=(0, 8))
    video_path_var = tk.StringVar()
    path_entry = ttk.Entry(file_row, textvariable=video_path_var, width=40, font=("Consolas", 10))
    path_entry.pack(side=tk.LEFT, padx=(0, 8), fill=tk.X, expand=True)

    def choose_video():
        p = filedialog.askopenfilename(title="选择视频", filetypes=[("视频", "*.mp4 *.avi *.mov *.mkv"), ("所有", "*.*")])
        if p:
            video_path_var.set(p)

    ttk.Button(file_row, text="浏览...", command=choose_video).pack(side=tk.LEFT)

    # 开始检测按钮（用 lambda 延迟绑定，避免 start_inference 未定义）
    ttk.Button(infer_frame, text="开始检测", style="Accent.TButton", command=lambda: start_inference()).pack(pady=(12, 12))

    # 推理结果区域
    infer_result_frame = ttk.Frame(infer_frame)
    infer_result_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

    # 推理结果高亮显示区（置顶）
    infer_result_banner = tk.Frame(infer_result_frame, bg="#16213e", height=56)
    infer_result_banner.pack(fill=tk.X, pady=(0, 8))
    infer_result_banner.pack_propagate(False)
    infer_label_var = tk.StringVar(value="")
    infer_conf_var = tk.StringVar(value="")
    ttk.Label(infer_result_banner, textvariable=infer_label_var, style="Title.TLabel", font=("Microsoft YaHei UI", 14)).pack(side=tk.LEFT, padx=(12, 8))
    ttk.Label(infer_result_banner, textvariable=infer_conf_var, style="TLabel").pack(side=tk.LEFT)

    infer_result_canvas = tk.Canvas(infer_result_frame, bg="#16213e", highlightthickness=0)
    infer_result_canvas.pack(fill=tk.BOTH, expand=True)

    infer_result_text = tk.Text(
        infer_result_canvas, bg="#16213e", fg="#eaeaea", font=("Consolas", 11),
        wrap=tk.WORD, state=tk.DISABLED, relief=tk.FLAT, padx=12, pady=12,
    )
    infer_result_canvas.create_window(0, 0, window=infer_result_text, anchor=tk.NW)

    infer_status_var = tk.StringVar(value="请选择模型和视频后点击「开始检测」")
    ttk.Label(infer_frame, textvariable=infer_status_var, style="TLabel").pack(pady=(8, 4))

    def update_infer_result(content, clear=True):
        infer_result_text.config(state=tk.NORMAL)
        if clear:
            infer_result_text.delete(1.0, tk.END)
        infer_result_text.insert(tk.END, content)
        infer_result_text.config(state=tk.DISABLED)
        infer_result_text.see(tk.END)

    def on_inference_done(success, label=None, confidence=None, error=None):
        if success:
            infer_status_var.set("识别完成")
            infer_label_var.set(f"结果: {label}")
            infer_conf_var.set(f"置信度: {confidence:.2%}" if confidence is not None else "")
            update_infer_result(f"【识别结果】\n类别: {label}\n置信度: {confidence:.2%}\n", clear=True)
        else:
            infer_status_var.set("识别失败")
            infer_label_var.set("")
            infer_conf_var.set("")
            update_infer_result(f"错误: {error}\n", clear=True)
            messagebox.showerror("识别失败", str(error))

    def start_inference():
        path = video_path_var.get().strip()
        if not path:
            messagebox.showwarning("提示", "请先选择视频文件")
            return
        if not os.path.isfile(path):
            messagebox.showerror("错误", f"文件不存在: {path}")
            return
        cp = get_selected_checkpoint()
        if not cp or not os.path.isfile(cp):
            messagebox.showerror("错误", "请选择有效的模型权重文件")
            return

        infer_status_var.set("正在加载模型并推理...")
        infer_label_var.set("")
        infer_conf_var.set("")
        update_infer_result("正在加载模型...\n正在读取视频...\n推理中...\n", clear=True)

        def run():
            cfg = dict(config)
            _run_inference(cfg, path, cp, on_inference_done)

        threading.Thread(target=run, daemon=True).start()

    # ==================== 训练 Tab ====================
    train_frame = ttk.Frame(notebook, padding=10)
    notebook.add(train_frame, text="  训练  ")

    # 模型保存位置输入
    train_save_row = ttk.Frame(train_frame)
    train_save_row.pack(fill=tk.X, pady=(0, 12))
    ttk.Label(train_save_row, text="模型保存位置:", style="TLabel").pack(side=tk.LEFT, padx=(0, 8))
    train_save_path_input_var = tk.StringVar(value=save_dir)
    train_save_path_entry = ttk.Entry(train_save_row, textvariable=train_save_path_input_var, width=45, font=("Consolas", 10))
    train_save_path_entry.pack(side=tk.LEFT, padx=(0, 8), fill=tk.X, expand=True)

    def browse_save_dir():
        d = filedialog.askdirectory(title="选择模型保存目录")
        if d:
            train_save_path_input_var.set(d)

    ttk.Button(train_save_row, text="浏览...", command=browse_save_dir).pack(side=tk.LEFT)

    train_status_var = tk.StringVar(value="配置数据路径后点击「开始训练」")
    ttk.Label(train_frame, textvariable=train_status_var, style="TLabel").pack(pady=(0, 8))

    train_progress_var = tk.DoubleVar(value=0)
    train_progress = ttk.Progressbar(train_frame, variable=train_progress_var, maximum=100, style="Horizontal.TProgressbar")
    train_progress.pack(fill=tk.X, pady=(0, 8))

    train_log = tk.Text(
        train_frame, bg="#16213e", fg="#eaeaea", font=("Consolas", 10),
        wrap=tk.WORD, state=tk.DISABLED, relief=tk.FLAT, padx=12, pady=12, height=12,
    )
    train_log.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

    train_result_frame = ttk.Frame(train_frame)
    train_result_frame.pack(fill=tk.X, pady=(0, 8))
    train_final_acc_var = tk.StringVar(value="")
    train_save_path_var = tk.StringVar(value="")
    ttk.Label(train_result_frame, text="最终验证准确率:", style="TLabel").pack(side=tk.LEFT, padx=(0, 8))
    ttk.Label(train_result_frame, textvariable=train_final_acc_var, style="Title.TLabel", font=("Microsoft YaHei UI", 12)).pack(side=tk.LEFT, padx=(0, 24))
    ttk.Label(train_result_frame, text="模型保存位置:", style="TLabel").pack(side=tk.LEFT, padx=(0, 8))
    ttk.Label(train_result_frame, textvariable=train_save_path_var, style="TLabel").pack(side=tk.LEFT)

    def append_train_log(msg):
        train_log.config(state=tk.NORMAL)
        train_log.insert(tk.END, msg + "\n")
        train_log.see(tk.END)
        train_log.config(state=tk.DISABLED)

    def on_train_progress(epoch, total_epochs, train_loss, train_acc, val_loss, val_acc, last_checkpoint, save_dir):
        pct = 100.0 * epoch / total_epochs if total_epochs else 0
        train_progress_var.set(pct)
        train_status_var.set(f"训练中... Epoch {epoch}/{total_epochs}")
        append_train_log(f"Epoch {epoch}/{total_epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | Val Loss: {val_loss:.4f} Acc: {val_acc:.2%}")
        if last_checkpoint:
            append_train_log(f"  -> 已保存: {last_checkpoint}")

    def on_train_done(success, result=None, error=None):
        train_progress_var.set(100)
        if success:
            train_status_var.set("训练完成")
            r = result or {}
            acc = r.get("final_val_acc", 0)
            save_dir = r.get("save_dir", "")
            last_cp = r.get("last_checkpoint", "")
            train_final_acc_var.set(f"{acc:.2%}")
            train_save_path_var.set(last_cp or save_dir or "-")
            append_train_log("\n========== 训练完成 ==========")
            append_train_log(f"最终验证准确率: {acc:.2%}")
            append_train_log(f"模型保存目录: {save_dir}")
            if last_cp:
                append_train_log(f"最新检查点: {last_cp}")
            messagebox.showinfo("训练完成", f"验证准确率: {acc:.2%}\n模型保存: {last_cp or save_dir}")
        else:
            train_status_var.set("训练失败")
            train_final_acc_var.set("-")
            train_save_path_var.set("-")
            append_train_log(f"\n错误: {error}")
            messagebox.showerror("训练失败", str(error))

    def start_train():
        train_progress_var.set(0)
        train_final_acc_var.set("")
        train_save_path_var.set("")
        train_log.config(state=tk.NORMAL)
        train_log.delete(1.0, tk.END)
        train_log.config(state=tk.DISABLED)
        train_status_var.set("正在准备数据...")
        save_path = train_save_path_input_var.get().strip()
        if not save_path:
            messagebox.showwarning("提示", "请填写模型保存位置")
            return
        cfg = dict(config)
        cfg.setdefault("model", {})["save_dir"] = save_path
        cfg.setdefault("training", {})["num_workers"] = 0

        def run():
            _run_train(cfg, root, on_train_progress, on_train_done)

        threading.Thread(target=run, daemon=True).start()

    ttk.Button(train_frame, text="开始训练", style="Accent.TButton", command=start_train).pack(pady=(4, 0))

    # 训练说明
    ttk.Label(train_frame, text="训练数据路径见 config.yaml (train_dir, val_dir, train_labels, val_labels)", style="TLabel").pack(pady=(8, 0))

    root.mainloop()
