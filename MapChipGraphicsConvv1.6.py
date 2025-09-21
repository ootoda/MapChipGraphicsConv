import cv2
import numpy as np
from PIL import Image
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import math
import gc
import threading
import tempfile
import uuid

def get_unique_filename(filepath):
    """
    ファイルが既に存在する場合、重複しない名前を生成する
    例: image.png -> image(1).png -> image(2).png ...
    """
    if not os.path.exists(filepath):
        return filepath
    
    # ファイルパスを分解
    directory = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)
    
    counter = 1
    while True:
        # 新しいファイル名を生成
        new_filename = f"{name}({counter}){ext}"
        new_filepath = os.path.join(directory, new_filename)
        
        if not os.path.exists(new_filepath):
            return new_filepath
        
        counter += 1
        
        # 無限ループ防止（念のため）
        if counter > 9999:
            raise Exception("適切なファイル名を生成できませんでした")

class PixelArtConverter:
    def __init__(self):
        self.base_patterns = []
        self.pattern_width = 16
        self.pattern_height = 16
        
    def clear_patterns(self):
        """メモリリークを防ぐためにパターンをクリア"""
        self.base_patterns.clear()
        gc.collect()  # ガベージコレクションを強制実行
        
    def load_base_patterns(self, base_image_path, pattern_width=16, pattern_height=16):
        """
        ベース画像からドット絵パターンを読み込む
        """
        # 既存のパターンをクリア
        self.clear_patterns()
        
        self.pattern_width = pattern_width
        self.pattern_height = pattern_height
        
        try:
            # ファイルの存在確認
            if not os.path.exists(base_image_path):
                raise ValueError(f"ファイルが見つかりません: {base_image_path}")
            
            # ベース画像を読み込み（まずPILで試す）
            base_image = None
            try:
                pil_image = Image.open(base_image_path)
                base_image = np.array(pil_image.convert('RGB'))
                print(f"PILで画像を読み込み成功: {base_image.shape}")
            except Exception as pil_error:
                print(f"PIL読み込みエラー: {pil_error}")
                # OpenCVで再試行
                base_image = cv2.imread(base_image_path)
                if base_image is None:
                    raise ValueError(f"画像を読み込めませんでした。サポートされていない形式の可能性があります: {base_image_path}")
                base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
                print(f"OpenCVで画像を読み込み成功: {base_image.shape}")
            
            height, width = base_image.shape[:2]
            print(f"画像サイズ: {width} x {height}")
            print(f"パターンサイズ: {pattern_width} x {pattern_height}")
            
            # パターン数を計算
            patterns_per_row = width // pattern_width
            patterns_per_col = height // pattern_height
            
            print(f"1行あたりのパターン数: {patterns_per_row}")
            print(f"列数: {patterns_per_col}")
            
            if patterns_per_row == 0 or patterns_per_col == 0:
                raise ValueError(f"画像サイズ({width}x{height})がパターンサイズ({pattern_width}x{pattern_height})より小さいです")
            
            # パターンを切り出し
            total_patterns = patterns_per_row * patterns_per_col
            processed = 0
            
            for row in range(patterns_per_col):
                for col in range(patterns_per_row):
                    y = row * pattern_height
                    x = col * pattern_width
                    pattern = base_image[y:y+pattern_height, x:x+pattern_width]
                    
                    # パターンのサイズチェック
                    if pattern.shape[:2] != (pattern_height, pattern_width):
                        print(f"警告: パターン({row}, {col})のサイズが不正: {pattern.shape}")
                        continue
                    
                    # パターンをコピーして追加（参照ではなく実際のコピー）
                    self.base_patterns.append(pattern.copy())
                    processed += 1
                    
                    # メモリ使用量を定期的にチェック
                    if processed % 100 == 0:
                        gc.collect()
            
            if len(self.base_patterns) == 0:
                raise ValueError("有効なパターンが見つかりませんでした")
            
            print(f"読み込んだパターン数: {len(self.base_patterns)}")
            
            # 元の画像データを削除してメモリを解放
            del base_image
            gc.collect()
            
            return True
            
        except Exception as e:
            print(f"ベース画像の読み込みエラー: {e}")
            return False, str(e)
    
    def calculate_color_similarity(self, img1, img2):
        """
        2つの画像の色の類似度を計算（HSV色空間での差を使用）
        """
        try:
            # 画像をHSV色空間に変換
            img1_hsv = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
            img2_hsv = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
            
            # 各画像の平均色を計算 (HSV)
            avg_hsv1 = np.mean(img1_hsv.reshape(-1, 3), axis=0)
            avg_hsv2 = np.mean(img2_hsv.reshape(-1, 3), axis=0)
            
            # HSV成分ごとの差を計算
            diff_h = abs(avg_hsv1[0] - avg_hsv2[0])
            # 色相(H)は0-179の範囲。円環状なので、差が180度を超える場合は小さい方を使う
            if diff_h > 90:
                diff_h = 180 - diff_h
            
            diff_s = abs(avg_hsv1[1] - avg_hsv2[1])
            diff_v = abs(avg_hsv1[2] - avg_hsv2[2])
            
            # 類似度を計算（各成分の差の重み付け和）
            similarity = (diff_h / 180.0) + (diff_s / 255.0) + (diff_v / 255.0)
            
            return similarity
        except Exception as e:
            print(f"色類似度計算エラー: {e}")
            return float('inf')  # エラーの場合は最大値を返す
    
    def find_best_pattern(self, target_region):
        """
        対象領域に最も近いパターンを見つける
        """
        if not self.base_patterns:
            return None
            
        min_distance = float('inf')
        best_pattern_idx = 0
        
        try:
            # 対象領域をパターンサイズにリサイズ
            target_resized = cv2.resize(target_region, (self.pattern_width, self.pattern_height))
            
            for i, pattern in enumerate(self.base_patterns):
                distance = self.calculate_color_similarity(target_resized, pattern)
                
                if distance < min_distance:
                    min_distance = distance
                    best_pattern_idx = i
            
            return self.base_patterns[best_pattern_idx].copy()
        except Exception as e:
            print(f"最適パターン検索エラー: {e}")
            return self.base_patterns[0].copy() if self.base_patterns else None
    
    def convert_image(self, input_image_path, output_image_path=None, progress_callback=None):
        """
        入力画像をドット絵に変換
        """
        if not self.base_patterns:
            return None, "ベースパターンが読み込まれていません"
        
        try:
            # ファイルの存在確認
            if not os.path.exists(input_image_path):
                raise ValueError(f"入力ファイルが見つかりません: {input_image_path}")
            
            # 入力画像を読み込み
            input_image = None
            try:
                pil_image = Image.open(input_image_path)
                input_image = np.array(pil_image.convert('RGB'))
                print(f"入力画像読み込み成功: {input_image.shape}")
            except Exception as pil_error:
                print(f"PIL読み込みエラー: {pil_error}")
                input_image = cv2.imread(input_image_path)
                if input_image is None:
                    raise ValueError(f"入力画像を読み込めませんでした: {input_image_path}")
                input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                print(f"入力画像読み込み成功: {input_image.shape}")
            
            height, width = input_image.shape[:2]
            
            # 出力画像のサイズを計算
            blocks_x = width // self.pattern_width
            blocks_y = height // self.pattern_height
            output_width = blocks_x * self.pattern_width
            output_height = blocks_y * self.pattern_height
            
            if blocks_x == 0 or blocks_y == 0:
                raise ValueError(f"入力画像({width}x{height})がパターンサイズ({self.pattern_width}x{self.pattern_height})より小さいです")
            
            # 出力画像を初期化
            output_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            
            total_blocks = blocks_x * blocks_y
            processed_blocks = 0
            
            print(f"変換中... ({blocks_x} x {blocks_y} ブロック)")
            
            # 各ブロックを処理
            for row in range(blocks_y):
                for col in range(blocks_x):
                    # 入力画像から対象領域を切り出し
                    y1 = row * self.pattern_height
                    x1 = col * self.pattern_width
                    y2 = y1 + self.pattern_height
                    x2 = x1 + self.pattern_width
                    
                    target_region = input_image[y1:y2, x1:x2]
                    
                    # 最適なパターンを見つける
                    best_pattern = self.find_best_pattern(target_region)
                    
                    if best_pattern is not None:
                        # 出力画像に配置
                        output_image[y1:y2, x1:x2] = best_pattern
                    
                    processed_blocks += 1
                    
                    # プログレス更新
                    if progress_callback and processed_blocks % 10 == 0:
                        progress = (processed_blocks / total_blocks) * 100
                        progress_callback(progress)
                    
                    # 定期的にガベージコレクション
                    if processed_blocks % 100 == 0:
                        gc.collect()
            
            # 結果を保存
            if output_image_path:
                # 重複しないファイル名を取得
                unique_output_path = get_unique_filename(output_image_path)
                
                output_pil = Image.fromarray(output_image)
                output_pil.save(unique_output_path)
                print(f"変換完了: {unique_output_path}")
                
                # 実際に使用されたファイル名を返す
                return output_image, unique_output_path
            
            # 入力画像データを削除してメモリを解放
            del input_image
            gc.collect()
            
            return output_image, None
            
        except Exception as e:
            error_msg = f"画像変換エラー: {e}"
            print(error_msg)
            return None, error_msg

class PixelArtGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MapChip Graphics conv")
        self.root.geometry("700x700")
        
        self.converter = PixelArtConverter()
        self.is_converting = False  # 変換中フラグ
        self.conversion_thread = None
        self.setup_gui()
    
    def setup_gui(self):
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # タイトル
        title_label = ttk.Label(main_frame, text="MapChip Graphics conv", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=4, pady=(0, 20))
        
        # パターンサイズ選択（縦・横別々）
        ttk.Label(main_frame, text="パターンサイズ:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky=tk.W, pady=5, padx=5)
        
        # 横幅
        ttk.Label(main_frame, text="横:").grid(row=1, column=1, sticky=tk.W, pady=5, padx=(10, 5))
        self.pattern_width_var = tk.StringVar(value="16")
        width_combo = ttk.Combobox(main_frame, textvariable=self.pattern_width_var,
                                 values=["8", "16", "24", "32", "48", "64"], state="readonly", width=8)
        width_combo.grid(row=1, column=2, sticky=tk.W, padx=(0, 10), pady=5)
        
        # 縦幅
        ttk.Label(main_frame, text="縦:").grid(row=1, column=3, sticky=tk.W, pady=5, padx=(10, 5))
        self.pattern_height_var = tk.StringVar(value="16")
        height_combo = ttk.Combobox(main_frame, textvariable=self.pattern_height_var,
                                  values=["8", "16", "24", "32", "48", "64"], state="readonly", width=8)
        height_combo.grid(row=1, column=4, sticky=tk.W, padx=(0, 10), pady=5)
        
        # ベース画像選択
        ttk.Label(main_frame, text="MAP Pattern:").grid(row=2, column=0, sticky=tk.W, pady=5, padx=5)
        self.base_path_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.base_path_var, width=40).grid(row=2, column=1, columnspan=3, sticky=(tk.W, tk.E), padx=(0, 5), pady=5)
        ttk.Button(main_frame, text="参照", command=self.select_base_image).grid(row=2, column=4, pady=5)
        
        # 入力画像選択
        ttk.Label(main_frame, text="変換する画像:").grid(row=3, column=0, sticky=tk.W, pady=5, padx=5)
        self.input_path_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.input_path_var, width=40).grid(row=3, column=1, columnspan=3, sticky=(tk.W, tk.E), padx=(0, 5), pady=5)
        ttk.Button(main_frame, text="参照", command=self.select_input_image).grid(row=3, column=4, pady=5)
        
        # 出力先選択
        ttk.Label(main_frame, text="出力先:").grid(row=4, column=0, sticky=tk.W, pady=5, padx=5)
        self.output_path_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.output_path_var, width=40).grid(row=4, column=1, columnspan=3, sticky=(tk.W, tk.E), padx=(0, 5), pady=5)
        ttk.Button(main_frame, text="参照", command=self.select_output_path).grid(row=4, column=4, pady=5)
        
        # --- 彩度・コントラスト調整機能 ---
        ttk.Label(main_frame, text="画像調整:", font=("Arial", 12, "bold")).grid(row=5, column=0, columnspan=5, sticky=tk.W, pady=(15, 5), padx=5)

        # 彩度スライダー（10刻みで移動）
        saturation_frame = ttk.Frame(main_frame)
        saturation_frame.grid(row=6, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        ttk.Label(saturation_frame, text="彩度:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.saturation_var = tk.DoubleVar(value=100)  # 100%が元の彩度（1.0に相当）
        self.saturation_label = ttk.Label(saturation_frame, text="100%")
        self.saturation_label.grid(row=0, column=1, padx=(0, 10))
        
        self.saturation_slider = ttk.Scale(saturation_frame, from_=0, to=200, orient=tk.HORIZONTAL, 
                                         variable=self.saturation_var, command=self.update_saturation_display)
        self.saturation_slider.grid(row=0, column=2, sticky=(tk.W, tk.E), padx=(0, 10))
        
        # 彩度調整ボタン
        ttk.Button(saturation_frame, text="-10", command=lambda: self.adjust_slider(self.saturation_var, -10, 0, 200)).grid(row=0, column=3, padx=2)
        ttk.Button(saturation_frame, text="+10", command=lambda: self.adjust_slider(self.saturation_var, 10, 0, 200)).grid(row=0, column=4, padx=2)
        ttk.Button(saturation_frame, text="リセット", command=lambda: self.reset_slider(self.saturation_var, 100)).grid(row=0, column=5, padx=2)
        
        # コントラストスライダー（10刻みで移動）
        contrast_frame = ttk.Frame(main_frame)
        contrast_frame.grid(row=7, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        ttk.Label(contrast_frame, text="コントラスト:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.contrast_var = tk.DoubleVar(value=100)  # 100%が元のコントラスト（1.0に相当）
        self.contrast_label = ttk.Label(contrast_frame, text="100%")
        self.contrast_label.grid(row=0, column=1, padx=(0, 10))
        
        self.contrast_slider = ttk.Scale(contrast_frame, from_=0, to=200, orient=tk.HORIZONTAL, 
                                       variable=self.contrast_var, command=self.update_contrast_display)
        self.contrast_slider.grid(row=0, column=2, sticky=(tk.W, tk.E), padx=(0, 10))
        
        # コントラスト調整ボタン
        ttk.Button(contrast_frame, text="-10", command=lambda: self.adjust_slider(self.contrast_var, -10, 0, 200)).grid(row=0, column=3, padx=2)
        ttk.Button(contrast_frame, text="+10", command=lambda: self.adjust_slider(self.contrast_var, 10, 0, 200)).grid(row=0, column=4, padx=2)
        ttk.Button(contrast_frame, text="リセット", command=lambda: self.reset_slider(self.contrast_var, 100)).grid(row=0, column=5, padx=2)
        
        # 明るさスライダー（10刻みで移動）
        brightness_frame = ttk.Frame(main_frame)
        brightness_frame.grid(row=8, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        ttk.Label(brightness_frame, text="明るさ:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.brightness_var = tk.DoubleVar(value=100)  # 100%が元の明るさ（1.0に相当）
        self.brightness_label = ttk.Label(brightness_frame, text="100%")
        self.brightness_label.grid(row=0, column=1, padx=(0, 10))
        
        self.brightness_slider = ttk.Scale(brightness_frame, from_=0, to=200, orient=tk.HORIZONTAL, 
                                         variable=self.brightness_var, command=self.update_brightness_display)
        self.brightness_slider.grid(row=0, column=2, sticky=(tk.W, tk.E), padx=(0, 10))
        
        # 明るさ調整ボタン
        ttk.Button(brightness_frame, text="-10", command=lambda: self.adjust_slider(self.brightness_var, -10, 0, 200)).grid(row=0, column=3, padx=2)
        ttk.Button(brightness_frame, text="+10", command=lambda: self.adjust_slider(self.brightness_var, 10, 0, 200)).grid(row=0, column=4, padx=2)
        ttk.Button(brightness_frame, text="リセット", command=lambda: self.reset_slider(self.brightness_var, 100)).grid(row=0, column=5, padx=2)
        
        # 変換ボタン
        self.convert_btn = ttk.Button(main_frame, text="変換実行", command=self.convert_image,
                               style="Accent.TButton")
        self.convert_btn.grid(row=9, column=0, columnspan=5, pady=20)
        
        # キャンセルボタン（変換中のみ表示）
        self.cancel_btn = ttk.Button(main_frame, text="キャンセル", command=self.cancel_conversion, state='disabled')
        self.cancel_btn.grid(row=10, column=0, columnspan=5, pady=5)
        
        # プログレスバー
        self.progress = ttk.Progressbar(main_frame, mode='determinate', maximum=100)
        self.progress.grid(row=11, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=5)
        
        # ログエリア
        ttk.Label(main_frame, text="ログ:").grid(row=12, column=0, sticky=tk.W, pady=(10, 5), padx=5)
        
        log_frame = ttk.Frame(main_frame)
        log_frame.grid(row=13, column=0, columnspan=5, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # グリッドの重み設定
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.columnconfigure(3, weight=1)
        main_frame.rowconfigure(13, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        saturation_frame.columnconfigure(2, weight=1)
        contrast_frame.columnconfigure(2, weight=1)
        brightness_frame.columnconfigure(2, weight=1)

        # スタイル設定
        style = ttk.Style()
        style.configure("Accent.TButton", foreground="black", background="blue")

        # 画像読み込み後のプレビュー表示用
        self.input_image_preview = None

    def adjust_slider(self, var, delta, min_val, max_val):
        """スライダーを指定した値だけ調整"""
        current = var.get()
        new_val = max(min_val, min(max_val, current + delta))
        var.set(new_val)

    def reset_slider(self, var, default_val):
        """スライダーをデフォルト値にリセット"""
        var.set(default_val)

    def update_saturation_display(self, *args):
        """彩度表示を更新"""
        value = int(self.saturation_var.get())
        self.saturation_label.config(text=f"{value}%")

    def update_contrast_display(self, *args):
        """コントラスト表示を更新"""
        value = int(self.contrast_var.get())
        self.contrast_label.config(text=f"{value}%")

    def update_brightness_display(self, *args):
        """明るさ表示を更新"""
        value = int(self.brightness_var.get())
        self.brightness_label.config(text=f"{value}%")

    def log_message(self, message):
        """ログメッセージを表示"""
        def update_log():
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)
            self.root.update()
        
        # メインスレッドで実行
        if threading.current_thread() != threading.main_thread():
            self.root.after(0, update_log)
        else:
            update_log()
    
    def select_base_image(self):
        """ベース画像を選択"""
        if self.is_converting:
            messagebox.showwarning("警告", "変換中は設定を変更できません")
            return
            
        file_path = filedialog.askopenfilename(
            title="ベース画像を選択",
            filetypes=[("画像ファイル", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("すべて", "*.*")]
        )
        if file_path:
            self.base_path_var.set(file_path)
            self.log_message(f"ベース画像を設定: {file_path}")
    
    def select_input_image(self):
        """入力画像を選択"""
        if self.is_converting:
            messagebox.showwarning("警告", "変換中は設定を変更できません")
            return
            
        file_path = filedialog.askopenfilename(
            title="変換する画像を選択",
            filetypes=[("画像ファイル", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("すべて", "*.*")]
        )
        if file_path:
            self.input_path_var.set(file_path)
            self.log_message(f"入力画像を設定: {file_path}")
            # 画像を読み込み、調整前の状態を保存
            try:
                pil_image = Image.open(file_path)
                self.input_image_preview = np.array(pil_image.convert('RGB'))
                self.reset_adjustments()  # 新しい画像が選択されたら調整をリセット
            except Exception as e:
                self.log_message(f"入力画像の読み込みエラー: {e}")
                self.input_image_preview = None

    def select_output_path(self):
        """出力先を選択"""
        if self.is_converting:
            messagebox.showwarning("警告", "変換中は設定を変更できません")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="出力先を選択",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("すべて", "*.*")]
        )
        if file_path:
            self.output_path_var.set(file_path)
            self.log_message(f"出力先を設定: {file_path}")
    
    def apply_adjustments(self, image):
        """彩度、コントラスト、明るさを適用する"""
        if image is None:
            return None
            
        saturation_factor = self.saturation_var.get() / 100.0  # パーセンテージを小数に変換
        contrast_factor = self.contrast_var.get() / 100.0
        brightness_factor = self.brightness_var.get() / 100.0

        try:
            # HSV色空間に変換
            hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # 彩度調整 (Sチャンネル)
            hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255)
            
            # 明るさ調整 (Vチャンネル) - コントラスト調整の前に適用
            hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * brightness_factor, 0, 255)
            
            # コントラスト調整 (Vチャンネル) - 中央値(127.5)を基準に拡大/縮小
            hsv_image[:, :, 2] = np.clip((hsv_image[:, :, 2] - 127.5) * contrast_factor + 127.5, 0, 255)
            
            # RGBに戻す
            adjusted_image = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2RGB)
            
            return adjusted_image
        except Exception as e:
            self.log_message(f"画像調整エラー: {e}")
            return image  # エラーの場合は元の画像を返す

    def reset_adjustments(self):
        """調整スライダーを初期値に戻す"""
        self.saturation_var.set(100)
        self.contrast_var.set(100)
        self.brightness_var.set(100)

    def update_progress(self, value):
        """プログレスバーを更新"""
        def update():
            self.progress['value'] = value
            self.root.update_idletasks()
        
        # メインスレッドで実行
        if threading.current_thread() != threading.main_thread():
            self.root.after(0, update)
        else:
            update()

    def cancel_conversion(self):
        """変換をキャンセル"""
        self.is_converting = False
        self.log_message("変換をキャンセルしました")

    def convert_image_thread(self):
        """バックグラウンドで画像変換を実行"""
        try:
            # パターンサイズを取得
            pattern_width = int(self.pattern_width_var.get())
            pattern_height = int(self.pattern_height_var.get())
            
            # ベースパターンを読み込み
            self.log_message("ベースパターンを読み込み中...")
            self.log_message(f"ベース画像パス: {self.base_path_var.get()}")
            
            result = self.converter.load_base_patterns(self.base_path_var.get(), pattern_width, pattern_height)
            if isinstance(result, tuple):
                success, error_msg = result
                if not success:
                    raise Exception(f"ベースパターンの読み込みに失敗しました: {error_msg}")
            elif not result:
                raise Exception("ベースパターンの読み込みに失敗しました")
            
            if not self.is_converting:  # キャンセルチェック
                return
                
            self.log_message(f"パターンサイズ: {pattern_width}x{pattern_height}")
            self.log_message(f"読み込んだパターン数: {len(self.converter.base_patterns)}")
            
            # 画像変換を実行
            self.log_message("画像変換を開始...")
            self.log_message(f"入力画像パス: {self.input_path_var.get()}")
            self.log_message(f"彩度: {int(self.saturation_var.get())}%, コントラスト: {int(self.contrast_var.get())}%, 明るさ: {int(self.brightness_var.get())}%")

            # 入力画像に彩度・コントラスト調整を適用
            adjusted_input_image = self.apply_adjustments(self.input_image_preview)
            if adjusted_input_image is None:
                raise Exception("画像調整の適用に失敗しました。")

            # 一意の一時ファイル名を生成
            temp_input_path = os.path.join(tempfile.gettempdir(), f"temp_adjusted_input_{uuid.uuid4().hex}.png")
            
            try:
                # 調整後の画像パスを一時的に保存
                Image.fromarray(adjusted_input_image).save(temp_input_path)
                
                if not self.is_converting:  # キャンセルチェック
                    return

                # プログレス更新用コールバック
                def progress_callback(progress):
                    if self.is_converting:  # キャンセルされていない場合のみ更新
                        self.update_progress(progress)

                result = self.converter.convert_image(
                    temp_input_path,  # 調整済みの画像を渡す
                    self.output_path_var.get(),
                    progress_callback
                )
                
                # 結果を処理
                if isinstance(result, tuple) and len(result) == 2:
                    result_image, actual_output_path = result
                    
                    if result_image is not None and self.is_converting:
                        if actual_output_path and actual_output_path != self.output_path_var.get():
                            # ファイル名が変更された場合、ログに記録
                            self.log_message(f"ファイル名を変更しました: {os.path.basename(actual_output_path)}")
                            self.log_message(f"実際の出力先: {actual_output_path}")
                        self.log_message("変換が完了しました！")
                        # メインスレッドでメッセージボックスを表示
                        self.root.after(0, lambda: messagebox.showinfo("完了", "画像の変換が完了しました"))
                    elif not self.is_converting:
                        self.log_message("変換がキャンセルされました")
                    else:
                        raise Exception(actual_output_path or "画像変換に失敗しました")
                else:
                    # 古い形式の戻り値（後方互換性のため）
                    result_image, error_msg = result
                    if result_image is not None and self.is_converting:
                        self.log_message("変換が完了しました！")
                        self.root.after(0, lambda: messagebox.showinfo("完了", "画像の変換が完了しました"))
                    elif not self.is_converting:
                        self.log_message("変換がキャンセルされました")
                    else:
                        raise Exception(error_msg or "画像変換に失敗しました")
                    
            finally:
                # 一時ファイルを削除
                if os.path.exists(temp_input_path):
                    try:
                        os.remove(temp_input_path)
                    except Exception as cleanup_error:
                        self.log_message(f"一時ファイル削除エラー: {cleanup_error}")
                
        except Exception as e:
            self.log_message(f"エラー: {e}")
            # メインスレッドでエラーメッセージを表示
            self.root.after(0, lambda: messagebox.showerror("エラー", str(e)))
        
        finally:
            # UI状態をリセット
            def reset_ui():
                self.is_converting = False
                self.convert_btn.config(state='normal', text='変換実行')
                self.cancel_btn.config(state='disabled')
                self.progress['value'] = 0
                # メモリクリーンアップ
                gc.collect()
            
            self.root.after(0, reset_ui)

    def convert_image(self):
        """画像変換を実行"""
        # 変換中の場合は何もしない
        if self.is_converting:
            return
            
        # 入力チェック
        if not self.base_path_var.get():
            messagebox.showerror("エラー", "ベース画像を選択してください")
            return
        
        if not self.input_path_var.get():
            messagebox.showerror("エラー", "変換する画像を選択してください")
            return
        
        if not self.output_path_var.get():
            messagebox.showerror("エラー", "出力先を選択してください")
            return
        
        if self.input_image_preview is None:
            messagebox.showerror("エラー", "入力画像を読み込めませんでした。再度選択してください。")
            return

        # UI状態を変換中に設定
        self.is_converting = True
        self.convert_btn.config(state='disabled', text='変換中...')
        self.cancel_btn.config(state='normal')
        self.progress['value'] = 0
        self.log_text.delete(1.0, tk.END)
        
        # バックグラウンドスレッドで変換を実行
        self.conversion_thread = threading.Thread(target=self.convert_image_thread, daemon=True)
        self.conversion_thread.start()

def main():
    """メイン関数"""
    root = tk.Tk()
    app = PixelArtGUI(root)
    
    # アプリケーション終了時の処理
    def on_closing():
        if app.is_converting:
            if messagebox.askokcancel("確認", "変換処理が実行中です。終了しますか？"):
                app.is_converting = False
                if app.conversion_thread and app.conversion_thread.is_alive():
                    app.conversion_thread.join(timeout=1.0)  # 最大1秒待機
                root.destroy()
        else:
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()