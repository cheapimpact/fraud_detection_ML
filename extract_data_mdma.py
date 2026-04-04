import sys
import os
import json
import fitz  # PyMuPDF
import re
import pandas as pd
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QLabel, QLineEdit, QFileDialog,
                             QListWidget, QMessageBox, QScrollArea)
from PyQt6.QtGui import QPixmap, QImage, QShortcut, QKeySequence
from PyQt6.QtCore import Qt, QEvent


class MDMAExtractor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MD&A Sentiment Analyzer (INSET Weighting)")
        self.base_folder = ""
        self.folders_to_process = []
        self.current_folder_index = 0
        self.current_doc = None
        self.current_page_num = 0
        self.current_pdf_path = ""

        script_dir = os.path.dirname(os.path.abspath(__file__))
        inset_path = os.path.join(script_dir, 'InSet')
        self.inset_dict = self.setup_inset(inset_path)
        self.init_ui()

    def setup_inset(self, folder_path):
        try:
            neg_path = os.path.join(folder_path, 'negative.tsv')
            pos_path = os.path.join(folder_path, 'positive.tsv')
            if not os.path.exists(neg_path) or not os.path.exists(pos_path):
                return {}
            df_neg = pd.read_csv(neg_path, sep='\t', header=None, names=['word', 'weight'])
            df_pos = pd.read_csv(pos_path, sep='\t', header=None, names=['word', 'weight'])
            
            # Memastikan kata-kata menjadi string dan huruf kecil semua tanpa spasi ekstra
            df_neg['word'] = df_neg['word'].astype(str).str.strip().str.lower()
            df_pos['word'] = df_pos['word'].astype(str).str.strip().str.lower()
            
            inset_dict = dict(zip(df_neg['word'], pd.to_numeric(df_neg['weight'], errors='coerce')))
            inset_dict.update(dict(zip(df_pos['word'], pd.to_numeric(df_pos['weight'], errors='coerce'))))
            return {k: v for k, v in inset_dict.items() if pd.notnull(v)}
        except:
            return {}

    def init_ui(self):
        main_layout = QHBoxLayout()

        # --- LEFT PANEL ---
        left_panel = QVBoxLayout()
        self.btn_select = QPushButton("1. Pilih Folder Utama")
        self.btn_select.setFixedHeight(40)
        self.btn_select.clicked.connect(self.scan_folder)
        left_panel.addWidget(self.btn_select)

        self.file_list = QListWidget()
        self.file_list.currentItemChanged.connect(self.on_list_item_changed)
        left_panel.addWidget(QLabel("Pilih File PDF:"))
        left_panel.addWidget(self.file_list)
        main_layout.addLayout(left_panel, 1)

        # --- CENTER PANEL (PDF VIEW) ---
        center_panel = QVBoxLayout()

        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("<")
        self.btn_prev.setFixedWidth(40)
        self.btn_prev.clicked.connect(self.prev_page)

        self.input_jump_page = QLineEdit()
        self.input_jump_page.setPlaceholderText("Hal...")
        self.input_jump_page.setFixedWidth(60)
        self.input_jump_page.returnPressed.connect(self.jump_to_page)

        self.lbl_page_count = QLabel("Halaman: 0 / 0")
        self.lbl_page_count.setStyleSheet("font-weight: bold;")

        self.btn_next = QPushButton(">")
        self.btn_next.setFixedWidth(40)
        self.btn_next.clicked.connect(self.next_page)

        nav_layout.addWidget(self.btn_prev)
        nav_layout.addStretch()
        nav_layout.addWidget(QLabel("Ke Halaman:"))
        nav_layout.addWidget(self.input_jump_page)
        nav_layout.addWidget(self.lbl_page_count)
        nav_layout.addStretch()
        nav_layout.addWidget(self.btn_next)
        center_panel.addLayout(nav_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("background-color: #525659;")
        self.pdf_label = QLabel("Preview PDF")
        self.pdf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pdf_label.setStyleSheet("background-color: white;")
        self.pdf_label.installEventFilter(self)
        self.is_zoomed = False
        self.scroll_area.setWidget(self.pdf_label)
        center_panel.addWidget(self.scroll_area)
        main_layout.addLayout(center_panel, 3)

        # --- RIGHT PANEL ---
        right_panel = QVBoxLayout()
        self.input_ticker = QLineEdit()
        right_panel.addWidget(QLabel("Ticker:"))
        right_panel.addWidget(self.input_ticker)

        self.input_start = QLineEdit()
        right_panel.addWidget(QLabel("Start Page (MD&A):"))
        right_panel.addWidget(self.input_start)

        self.input_end = QLineEdit()
        right_panel.addWidget(QLabel("End Page (MD&A):"))
        right_panel.addWidget(self.input_end)

        self.input_offside = QLineEdit()
        self.input_offside.setPlaceholderText("0")
        right_panel.addWidget(QLabel("Offside (offset, e.g. 2 or -2):"))
        right_panel.addWidget(self.input_offside)

        right_panel.addStretch()

        # Tombol Normal Generate
        self.btn_generate = QPushButton("GENERATE & ANALYZE")
        self.btn_generate.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; height: 50px;")
        self.btn_generate.clicked.connect(self.process_and_save)
        right_panel.addWidget(self.btn_generate)

        # TOMBOL BARU: NO MD&A
        self.btn_no_mdma = QPushButton("TIDAK ADA MD&A (N/A)")
        self.btn_no_mdma.setStyleSheet("background-color: #c62828; color: white; font-weight: bold; height: 40px;")
        self.btn_no_mdma.clicked.connect(self.skip_with_na)
        right_panel.addWidget(self.btn_no_mdma)

        main_layout.addLayout(right_panel, 1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        self.resize(1280, 850)

        # Global Shortcuts
        QShortcut(QKeySequence(Qt.Key.Key_Left), self).activated.connect(self.prev_page)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self).activated.connect(self.next_page)
        QShortcut(QKeySequence(Qt.Key.Key_Up), self).activated.connect(self.prev_pdf)
        QShortcut(QKeySequence(Qt.Key.Key_Down), self).activated.connect(self.next_pdf)
        QShortcut(QKeySequence(Qt.Key.Key_P), self).activated.connect(self.process_and_save)
        QShortcut(QKeySequence(Qt.Key.Key_Q), self).activated.connect(self.skip_with_na)

    def jump_to_page(self):
        if self.current_doc:
            try:
                page_num = int(self.input_jump_page.text()) - 1
                if 0 <= page_num < self.current_doc.page_count:
                    self.current_page_num = page_num
                    self.update_pdf_display()
                else:
                    QMessageBox.warning(self, "Halaman", "Halaman tidak ditemukan!")
            except ValueError:
                pass
            self.input_jump_page.clear()

    def update_pdf_display(self):
        if not self.current_doc: return
        self.is_zoomed = False
        page = self.current_doc.load_page(self.current_page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        
        v_width = self.scroll_area.viewport().width()
        v_height = self.scroll_area.viewport().height()
        
        scaled_pixmap = pixmap.scaled(
            max(10, v_width - 20),
            max(10, v_height),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.pdf_label.setPixmap(scaled_pixmap)
        self.lbl_page_count.setText(f"Halaman: {self.current_page_num + 1} / {self.current_doc.page_count}")

    def eventFilter(self, source, event):
        if source is self.pdf_label and event.type() == QEvent.Type.MouseButtonPress:
            self.toggle_zoom(event.pos())
            return True
        return super().eventFilter(source, event)

    def toggle_zoom(self, pos):
        if not self.current_doc: return
        page = self.current_doc.load_page(self.current_page_num)

        if not self.is_zoomed:
            if not self.pdf_label.pixmap(): return
            
            label_w = self.pdf_label.width()
            label_h = self.pdf_label.height()
            pix_w = self.pdf_label.pixmap().width()
            pix_h = self.pdf_label.pixmap().height()
            
            margin_x = (label_w - pix_w) / 2
            margin_y = (label_h - pix_h) / 2
            
            click_x = pos.x() - margin_x
            click_y = pos.y() - margin_y
            
            if click_x < 0 or click_y < 0 or click_x > pix_w or click_y > pix_h:
                return
                
            prop_x = click_x / pix_w
            prop_y = click_y / pix_h
            
            pix = page.get_pixmap(matrix=fitz.Matrix(5, 5))
            img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format.Format_RGB888)
            zoomed_pixmap = QPixmap.fromImage(img)
            
            self.pdf_label.setPixmap(zoomed_pixmap)
            self.pdf_label.resize(zoomed_pixmap.size())
            self.is_zoomed = True
            
            # Allow UI to process the new widget size before scrolling
            QApplication.processEvents()
            
            target_x = prop_x * zoomed_pixmap.width()
            target_y = prop_y * zoomed_pixmap.height()
            
            self.scroll_area.horizontalScrollBar().setValue(int(target_x - self.scroll_area.viewport().width() / 2))
            self.scroll_area.verticalScrollBar().setValue(int(target_y - self.scroll_area.viewport().height() / 2))
            
        else:
            self.is_zoomed = False
            self.update_pdf_display()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_doc and not getattr(self, "is_zoomed", False):
            self.update_pdf_display()

    def get_current_file_index(self):
        if not self.current_pdf_path:
            return -1
        current_name = os.path.basename(self.current_pdf_path)
        for i in range(self.file_list.count()):
            if self.file_list.item(i).text() == current_name:
                return i
        return -1

    def next_pdf(self):
        idx = self.get_current_file_index()
        if idx != -1 and idx < self.file_list.count() - 1:
            self.file_list.setCurrentRow(idx + 1)

    def prev_pdf(self):
        idx = self.get_current_file_index()
        if idx > 0:
            self.file_list.setCurrentRow(idx - 1)

    def scan_folder(self):
        self.base_folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if self.base_folder:
            all_subfolders = [f.path for f in os.scandir(self.base_folder) if f.is_dir()]
            self.folders_to_process = [f for f in all_subfolders if
                                       not os.path.exists(os.path.join(f, "analysis_result.json"))]
            self.current_folder_index = 0
            self.load_folder_context()

    def load_folder_context(self):
        if self.current_folder_index < len(self.folders_to_process):
            folder_path = self.folders_to_process[self.current_folder_index]
            self.input_ticker.setText(os.path.basename(folder_path))
            self.file_list.clear()
            pdfs = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
            self.file_list.addItems(pdfs)
            if pdfs:
                self.file_list.setCurrentRow(0)
        else:
            QMessageBox.information(self, "Selesai", "Semua folder telah diproses!")
            self.pdf_label.setText("Selesai")

    def load_pdf(self, path):
        self.current_pdf_path = path
        self.current_doc = fitz.open(path)
        self.current_page_num = 0
        self.update_pdf_display()

    def on_list_item_changed(self, current, previous):
        if current:
            folder_path = self.folders_to_process[self.current_folder_index]
            self.load_pdf(os.path.join(folder_path, current.text()))

    def next_page(self):
        if self.current_doc and self.current_page_num < self.current_doc.page_count - 1:
            self.current_page_num += 1
            self.update_pdf_display()

    def prev_page(self):
        if self.current_doc and self.current_page_num > 0:
            self.current_page_num -= 1
            self.update_pdf_display()

    def skip_with_na(self):
        """Fungsi khusus untuk menandai laporan tanpa MD&A"""
        if not self.current_doc: return
        confirm = QMessageBox.question(self, "Konfirmasi", "Tandai laporan ini tidak memiliki MD&A?",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if confirm == QMessageBox.StandardButton.Yes:
            self.save_result(is_na=True)

    def process_and_save(self):
        if not self.current_doc: return
        try:
            start_txt = self.input_start.text()
            end_txt = self.input_end.text()
            if not start_txt or not end_txt:
                raise ValueError("Isi Start dan End Page!")
            self.save_result(is_na=False)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def save_result(self, is_na=False):
        folder_path = self.folders_to_process[self.current_folder_index]
        ticker = self.input_ticker.text()

        if is_na:
            output = {
                "target_file": os.path.basename(self.current_pdf_path),
                "ticker": ticker,
                "mdma_start": "N/A",
                "mdma_end": "N/A",
                "Positive_Sum": 0,
                "Negative_Sum": 0,
                "Total_Matched_Words": 0,
                "Total_Word": 0,
                "processed_at": datetime.now().strftime("%Y-%m-%d"),
                "status": "No MD&A Found"
            }
        else:
            try:
                offside_txt = self.input_offside.text().strip()
                offside = int(offside_txt) if offside_txt else 0
            except ValueError:
                offside = 0

            start = max(0, int(self.input_start.text()) - 1 + offside)
            end = int(self.input_end.text()) + offside
            raw_text = ""
            for i in range(start, min(end, self.current_doc.page_count)):
                raw_text += self.current_doc.load_page(i).get_text()

            words = re.findall(r'\b\w+\b', raw_text.lower())
            pos_sum = 0.0
            neg_sum = 0.0
            matched_count = 0 
            for word in words:
                if word in self.inset_dict:
                    weight = self.inset_dict[word]  
                    if weight > 0:
                        pos_sum += weight
                    else:
                        neg_sum += abs(weight)
                    matched_count += 1

            with open(os.path.join(folder_path, f"raw_text_{ticker}.txt"), "w", encoding="utf-8") as f:
                f.write(raw_text)

            output = {
                "target_file": os.path.basename(self.current_pdf_path),
                "ticker": ticker,
                "mdma_start": str(start + 1),
                "mdma_end": str(end),
                "Positive_Sum": round(pos_sum, 2),
                "Negative_Sum": round(neg_sum, 2),
                "Total_Matched_Words": matched_count,
                "Total_Word": len(words),
                "processed_at": datetime.now().strftime("%Y-%m-%d"),
                "status": "Success"
            }

        with open(os.path.join(folder_path, "analysis_result.json"), "w") as f:
            json.dump(output, f, indent=4)

        # Reset UI untuk folder berikutnya
        self.input_start.clear()
        self.input_end.clear()
        self.input_offside.clear()
        self.current_folder_index += 1
        self.load_folder_context()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MDMAExtractor()
    window.show()
    sys.exit(app.exec())