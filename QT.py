import sys
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QLineEdit, QLabel, QFileDialog, QMessageBox,
                            QProgressBar, QGroupBox, QTabWidget, QSplitter)
from PyQt5.QtGui import QValidator, QPixmap
from PyQt5 import QtGui
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from ctypes import CDLL, c_char_p, c_int, c_double, POINTER, Structure, c_float, CFUNCTYPE
import tempfile
import shutil
import matplotlib.animation as animation
from matplotlib.patches import Circle, Arrow

# Load the Rust library
lib_path = os.path.join(os.path.dirname(__file__), 'target/release/libairquality_predictor.so')
rust_lib = CDLL(lib_path)

# Define C structures to match Rust
class PredictionResult(Structure):
    _fields_ = [
        ("predicted_class", c_int),
        ("probabilities", POINTER(c_double)),  # Changed to c_double to match Rust's f64
        ("probabilities_len", c_int)
    ]

# Define callback type
PROGRESS_CALLBACK = CFUNCTYPE(None, c_int, c_double, c_double)

# Define function prototypes
rust_lib.train_model_with_progress.argtypes = [
    c_char_p,  # csv_path
    c_int,     # epochs
    c_char_p,  # plot_path
    c_char_p,  # model_path
    POINTER(c_double),  # accuracy
    PROGRESS_CALLBACK   # callback
]
rust_lib.train_model_with_progress.restype = c_int

rust_lib.predict_air_quality.argtypes = [
    c_double,  # pm10
    c_double,  # so2
    c_double,  # co
    c_double,  # o3
    c_double,  # no2
    c_char_p   # model_path
]
rust_lib.predict_air_quality.restype = POINTER(PredictionResult)

rust_lib.free_prediction_result.argtypes = [POINTER(PredictionResult)]
rust_lib.free_prediction_result.restype = None

class TrainingThread(QThread):
    update_progress = pyqtSignal(int, float, float)
    training_complete = pyqtSignal(bool, float)
    
    def __init__(self, csv_path, epochs, learning_rate, parent=None):
        super().__init__(parent)
        self.csv_path = csv_path
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.running = True
        
    def progress_callback(self, epoch, accuracy, loss):
        self.update_progress.emit(epoch, accuracy, loss)
        print(f"Epoch {epoch}/{self.epochs} - Accuracy: {accuracy*100:.2f}%, Loss: {loss:.4f}")
        
    def run(self):
        # Create paths in project directory
        project_dir = os.path.dirname(os.path.abspath(__file__))
        plot_path = os.path.join(project_dir, "training_plot.png")
        model_path = os.path.join(project_dir, "trained_model.bin")
        
        accuracy = c_double(0.0)
        
        csv_path_bytes = self.csv_path.encode('utf-8')
        plot_path_bytes = plot_path.encode('utf-8')
        model_path_bytes = model_path.encode('utf-8')
        
        callback = PROGRESS_CALLBACK(self.progress_callback)
        
        success = rust_lib.train_model_with_progress(
            csv_path_bytes,
            self.epochs,
            plot_path_bytes,
            model_path_bytes,
            accuracy,
            callback
        )
        
        if success:
            self.training_complete.emit(True, accuracy.value)
            self.plot_path = plot_path
            self.model_path = model_path
        else:
            self.training_complete.emit(False, 0.0)
            if os.path.exists(plot_path):
                os.unlink(plot_path)
            if os.path.exists(model_path):
                os.unlink(model_path)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize model path first
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_model.bin")
        
        self.setWindowTitle("Air Quality Predictor with Neural Network Visualization")
        self.setGeometry(100, 100, 1200, 1000)  # Increased height to accommodate title image
        
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        
        # Main layout with title image at top
        main_layout = QVBoxLayout(self.main_widget)
        
        # Add title image section
        self.create_title_section(main_layout)
        
        # Create a horizontal splitter for main content
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Left side (main content)
        self.left_widget = QWidget()
        self.left_layout = QVBoxLayout(self.left_widget)
        
        # Right side (prediction panel)
        self.right_widget = QWidget()
        self.right_layout = QVBoxLayout(self.right_widget)
        
        self.csv_path = ""
        self.data_count = 0
        self.training_thread = None
        self.plot_path = ""
        self.epoch_data = []
        self.accuracy_data = []
        self.loss_data = []
        self.animation = None
        self.nn_weights = []
        
        self.create_data_loading_section()
        self.create_training_parameters_section()
        self.create_progress_section()
        self.create_visualization_section()
        self.create_prediction_section()
        
        # Add widgets to splitter
        self.splitter.addWidget(self.left_widget)
        self.splitter.addWidget(self.right_widget)
        
        # Set splitter sizes (left side takes more space)
        self.splitter.setSizes([1000, 400])
        
        # Add splitter to main layout
        main_layout.addWidget(self.splitter)
        
        # Enable predict button if model exists
        self.predict_button.setEnabled(os.path.exists(self.model_path))
    
    def create_title_section(self, layout):
        """Create the title section with image only"""
        title_widget = QWidget()
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(0, 0, 0, 20)  # Add some bottom margin

        self.title_label = QLabel()
        self.title_label.setAlignment(Qt.AlignCenter)

        image_name = "header.png"
        base_dir = os.path.dirname(os.path.abspath(__file__))

        possible_paths = [
            os.path.join(base_dir, image_name),
            os.path.join(base_dir, "neural_network_qtpython", image_name),
            os.path.join(base_dir, "images", image_name)
        ]

        image_loaded = False
        for image_path in possible_paths:
            if os.path.exists(image_path):
                pixmap = QPixmap(image_path)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(1200, 100, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
                    self.title_label.setPixmap(scaled_pixmap)
                    image_loaded = True
                    break

        if not image_loaded:
            print("⚠️  Gambar title_image.png tidak ditemukan atau gagal dimuat.")

        title_layout.addWidget(self.title_label)
        layout.addWidget(title_widget)
        
    def create_data_loading_section(self):
        group = QGroupBox("Data Loading")
        group.setStyleSheet("""
            QGroupBox {
                font-size: 15px;
                font-weight: bold;
                color: #222;
                border: 2px solid #0d47a1;
                border-radius: 8px;
                margin-top: 10px;
                padding: 10px;
            }
            QGroupBox:title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                background-color: #e3eafc;
                color: #0d47a1;
            }
        """)
        layout = QHBoxLayout()
        
        self.load_button = QPushButton("Load CSV")
        self.load_button.clicked.connect(self.load_csv)
        self.load_button.setStyleSheet("""
            QPushButton {
                background-color: #4285f4;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3367d6;
            }
            QPushButton:disabled {
                background-color: #bdbdbd;
                color: #eeeeee;
            }
        """)
        self.csv_label = QLabel("No file selected")
        self.data_count_label = QLabel("Data count: 0")
        
        layout.addWidget(self.load_button)
        layout.addWidget(self.csv_label)
        layout.addWidget(self.data_count_label)
        layout.addStretch()
        
        group.setLayout(layout)
        self.left_layout.addWidget(group)
    
    def create_training_parameters_section(self):
        group = QGroupBox("Training Parameters")
        group.setStyleSheet("""
            QGroupBox {
                font-size: 15px;
                font-weight: bold;
                color: #222;
                border: 2px solid #0d47a1;
                border-radius: 8px;
                margin-top: 10px;
                padding: 10px;
            }
            QGroupBox:title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                background-color: #e3eafc;
                color: #0d47a1;
            }
        """)
        layout = QHBoxLayout()
        epoch_layout = QVBoxLayout()
        epoch_label = QLabel("Epochs:")
        epoch_label.setStyleSheet("""
        QLabel {
            font-size: 14px;
            font-weight: bold;
            color: #333;
        }
        """)
        self.epoch_input = QLineEdit("100")
        self.epoch_input.setValidator(QtGui.QIntValidator(1, 10000))
        self.epoch_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 2px solid #cccccc;
                border-radius: 6px;
                font-size: 14px;
                background-color: #f9f9f9;
            }
            QLineEdit:focus {
                border: 2px solid #fb8c00;
                background-color: #fffaf3;
            }
        """)
        epoch_layout.addWidget(epoch_label)
        epoch_layout.addWidget(self.epoch_input)
        lr_layout = QVBoxLayout()
        lr_label = QLabel("Learning Rate:")
        lr_label.setStyleSheet("""
        QLabel {
            font-size: 14px;
            font-weight: bold;
            color: #333;
        }
        """)
        self.lr_input = QLineEdit("0.001")
        self.lr_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 2px solid #cccccc;
                border-radius: 6px;
                font-size: 14px;
                background-color: #f9f9f9;
            }
            QLineEdit:focus {
                border: 2px solid #fb8c00;
                background-color: #fffaf3;
            }
        """)
        self.lr_input.setValidator(QtGui.QDoubleValidator(0.0001, 1.0, 4))
        lr_layout.addWidget(lr_label)
        lr_layout.addWidget(self.lr_input)
        
        self.train_button = QPushButton("Train Model")
        self.train_button.setStyleSheet("""
            QPushButton {
                background-color: #e53935;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:disabled {
                background-color: #bdbdbd;
                color: #eeeeee;
            }
        """)
        self.train_button.clicked.connect(self.start_training)
        self.train_button.setEnabled(False)
        
        layout.addLayout(epoch_layout)
        layout.addLayout(lr_layout)
        layout.addWidget(self.train_button)
        layout.addStretch()
        
        group.setLayout(layout)
        self.left_layout.addWidget(group)
    
    def create_progress_section(self):
        group = QGroupBox("Training Progress")
        group.setStyleSheet("""
            QGroupBox {
                font-size: 15px;
                font-weight: bold;
                color: #222;
                border: 2px solid #0d47a1;
                border-radius: 8px;
                margin-top: 10px;
                padding: 10px;
            }
            QGroupBox:title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                background-color: #e3eafc;
                color: #0d47a1;
            }
        """)
        layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        
        self.epoch_label = QLabel("Epoch: 0/0")
        self.accuracy_label = QLabel("Accuracy: 0.0%")
        self.loss_label = QLabel("Loss: 0.0")
        
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(self.epoch_label)
        progress_layout.addWidget(self.accuracy_label)
        progress_layout.addWidget(self.loss_label)
        progress_layout.addStretch()
        
        layout.addWidget(self.progress_bar)
        layout.addLayout(progress_layout)
        
        group.setLayout(layout)
        self.left_layout.addWidget(group)
    
    def create_visualization_section(self):
        group = QGroupBox("Neural Network Visualization")
        group.setStyleSheet("""
            QGroupBox {
                font-size: 15px;
                font-weight: bold;
                color: #222;
                border: 2px solid #0d47a1;
                border-radius: 8px;
                margin-top: 10px;
                padding: 10px;
            }
            QGroupBox:title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                background-color: #e3eafc;
                color: #0d47a1;
            }
        """)
        layout = QVBoxLayout()
        
        self.tab_widget = QTabWidget()
        
        # Tab 1: Training Plots
        self.plots_tab = QWidget()
        self.plots_layout = QVBoxLayout(self.plots_tab)
        
        # Accuracy Plot
        self.accuracy_figure = Figure(figsize=(8, 3), dpi=100)
        self.accuracy_canvas = FigureCanvas(self.accuracy_figure)
        self.accuracy_ax = self.accuracy_figure.add_subplot(111)
        self.accuracy_ax.set_xlabel('Epoch')
        self.accuracy_ax.set_ylabel('Accuracy')
        self.accuracy_ax.grid(True)
        self.accuracy_line, = self.accuracy_ax.plot([], [], 'r-')
        
        # Loss Plot
        self.loss_figure = Figure(figsize=(8, 3), dpi=100)
        self.loss_canvas = FigureCanvas(self.loss_figure)
        self.loss_ax = self.loss_figure.add_subplot(111)
        self.loss_ax.set_xlabel('Epoch')
        self.loss_ax.set_ylabel('Loss')
        self.loss_ax.grid(True)
        self.loss_line, = self.loss_ax.plot([], [], 'b-')
        
        self.plots_layout.addWidget(self.accuracy_canvas)
        self.plots_layout.addWidget(self.loss_canvas)
        
        # Tab 2: Neural Network Architecture
        self.nn_tab = QWidget()
        self.nn_layout = QVBoxLayout(self.nn_tab)
        
        self.nn_figure = Figure(figsize=(10, 6), dpi=100)
        self.nn_canvas = FigureCanvas(self.nn_figure)
        self.nn_ax = self.nn_figure.add_subplot(111)
        self.nn_ax.axis('off')
        self.nn_layout.addWidget(self.nn_canvas)
        
        # Add tabs
        self.tab_widget.addTab(self.plots_tab, "Training Progress")
        self.tab_widget.addTab(self.nn_tab, "Network Architecture")
        
        layout.addWidget(self.tab_widget)
        group.setLayout(layout)
        self.left_layout.addWidget(group)
        
        # Initialize NN visualization
        self.draw_neural_net()
    
    def draw_neural_net(self, weights=None):
        self.nn_ax.clear()
        self.nn_ax.axis('off')
        
        # Network parameters
        layer_sizes = [5, 10, 10, 10, 3]  # Input, Hidden1, Hidden2, Hidden3, Output
        layer_colors = ['skyblue', 'lightgreen', 'lightgreen', 'lightgreen', 'salmon']
        
        # Calculate positions
        v_spacing = 1.0 / float(max(layer_sizes))
        h_spacing = 1.0 / float(len(layer_sizes) - 1)
        
        # Set limits with extra padding
        self.nn_ax.set_xlim(-0.1, 1.1)  # Padding on left and right
        self.nn_ax.set_ylim(-0.1, 1.1)  # Padding on top and bottom
        # Draw nodes
        for i, (n_nodes, color) in enumerate(zip(layer_sizes, layer_colors)):
            layer_top = v_spacing * (n_nodes - 1) / 2. + 0.5
            for j in range(n_nodes):
                circle = Circle((i * h_spacing, layer_top - j * v_spacing), 
                               v_spacing / 4.0, 
                               color=color, 
                               ec='k', 
                               zorder=4)
                self.nn_ax.add_patch(circle)
                
                # Add text for input/output labels
                if i == 0:
                    inputs = ['PM10', 'SO2', 'CO', 'O3', 'NO2']
                    self.nn_ax.text(i * h_spacing - 0.05, 
                                   layer_top - j * v_spacing, 
                                   inputs[j], 
                                   ha='right', 
                                   va='center')
                elif i == len(layer_sizes) - 1:
                    outputs = ['BAIK', 'SEDANG', 'TIDAK SEHAT']
                    self.nn_ax.text(i * h_spacing + 0.05, 
                                   layer_top - j * v_spacing, 
                                   outputs[j], 
                                   ha='left', 
                                   va='center')
        
        # Draw connections with weights
        if weights is not None and len(weights) == len(layer_sizes)-1:
            for i, (n_nodes, next_nodes) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                layer_top = v_spacing * (n_nodes - 1) / 2. + 0.5
                next_layer_top = v_spacing * (next_nodes - 1) / 2. + 0.5
                
                for j in range(n_nodes):
                    for k in range(next_nodes):
                        weight = weights[i][j][k]
                        linewidth = abs(weight) * 2.0
                        alpha = min(0.8, abs(weight))
                        color = 'green' if weight > 0 else 'red'
                        
                        line = Arrow(i * h_spacing, 
                                    layer_top - j * v_spacing,
                                    (i + 1) * h_spacing - i * h_spacing,
                                    (next_layer_top - k * v_spacing) - (layer_top - j * v_spacing),
                                    width=linewidth/100.0,
                                    color=color,
                                    alpha=alpha)
                        self.nn_ax.add_patch(line)
        
        self.nn_canvas.draw()
    
    def animate_nn(self, i):
        if i < len(self.nn_weights):
            self.draw_neural_net(self.nn_weights[i])
        return []
    
    def create_prediction_section(self):
        group = QGroupBox("Prediction")
        group.setStyleSheet("""
            QGroupBox {
                font-size: 15px;
                font-weight: bold;
                color: #222;
                border: 2px solid #0d47a1;
                border-radius: 8px;
                margin-top: 10px;
                padding: 10px;
            }
            QGroupBox:title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                background-color: #e3eafc;
                color: #0d47a1;
            }
        """)
        layout = QVBoxLayout()
        
        # Input fields
        input_layout = QVBoxLayout()
        
        # Create input fields with labels
        def create_input_field(label_text, placeholder):
            hbox = QHBoxLayout()
            label = QLabel(label_text)
            label.setFixedWidth(50)
            input_field = QLineEdit()
            input_field.setPlaceholderText(placeholder)
            input_field.setStyleSheet("""
                QLineEdit {
                    padding: 8px;
                    border: 2px solid #cccccc;
                    border-radius: 6px;
                    font-size: 14px;
                    background-color: #f9f9f9;
                }
                QLineEdit:focus {
                    border: 2px solid #fb8c00;
                    background-color: #fffaf3;
                }
            """)
            hbox.addWidget(label)
            hbox.addWidget(input_field)
            return hbox, input_field
        
        # Create all input fields
        pm10_layout, self.pm10_input = create_input_field("PM10:", "Enter PM10 value")
        so2_layout, self.so2_input = create_input_field("SO2:", "Enter SO2 value")
        co_layout, self.co_input = create_input_field("CO:", "Enter CO value")
        o3_layout, self.o3_input = create_input_field("O3:", "Enter O3 value")
        no2_layout, self.no2_input = create_input_field("NO2:", "Enter NO2 value")
        
        # Add input fields to layout
        input_layout.addLayout(pm10_layout)
        input_layout.addLayout(so2_layout)
        input_layout.addLayout(co_layout)
        input_layout.addLayout(o3_layout)
        input_layout.addLayout(no2_layout)
        
        # Predict button
        self.predict_button = QPushButton("Predict Air Quality")
        self.predict_button.setStyleSheet("""
            QPushButton {
                background-color: #fb8c00;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                margin-top: 15px;
                margin-bottom: 15px;
            }
            QPushButton:hover {
                background-color: #ef6c00;
            }
            QPushButton:disabled {
                background-color: #bdbdbd;
                color: #eeeeee;
            }
        """)
        self.predict_button.clicked.connect(self.predict)
        
        # Results display
        self.prediction_result = QLabel("Prediction: -")
        self.prediction_result.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #333;
                margin-top: 10px;
            }
        """)
        
        self.probabilities_label = QLabel("Probabilities: -")
        self.probabilities_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #555;
                margin-top: 5px;
            }
        """)
        
        # Add widgets to layout
        layout.addLayout(input_layout)
        layout.addWidget(self.predict_button)
        layout.addWidget(self.prediction_result)
        layout.addWidget(self.probabilities_label)
        layout.addStretch()
        
        group.setLayout(layout)
        self.right_layout.addWidget(group)
    
    def load_csv(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "", "CSV Files (*.csv)", options=options)
        
        if file_name:
            self.csv_path = file_name
            self.csv_label.setText(os.path.basename(file_name))
            
            try:
                with open(file_name, 'r') as f:
                    reader = csv.reader(f)
                    self.data_count = sum(1 for row in reader) - 1
                    self.data_count_label.setText(f"Data count: {self.data_count}")
                    
                    if self.data_count > 0:
                        self.train_button.setEnabled(True)
                    else:
                        QMessageBox.warning(self, "Warning", "The selected CSV file has no data rows.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to read CSV file: {str(e)}")
    
    def start_training(self):
        if not self.csv_path:
            QMessageBox.warning(self, "Warning", "Please load a CSV file first.")
            return
        
        try:
            epochs = int(self.epoch_input.text())
            learning_rate = float(self.lr_input.text())
        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter valid numbers for epochs and learning rate.")
            return
        
        # Reset animation if running
        if self.animation is not None:
            self.animation.event_source.stop()
            self.animation = None
        
        self.progress_bar.setValue(0)
        self.epoch_label.setText(f"Epoch: 0/{epochs}")
        self.accuracy_label.setText("Accuracy: 0.0%")
        self.loss_label.setText("Loss: 0.0")
        
        self.epoch_data = []
        self.accuracy_data = []
        self.loss_data = []
        self.nn_weights = []
        
        # Reset plots
        self.accuracy_ax.clear()
        self.accuracy_line, = self.accuracy_ax.plot([], [], 'r-')
        self.accuracy_ax.set_xlabel('Epoch')
        self.accuracy_ax.set_ylabel('Accuracy')
        self.accuracy_ax.grid(True)
        
        self.loss_ax.clear()
        self.loss_line, = self.loss_ax.plot([], [], 'b-')
        self.loss_ax.set_xlabel('Epoch')
        self.loss_ax.set_ylabel('Loss')
        self.loss_ax.grid(True)
        
        self.accuracy_canvas.draw()
        self.loss_canvas.draw()
        
        # Reset NN visualization
        self.draw_neural_net()
        
        self.load_button.setEnabled(False)
        self.train_button.setEnabled(False)
        
        self.training_thread = TrainingThread(self.csv_path, epochs, learning_rate)
        self.training_thread.update_progress.connect(self.update_training_progress)
        self.training_thread.training_complete.connect(self.training_finished)
        self.training_thread.start()
    
    def update_training_progress(self, epoch, accuracy, loss):
        epochs_total = int(self.epoch_input.text())
        progress = int((epoch / epochs_total) * 100)
        
        self.progress_bar.setValue(progress)
        self.epoch_label.setText(f"Epoch: {epoch}/{epochs_total}")
        self.accuracy_label.setText(f"Accuracy: {accuracy*100:.2f}%")
        self.loss_label.setText(f"Loss: {loss:.4f}")
        
        self.epoch_data.append(epoch)
        self.accuracy_data.append(accuracy)
        self.loss_data.append(loss)
        
        # Update accuracy plot
        self.accuracy_line.set_data(self.epoch_data, self.accuracy_data)
        if len(self.epoch_data) > 0:
            self.accuracy_ax.set_xlim(0, max(self.epoch_data))
            self.accuracy_ax.set_ylim(0, max(self.accuracy_data) * 1.1)
        self.accuracy_canvas.draw()
        
        # Update loss plot
        self.loss_line.set_data(self.epoch_data, self.loss_data)
        if len(self.epoch_data) > 0:
            self.loss_ax.set_xlim(0, max(self.epoch_data))
            self.loss_ax.set_ylim(0, max(self.loss_data) * 1.1)
        self.loss_canvas.draw()
        
        # For demo purposes, generate random weights (in real app, get from Rust)
        if epoch % 10 == 0:  # Update every 10 epochs for performance
            weights = [
                np.random.randn(5, 10) * 0.5,  # Input to Hidden1
                np.random.randn(10, 10) * 0.5,  # Hidden1 to Hidden2
                np.random.randn(10, 10) * 0.5,  # Hidden2 to Hidden3
                np.random.randn(10, 3) * 0.5    # Hidden3 to Output
            ]
            self.nn_weights.append(weights)
            
            # Start animation if not already running
            if len(self.nn_weights) > 1 and self.animation is None:
                self.animation = animation.FuncAnimation(
                    self.nn_figure, 
                    self.animate_nn,
                    frames=len(self.nn_weights),
                    interval=200,
                    blit=False,
                    repeat=True
                )
                self.nn_canvas.draw()
    
    def training_finished(self, success, final_accuracy):
        self.load_button.setEnabled(True)
        self.train_button.setEnabled(True)
        self.predict_button.setEnabled(True)
        
        # Stop animation when training completes
        if self.animation is not None:
            self.animation.event_source.stop()
            self.animation = None
            
        if success:
            self.progress_bar.setValue(100)
            self.accuracy_label.setText(f"Final Accuracy: {final_accuracy*100:.2f}%")
            QMessageBox.information(self, "Success", "Training completed successfully! Model saved to 'trained_model.bin'")
            
            if hasattr(self.training_thread, 'plot_path'):
                self.display_final_plot(self.training_thread.plot_path)
        else:
            QMessageBox.critical(self, "Error", "Training failed. Check your data and parameters.")
    
    def predict(self):
        # Ensure model exists
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_model.bin")
        if not os.path.exists(self.model_path):
            QMessageBox.warning(self, "Warning", "No trained model found. Please train the model first.")
            return
        
        try:
            # Get input values
            pm10 = float(self.pm10_input.text())
            so2 = float(self.so2_input.text())
            co = float(self.co_input.text())
            o3 = float(self.o3_input.text())
            no2 = float(self.no2_input.text())
            
            # Convert model path to bytes for Rust
            model_path_bytes = self.model_path.encode('utf-8')
            
            # Call Rust prediction function
            prediction_ptr = rust_lib.predict_air_quality(
                c_double(pm10),
                c_double(so2),
                c_double(co),
                c_double(o3),
                c_double(no2),
                model_path_bytes
            )
            
            if not prediction_ptr:
                QMessageBox.critical(self, "Error", "Prediction failed")
                return
                
            prediction = prediction_ptr.contents
            
            # Get probabilities array
            prob_array = np.ctypeslib.as_array(
                prediction.probabilities,
                shape=(prediction.probabilities_len,)
            ).copy()
            
            # Determine category
            category = "UNKNOWN"
            predicted_class = prediction.predicted_class
            
            if predicted_class == 0:
                category = "BAIK"
                image_name = "baik.png"
            elif predicted_class == 1:
                category = "SEDANG"
                image_name = "sedang.png"
            elif predicted_class == 2:
                category = "TIDAK SEHAT"
                image_name = "tidak_sehat.png"
            else:
                # Fallback to highest probability
                predicted_class = np.argmax(prob_array)
                if predicted_class == 0:
                    category = "BAIK"
                    image_name = "baik.png"
                elif predicted_class == 1:
                    category = "SEDANG"
                    image_name = "sedang.png"
                elif predicted_class == 2:
                    category = "TIDAK SEHAT"
                    image_name = "tidak_sehat.png"
            
            # Update UI with prediction result
            self.prediction_result.setText(f"Prediction: <b>{category}</b>")
            
            # Clear previous probability widgets if they exist
            if hasattr(self, 'prob_bars_group'):
                self.right_layout.removeWidget(self.prob_bars_group)
                self.prob_bars_group.deleteLater()
            
            # Create new group box for probability bars
            self.prob_bars_group = QGroupBox("Probability Distribution")
            self.prob_bars_group.setStyleSheet("""
                QGroupBox {
                    font-size: 14px;
                    font-weight: bold;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    margin-top: 10px;
                    padding: 10px;
                }
            """)
            prob_layout = QVBoxLayout()
            
            # Colors for each category
            colors = ['#4CAF50', '#FFC107', '#F44336']  # Green, Yellow, Red
            categories = ['BAIK', 'SEDANG', 'TIDAK SEHAT']
            
            # Create vertical bars for each category
            for i, (prob, color, cat) in enumerate(zip(prob_array, colors, categories)):
                # Create horizontal layout for this bar
                bar_layout = QHBoxLayout()
                
                # Category label
                cat_label = QLabel(cat)
                cat_label.setFixedWidth(80)
                cat_label.setStyleSheet("font-weight: bold;")
                bar_layout.addWidget(cat_label)
                
                # Progress bar (now horizontal)
                progress = QProgressBar()
                progress.setOrientation(Qt.Horizontal)  # Changed to horizontal
                progress.setRange(0, 100)
                progress.setValue(int(prob * 100))
                progress.setFormat("%p%")
                progress.setTextVisible(True)
                progress.setFixedHeight(30)  # Reduced height for horizontal bars
                progress.setStyleSheet(f"""
                    QProgressBar {{
                        border: 1px solid #999;
                        border-radius: 5px;
                        text-align: center;
                        height: 25px;
                    }}
                    QProgressBar::chunk {{
                        background-color: {color};
                        border-radius: 3px;
                    }}
                """)
                bar_layout.addWidget(progress, stretch=1)  # Allow progress bar to expand
                
                # Percentage label
                percent_label = QLabel(f"{prob*100:.1f}%")
                percent_label.setFixedWidth(50)  # Fixed width for percentage
                percent_label.setStyleSheet("font-size: 12px;")
                bar_layout.addWidget(percent_label)
                
                bar_layout.addStretch()
                prob_layout.addLayout(bar_layout)
            
            self.prob_bars_group.setLayout(prob_layout)
            self.right_layout.addWidget(self.prob_bars_group)
            
            # Display prediction image
            if hasattr(self, 'image_label'):
                self.right_layout.removeWidget(self.image_label)
                self.image_label.deleteLater()
            
            self.image_label = QLabel()
            self.image_label.setAlignment(Qt.AlignCenter)
            
            # Get the correct base directory
            base_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Check for image in several possible locations
            possible_paths = [
                os.path.join(base_dir, image_name),  # Directly in main folder
                os.path.join(base_dir, "neural_network_qtpython", image_name),  # In subfolder
                os.path.join(base_dir, "images", image_name)  # In images subfolder
            ]
            
            image_found = False
            for image_path in possible_paths:
                if os.path.exists(image_path):
                    pixmap = QtGui.QPixmap(image_path)
                    if not pixmap.isNull():
                        scaled_pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        self.image_label.setPixmap(scaled_pixmap)
                        image_found = True
                        break
            
            if not image_found:
                # Try absolute path as fallback
                abs_path = os.path.join("/home/lintang/Coding/neural_network_qtpython", image_name)
                if os.path.exists(abs_path):
                    pixmap = QtGui.QPixmap(abs_path)
                    if not pixmap.isNull():
                        scaled_pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        self.image_label.setPixmap(scaled_pixmap)
                        image_found = True
                
                if not image_found:
                    self.image_label.setText(f"Image not found. Tried:\n{possible_paths}\nand\n{abs_path}")
            
            self.right_layout.addWidget(self.image_label)
            self.right_layout.addStretch()
            
            # Free memory
            rust_lib.free_prediction_result(prediction_ptr)
            
        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter valid numbers for all input fields")
    
    def display_final_plot(self, plot_path):
        try:
            # This would display the final training plot
            # You could implement this if you want to show the final plot in a separate window
            pass
        except Exception as e:
            print(f"Error displaying final plot: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


