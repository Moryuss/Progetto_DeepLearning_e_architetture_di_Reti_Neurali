import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
import shutil
import subprocess
import threading
from pathlib import Path
from src.config import (
    DATASET_DIR, PEOPLE_DIR, KNOWN_PEOPLE_DIR, CLASSIFY_IMAGES_DIR,
    AVAILABLE_MODELS, DEFAULT_MODEL
)


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Manager")
        self.root.geometry("900x900")  # ← Era 900x750, aumenta altezza

        # Style
        style = ttk.Style()
        style.theme_use('clam')

        # ========== AGGIUNGI CANVAS + SCROLLBAR ==========
        # Canvas con scrollbar
        canvas = tk.Canvas(root)
        scrollbar = ttk.Scrollbar(
            root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, padding="10")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Grid canvas e scrollbar
        canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        # Abilita scroll con mouse wheel

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        # Main container è ora scrollable_frame invece di ttk.Frame(root)
        main_frame = scrollable_frame  # ← Usa questo invece di creare nuovo frame

        # ========== SECTION 0: Model Selection (NUOVO!) ==========
        model_frame = ttk.LabelFrame(
            main_frame, text="🤖 Model Selection", padding="10")
        model_frame.grid(row=0, column=0, columnspan=2,
                         sticky=(tk.W, tk.E), pady=5)

        ttk.Label(model_frame, text="Recognition Model:").grid(
            row=0, column=0, sticky=tk.W, padx=5)

        # Dropdown con modelli disponibili
        self.selected_model = tk.StringVar(value=DEFAULT_MODEL)
        self.model_dropdown = ttk.Combobox(
            model_frame,
            textvariable=self.selected_model,
            values=list(AVAILABLE_MODELS.keys()),
            state="readonly",
            width=40
        )
        self.model_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        self.model_dropdown.bind("<<ComboboxSelected>>", self.on_model_changed)

        # Label con info sul modello
        self.model_info_label = ttk.Label(
            model_frame,
            text=AVAILABLE_MODELS[DEFAULT_MODEL]["description"],
            foreground="gray",
            wraplength=600
        )
        self.model_info_label.grid(
            row=1, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)

        # Label con path del modello
        model_path = AVAILABLE_MODELS[DEFAULT_MODEL]["model_path"]
        model_exists = model_path.exists() if isinstance(
            model_path, Path) else Path(model_path).exists()
        status_text = f"✓ Model found: {model_path}" if model_exists else f"✗ Model NOT found: {model_path}"
        status_color = "green" if model_exists else "red"

        self.model_path_label = ttk.Label(
            model_frame,
            text=status_text,
            foreground=status_color,
            font=("TkDefaultFont", 8)
        )
        self.model_path_label.grid(
            row=2, column=0, columnspan=3, sticky=tk.W, padx=5)

        # ========== SECTION 1: Import Images ==========
        import_frame = ttk.LabelFrame(
            main_frame, text="1. Import & Process Images", padding="10")
        import_frame.grid(row=1, column=0, columnspan=2,
                          sticky=(tk.W, tk.E), pady=5)

        ttk.Label(import_frame, text="Target:").grid(
            row=0, column=0, sticky=tk.W)
        self.target_var = tk.StringVar(value="dataset")
        targets = [
            ("Dataset (Camera/Images)", "dataset"),
            ("Known People (Look-alike)", "known"),
            ("People (Look-alike)", "people")
        ]
        for i, (text, value) in enumerate(targets):
            ttk.Radiobutton(import_frame, text=text, variable=self.target_var,
                            value=value).grid(row=0, column=i+1, padx=5)

        ttk.Label(import_frame, text="Person Name:").grid(
            row=1, column=0, sticky=tk.W, pady=5)
        self.person_name_entry = ttk.Entry(import_frame, width=30)
        self.person_name_entry.grid(
            row=1, column=1, columnspan=2, sticky=tk.W, pady=5)

        # Checkbox per force reprocess
        self.force_reprocess_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(import_frame, text="Force reprocess (ignore cache)",
                        variable=self.force_reprocess_var).grid(row=2, column=2, sticky=tk.W)

        ttk.Button(import_frame, text="📁 Select Images",
                   command=self.select_images).grid(row=2, column=0, pady=5)
        ttk.Button(import_frame, text="▶️ Import & Process",
                   command=self.import_and_process).grid(row=2, column=1, pady=5)

        self.selected_files_label = ttk.Label(import_frame, text="No images selected",
                                              foreground="gray")
        self.selected_files_label.grid(
            row=3, column=0, columnspan=3, sticky=tk.W)

        # ========== SECTION 2: Run Scripts ==========
        run_frame = ttk.LabelFrame(
            main_frame, text="2. Load Images to Classify", padding="10")
        run_frame.grid(row=2, column=0, columnspan=2,
                       sticky=(tk.W, tk.E), pady=5)

        ttk.Label(run_frame, text="Images for recognition:").grid(
            row=0, column=0, sticky=tk.W)
        ttk.Button(run_frame, text="📁 Select Images to Classify",
                   command=self.load_classify_images).grid(row=0, column=1, padx=5, pady=5)
        self.classify_images_label = ttk.Label(run_frame, text="No images loaded",
                                               foreground="gray")
        self.classify_images_label.grid(row=0, column=2, sticky=tk.W, padx=5)

        # ========== SECTION 3: Run Scripts ==========
        scripts_frame = ttk.LabelFrame(
            main_frame, text="3. Run Main Scripts", padding="10")
        scripts_frame.grid(row=3, column=0, columnspan=2,
                           sticky=(tk.W, tk.E), pady=5)

        scripts = [
            ("📷 Camera Recognition", "src.main_camera"),
            ("🖼️ Image Recognition", "src.main_image"),
            ("👥 Look-alike Offline", "src.main_look_alike_offline"),
            ("🎥 Look-alike Online", "src.main_look_alike_online")
        ]

        for i, (text, script) in enumerate(scripts):
            row = i // 2
            col = i % 2
            ttk.Button(scripts_frame, text=text,
                       command=lambda s=script: self.run_script(s),
                       width=30).grid(row=row, column=col, padx=5, pady=5)

        # ========== SECTION 4: Utilities ==========
        util_frame = ttk.LabelFrame(
            main_frame, text="4. Utilities", padding="10")
        util_frame.grid(row=4, column=0, columnspan=2,
                        sticky=(tk.W, tk.E), pady=5)

        ttk.Button(util_frame, text="🔄 Re-extract All Embeddings",
                   command=self.reextract_embeddings).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(util_frame, text="🔄 Force Re-extract All",
                   command=lambda: self.reextract_embeddings(force=True)).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(util_frame, text="📊 Show Dataset Info",
                   command=self.show_dataset_info).grid(row=0, column=2, padx=5, pady=5)

        # ========== SECTION 5: Model Testing  ==========
        test_frame = ttk.LabelFrame(
            main_frame, text="5. Model Testing", padding="10")
        test_frame.grid(row=5, column=0, columnspan=2,
                        sticky=(tk.W, tk.E), pady=5)

        # Row 0: Load test images
        ttk.Label(test_frame, text="Test Images:").grid(
            row=0, column=0, sticky=tk.W)
        ttk.Button(test_frame, text="📁 Load Test Images",
                   command=self.load_test_images).grid(row=0, column=1, padx=5, pady=5)
        self.test_images_label = ttk.Label(test_frame, text="No test images loaded",
                                           foreground="gray")
        self.test_images_label.grid(row=0, column=2, sticky=tk.W, padx=5)

        # Row 1: Threshold setting
        ttk.Label(test_frame, text="Recognition Threshold:").grid(
            row=1, column=0, sticky=tk.W, pady=5)
        self.threshold_var = tk.DoubleVar(value=0.60)
        threshold_spinbox = ttk.Spinbox(
            test_frame,
            from_=0.0,
            to=1.0,
            increment=0.05,
            textvariable=self.threshold_var,
            width=10
        )
        threshold_spinbox.grid(row=1, column=1, sticky=tk.W, pady=5)
        ttk.Label(test_frame, text="(0.0 - 1.0)", foreground="gray").grid(
            row=1, column=2, sticky=tk.W)

        # Row 2: Model selection for testing
        ttk.Label(test_frame, text="Models to test:").grid(
            row=2, column=0, sticky=tk.W)
        self.test_all_models_var = tk.BooleanVar(value=True)
        ttk.Radiobutton(test_frame, text="All models",
                        variable=self.test_all_models_var,
                        value=True).grid(row=2, column=1, sticky=tk.W)
        ttk.Radiobutton(test_frame, text="Current model only",
                        variable=self.test_all_models_var,
                        value=False).grid(row=2, column=2, sticky=tk.W)

        # Row 3: Run test button
        ttk.Button(test_frame, text="🧪 Run Model Testing",
                   command=self.run_model_testing,
                   width=30).grid(row=3, column=0, columnspan=3, pady=10)

        ttk.Label(test_frame,
                  text="Note: Test images must be in folders named by person (e.g., test_images/mario_rossi/img1.jpg)",
                  foreground="gray",
                  font=("TkDefaultFont", 8),
                  wraplength=700).grid(row=4, column=0, columnspan=3, sticky=tk.W)

        # ========== SECTION 6: Console Output ==========
        console_frame = ttk.LabelFrame(
            main_frame, text="Console Output", padding="10")
        console_frame.grid(row=6, column=0, columnspan=2,
                           sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        self.console = scrolledtext.ScrolledText(console_frame, height=15,
                                                 state='disabled', wrap=tk.WORD)
        self.console.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Button(console_frame, text="Clear Console",
                   command=self.clear_console).grid(row=1, column=0, pady=5)

        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        # main_frame.columnconfigure(0, weight=1)
        # main_frame.rowconfigure(6, weight=1)
        console_frame.columnconfigure(0, weight=1)
        console_frame.rowconfigure(0, weight=1)

        self.selected_files = []
        self.classify_files = []
        self.test_images_dir = None  # Path alla cartella test images
        self.log("App initialized. Ready to use!")
        self.log(f"Selected model: {DEFAULT_MODEL}", "INFO")

    def on_model_changed(self, event=None):
        """Callback quando cambia il modello selezionato"""
        model_name = self.selected_model.get()
        model_config = AVAILABLE_MODELS[model_name]

        # Aggiorna descrizione
        self.model_info_label.config(text=model_config["description"])

        # Verifica se il modello esiste
        model_path = model_config["model_path"]
        model_exists = model_path.exists() if isinstance(
            model_path, Path) else Path(model_path).exists()

        status_text = f"✓ Model found: {model_path}" if model_exists else f"✗ Model NOT found: {model_path}"
        status_color = "green" if model_exists else "red"

        self.model_path_label.config(text=status_text, foreground=status_color)

        self.log(f"Model changed to: {model_name}", "INFO")
        if not model_exists:
            self.log(
                f"⚠️ Warning: Model file not found at {model_path}", "WARNING")

    def get_current_model_name(self):
        """Ottiene il nome del modello attualmente selezionato"""
        return self.selected_model.get()

    def log(self, message, level="INFO"):
        """Log message to console"""
        self.console.config(state='normal')
        color_tag = {
            "INFO": "black",
            "SUCCESS": "green",
            "WARNING": "orange",
            "ERROR": "red"
        }.get(level, "black")

        self.console.insert(tk.END, f"[{level}] {message}\n")
        self.console.see(tk.END)
        self.console.config(state='disabled')

    def clear_console(self):
        self.console.config(state='normal')
        self.console.delete(1.0, tk.END)
        self.console.config(state='disabled')

    def select_images(self):
        """Select multiple images"""
        files = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[("Image files", "*.jpg *.jpeg *.png"),
                       ("All files", "*.*")]
        )
        if files:
            self.selected_files = list(files)
            self.selected_files_label.config(
                text=f"{len(self.selected_files)} images selected",
                foreground="green"
            )
            self.log(f"Selected {len(self.selected_files)} images")

    def load_classify_images(self):
        """Load images to classify"""
        files = filedialog.askopenfilenames(
            title="Select Images to Classify",
            filetypes=[("Image files", "*.jpg *.jpeg *.png"),
                       ("All files", "*.*")]
        )
        if not files:
            return

        # Conferma
        confirm_msg = (
            f"Copy {len(files)} images to classification folder?\n\n"
            f"Target: {CLASSIFY_IMAGES_DIR}\n\n"
            f"These images will be used by 'Image Recognition'."
        )

        if not messagebox.askyesno("Confirm Load", confirm_msg):
            return

        try:
            # Crea directory se non esiste
            classify_dir = Path(CLASSIFY_IMAGES_DIR)
            classify_dir.mkdir(parents=True, exist_ok=True)

            # Copia immagini
            for file in files:
                filename = os.path.basename(file)
                dest = classify_dir / filename
                shutil.copy2(file, dest)
                self.log(f"Copied: {filename}")

            self.classify_images_label.config(
                text=f"{len(files)} images loaded",
                foreground="green"
            )
            self.log(
                f"✓ {len(files)} images loaded for classification", "SUCCESS")

        except Exception as e:
            self.log(f"✗ Error loading images: {str(e)}", "ERROR")
            messagebox.showerror("Error", f"Failed to load images:\n{str(e)}")

    def import_and_process(self):
        """Import images and run augmentation + embedding extraction"""
        if not self.selected_files:
            messagebox.showwarning("Warning", "Please select images first!")
            return

        person_name = self.person_name_entry.get().strip()
        if not person_name:
            messagebox.showwarning("Warning", "Please enter a person name!")
            return

        target = self.target_var.get()
        model_name = self.get_current_model_name()

        # Confirm
        confirm_msg = (
            f"Import {len(self.selected_files)} images as '{person_name}'\n"
            f"Target: {target}\n"
            f"Model: {model_name}\n\n"
            f"This will:\n"
            f"1. Copy images to appropriate folder\n"
            f"2. Run augmentation\n"
            f"3. Extract embeddings using selected model\n\n"
            f"Continue?"
        )

        if not messagebox.askyesno("Confirm Import", confirm_msg):
            return

        # Run in thread to avoid blocking UI
        thread = threading.Thread(target=self._do_import_and_process,
                                  args=(person_name, target, model_name))
        thread.start()

    def _do_import_and_process(self, person_name, target, model_name):
        """Internal method to import and process"""
        try:
            # Determine target directory
            if target == "dataset":
                base_dir = Path(DATASET_DIR)
                person_dir = base_dir / person_name / "images"
                needs_augment = True
                script = "image_to_embedding.py"
            elif target == "known":
                base_dir = Path(KNOWN_PEOPLE_DIR)
                person_dir = base_dir
                needs_augment = False
                script = "extract_embeddings.py"
            else:  # people
                base_dir = Path(PEOPLE_DIR)
                person_dir = base_dir
                needs_augment = False
                script = "extract_embeddings.py"

            # Create directory
            person_dir.mkdir(parents=True, exist_ok=True)
            self.log(f"Target directory: {person_dir}")

            # Copy images
            for file in self.selected_files:
                filename = os.path.basename(file)
                dest = person_dir / filename
                shutil.copy2(file, dest)
                self.log(f"Copied: {filename}")

            self.log(f"✓ Copied {len(self.selected_files)} images", "SUCCESS")

            # Run augmentation if needed
            if needs_augment:
                self.log("Running augmentation...")
                result = subprocess.run(
                    [sys.executable, "-m", "src.augment"],
                    capture_output=True,
                    text=True
                )
                self.log(result.stdout)
                if result.returncode == 0:
                    self.log("✓ Augmentation completed", "SUCCESS")
                else:
                    self.log(
                        f"✗ Augmentation failed: {result.stderr}", "ERROR")
                    return

            # Extract embeddings
            self.log(
                f"Extracting embeddings ({script}) with model: {model_name}...")

            # Aggiungi parametri per processing incrementale + modello
            cmd = [sys.executable, "-m", f"src.{script.replace('.py', '')}"]
            if target == "dataset" and person_name:
                cmd.extend(["--person", person_name])

            # Passa il modello selezionato
            cmd.extend(["--model", model_name])

            result = subprocess.run(cmd, capture_output=True, text=True)
            self.log(result.stdout)
            if result.returncode == 0:
                self.log("✓ Embeddings extracted", "SUCCESS")
                self.log("=== PROCESS COMPLETED SUCCESSFULLY ===", "SUCCESS")
            else:
                self.log(
                    f"✗ Embedding extraction failed: {result.stderr}", "ERROR")

        except Exception as e:
            self.log(f"✗ Error: {str(e)}", "ERROR")
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")

        finally:
            # Reset UI
            self.root.after(0, self._reset_import_ui)

    def _reset_import_ui(self):
        self.selected_files = []
        self.selected_files_label.config(
            text="No images selected", foreground="gray")
        self.person_name_entry.delete(0, tk.END)

    def run_script(self, script_name):
        """Run a main script"""
        model_name = self.get_current_model_name()

        confirm_msg = (
            f"Run {script_name}?\n\n"
            f"Model: {model_name}\n\n"
            f"This will open a new window."
        )

        if not messagebox.askyesno("Confirm Run", confirm_msg):
            return

        self.log(f"Starting {script_name} with model: {model_name}...")

        thread = threading.Thread(
            target=self._do_run_script, args=(script_name, model_name))
        thread.start()

    def _do_run_script(self, script_name, model_name):
        """Internal method to run script"""
        try:
            # Passa il modello come argomento
            result = subprocess.run(
                [sys.executable, "-m", script_name, "--model", model_name],
                capture_output=True,
                text=True
            )
            self.log(result.stdout)
            if result.returncode == 0:
                self.log(f"✓ {script_name} completed", "SUCCESS")
            else:
                self.log(f"✗ {script_name} failed: {result.stderr}", "ERROR")
        except Exception as e:
            self.log(f"✗ Error running {script_name}: {str(e)}", "ERROR")

    def reextract_embeddings(self, force=False):
        """Re-extract all embeddings"""
        model_name = self.get_current_model_name()
        force_text = " (FORCE)" if force else ""

        confirm_msg = (
            f"Re-extract all embeddings{force_text}?\n\n"
            f"Model: {model_name}\n\n"
            f"This will run:\n"
            f"- image_to_embedding.py{' --force' if force else ''}\n"
            f"- extract_embeddings.py\n\n"
            f"{'FORCE will ignore cache and reprocess everything.' if force else 'Only changed files will be processed (incremental).'}\n\n"
            f"This may take several minutes."
        )

        if not messagebox.askyesno("Confirm Re-extraction", confirm_msg):
            return

        thread = threading.Thread(
            target=self._do_reextract, args=(force, model_name))
        thread.start()

    def _do_reextract(self, force=False, model_name=None):
        """Internal method to re-extract embeddings"""
        scripts = ["src.image_to_embedding", "src.extract_embeddings"]

        for script in scripts:
            self.log(f"Running {script} with model: {model_name}...")
            try:
                cmd = [sys.executable, "-m", script]
                if force and script == "src.image_to_embedding":
                    cmd.append("--force")

                # Passa il modello
                if model_name:
                    cmd.extend(["--model", model_name])

                result = subprocess.run(cmd, capture_output=True, text=True)
                self.log(result.stdout)
                if result.returncode == 0:
                    self.log(f"✓ {script} completed", "SUCCESS")
                else:
                    self.log(f"✗ {script} failed: {result.stderr}", "ERROR")
                    return
            except Exception as e:
                self.log(f"✗ Error: {str(e)}", "ERROR")
                return

        self.log("=== ALL EMBEDDINGS RE-EXTRACTED ===", "SUCCESS")

    def show_dataset_info(self):
        """Show dataset statistics"""
        try:
            model_name = self.get_current_model_name()
            model_config = AVAILABLE_MODELS[model_name]
            suffix = model_config["embeddings_suffix"]

            dataset_path = Path(DATASET_DIR)
            info = [f"Current Model: {model_name}",
                    f"Embeddings suffix: {suffix}", ""]

            if dataset_path.exists():
                people = [d for d in dataset_path.iterdir() if d.is_dir()]
                info.append(f"Dataset People: {len(people)}")
                for person_dir in people:
                    images_dir = person_dir / "images"
                    augmented_dir = person_dir / "augmented"
                    emb_files = list(person_dir.glob(
                        f"embeddings_{suffix}.npz"))

                    n_images = len(list(images_dir.glob("*.*"))
                                   ) if images_dir.exists() else 0
                    n_aug = len(list(augmented_dir.glob("*.*"))
                                ) if augmented_dir.exists() else 0

                    emb_status = "✓" if emb_files else "✗"
                    info.append(
                        f"  {emb_status} {person_dir.name}: {n_images} orig, {n_aug} aug")

            people_path = Path(PEOPLE_DIR)
            if people_path.exists():
                n_people = len(list(people_path.glob("*.*")))
                info.append(f"\nPeople (look-alike): {n_people} images")

            known_path = Path(KNOWN_PEOPLE_DIR)
            if known_path.exists():
                n_known = len(list(known_path.glob("*.*")))
                info.append(f"Known People: {n_known} images")

            messagebox.showinfo("Dataset Info", "\n".join(info))
            self.log("Dataset info displayed")

        except Exception as e:
            messagebox.showerror(
                "Error", f"Could not read dataset info:\n{str(e)}")

    def load_test_images(self):
        """Load test images from folder structure"""
        test_dir = filedialog.askdirectory(
            title="Select Test Images Folder",
            initialdir=str(Path.cwd() / "data")
        )

        if not test_dir:
            return

        # Conta immagini e persone
        test_path = Path(test_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

        total_images = 0
        people = []

        for person_folder in test_path.iterdir():
            if not person_folder.is_dir():
                continue

            images = [f for f in person_folder.iterdir()
                      if f.suffix.lower() in image_extensions]

            if len(images) > 0:
                people.append(person_folder.name)
                total_images += len(images)

        if total_images == 0:
            messagebox.showwarning(
                "No Images Found",
                "No test images found in selected folder.\n\n"
                "Expected structure:\n"
                "test_images/\n"
                "├── person1/\n"
                "│   └── img1.jpg\n"
                "└── person2/\n"
                "    └── img1.jpg"
            )
            return

        # Salva path
        self.test_images_dir = test_dir

        # Aggiorna UI
        self.test_images_label.config(
            text=f"{total_images} images from {len(people)} people loaded",
            foreground="green"
        )

        self.log(
            f"Loaded {total_images} test images from {len(people)} people")
        self.log(f"Test directory: {test_dir}")

    def run_model_testing(self):
        """Run systematic model testing"""
        # Verifica test images
        if not hasattr(self, 'test_images_dir'):
            messagebox.showwarning(
                "Warning",
                "Please load test images first!"
            )
            return

        # Determina quali modelli testare
        test_all = self.test_all_models_var.get()
        threshold = self.threshold_var.get()

        if test_all:
            models_list = [m for m in AVAILABLE_MODELS.keys()
                           if "not working" not in m.lower()]
            confirm_msg = (
                f"Run testing on ALL models?\n\n"
                f"Models to test: {len(models_list)}\n"
                f"Threshold: {threshold}\n\n"
                f"This will:\n"
                f"- Test all models on test images\n"
                f"- Generate accuracy tables and confusion matrices\n"
                f"- Save results in test_results/ folder\n\n"
                f"This may take several minutes.\n\n"
                f"Continue?"
            )
        else:
            current_model = self.get_current_model_name()
            models_list = [current_model]
            confirm_msg = (
                f"Run testing on current model?\n\n"
                f"Model: {current_model}\n"
                f"Threshold: {threshold}\n\n"
                f"This will test the model and generate results.\n\n"
                f"Continue?"
            )

        if not messagebox.askyesno("Confirm Testing", confirm_msg):
            return

        self.log("="*60, "INFO")
        self.log("🧪 STARTING MODEL TESTING", "INFO")
        self.log(f"Test directory: {self.test_images_dir}", "INFO")
        self.log(f"Models to test: {len(models_list)}", "INFO")
        self.log(f"Threshold: {threshold}", "INFO")
        self.log("="*60, "INFO")

        # Run in thread
        thread = threading.Thread(
            target=self._do_model_testing,
            args=(models_list, threshold)
        )
        thread.start()

    def _do_model_testing(self, models_list, threshold):
        """Internal method to run model testing"""
        try:
            # Verifica che test_images_dir sia valido
            if not self.test_images_dir:
                self.log("✗ Error: No test directory specified", "ERROR")
                return

            # Costruisci comando - converti tutto in stringhe
            cmd = [
                sys.executable, "-m", "src.test_models",
                "--test-dir", str(self.test_images_dir),
                "--threshold", str(threshold)
            ]

            # Aggiungi modelli specifici se non tutti
            if len(models_list) < len(AVAILABLE_MODELS):
                cmd.append("--models")
                cmd.extend([str(m) for m in models_list])

            self.log(f"Running: {' '.join(cmd)}", "INFO")

            # Esegui test - AGGIUNGI encoding='utf-8' e errors='replace'
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',      # ← AGGIUNGI
                # ← AGGIUNGI (sostituisce caratteri non decodificabili)
                errors='replace'
            )

            # Log output
            if result.stdout:
                for line in result.stdout.split('\n'):
                    if line.strip():
                        self.log(line, "INFO")

            if result.returncode == 0:
                self.log("="*60, "SUCCESS")
                self.log("✓ MODEL TESTING COMPLETED!", "SUCCESS")
                self.log(
                    "Check test_results/ folder for detailed results", "SUCCESS")
                self.log("="*60, "SUCCESS")

                # Messagebox finale
                self.root.after(0, lambda: messagebox.showinfo(
                    "Testing Complete",
                    "Model testing completed successfully!\n\n"
                    "Results saved in test_results/ folder:\n"
                    "- results.csv (detailed data)\n"
                    "- results_table.png (visual table)\n"
                    "- accuracy_comparison.png (bar chart)\n"
                    "- confusion_matrices/ (one per model)\n"
                    "- wrong_predictions/ (images with errors)"
                ))
            else:
                self.log(f"✗ Testing failed: {result.stderr}", "ERROR")

        except Exception as e:
            self.log(f"✗ Error running testing: {str(e)}", "ERROR")
            import traceback
            self.log(traceback.format_exc(), "ERROR")


def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
