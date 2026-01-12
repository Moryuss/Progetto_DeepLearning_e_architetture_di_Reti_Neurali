import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
import shutil
import subprocess
import threading
from pathlib import Path
from src.config import DATASET_DIR, PEOPLE_DIR, KNOWN_PEOPLE_DIR, CLASSIFY_IMAGES_DIR


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Manager")
        self.root.geometry("900x700")

        # Style
        style = ttk.Style()
        style.theme_use('clam')

        # Main container
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # ========== SECTION 1: Import Images ==========
        import_frame = ttk.LabelFrame(
            main_frame, text="1. Import & Process Images", padding="10")
        import_frame.grid(row=0, column=0, columnspan=2,
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
        run_frame.grid(row=1, column=0, columnspan=2,
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
        scripts_frame.grid(row=2, column=0, columnspan=2,
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
        util_frame.grid(row=3, column=0, columnspan=2,
                        sticky=(tk.W, tk.E), pady=5)

        ttk.Button(util_frame, text="🔄 Re-extract All Embeddings",
                   command=self.reextract_embeddings).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(util_frame, text="🔄 Force Re-extract All",
                   command=lambda: self.reextract_embeddings(force=True)).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(util_frame, text="📊 Show Dataset Info",
                   command=self.show_dataset_info).grid(row=0, column=2, padx=5, pady=5)

        # ========== SECTION 5: Console Output ==========
        console_frame = ttk.LabelFrame(
            main_frame, text="Console Output", padding="10")
        console_frame.grid(row=4, column=0, columnspan=2,
                           sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        self.console = scrolledtext.ScrolledText(console_frame, height=15,
                                                 state='disabled', wrap=tk.WORD)
        self.console.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Button(console_frame, text="Clear Console",
                   command=self.clear_console).grid(row=1, column=0, pady=5)

        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        console_frame.columnconfigure(0, weight=1)
        console_frame.rowconfigure(0, weight=1)

        self.selected_files = []
        self.classify_files = []
        self.log("App initialized. Ready to use!")

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

        # Confirm
        confirm_msg = (
            f"Import {len(self.selected_files)} images as '{person_name}'\n"
            f"Target: {target}\n\n"
            f"This will:\n"
            f"1. Copy images to appropriate folder\n"
            f"2. Run augmentation\n"
            f"3. Extract embeddings\n\n"
            f"Continue?"
        )

        if not messagebox.askyesno("Confirm Import", confirm_msg):
            return

        # Run in thread to avoid blocking UI
        thread = threading.Thread(target=self._do_import_and_process,
                                  args=(person_name, target))
        thread.start()

    def _do_import_and_process(self, person_name, target):
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
            self.log(f"Extracting embeddings ({script})...")

            # Aggiungi parametri per processing incrementale
            cmd = [sys.executable, "-m", f"src.{script.replace('.py', '')}"]
            if target == "dataset" and person_name:
                cmd.extend(["--person", person_name])

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
        confirm_msg = f"Run {script_name}?\n\nThis will open a new window."

        if not messagebox.askyesno("Confirm Run", confirm_msg):
            return

        self.log(f"Starting {script_name}...")

        thread = threading.Thread(
            target=self._do_run_script, args=(script_name,))
        thread.start()

    def _do_run_script(self, script_name):
        """Internal method to run script"""
        try:
            result = subprocess.run(
                [sys.executable, "-m", script_name],
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
        force_text = " (FORCE)" if force else ""
        confirm_msg = (
            f"Re-extract all embeddings{force_text}?\n\n"
            f"This will run:\n"
            f"- image_to_embedding.py{' --force' if force else ''}\n"
            f"- extract_embeddings.py\n\n"
            f"{'FORCE will ignore cache and reprocess everything.' if force else 'Only changed files will be processed (incremental).'}\n\n"
            f"This may take several minutes."
        )

        if not messagebox.askyesno("Confirm Re-extraction", confirm_msg):
            return

        thread = threading.Thread(target=self._do_reextract, args=(force,))
        thread.start()

    def _do_reextract(self, force=False):
        """Internal method to re-extract embeddings"""
        scripts = ["src.image_to_embedding", "src.extract_embeddings"]

        for script in scripts:
            self.log(f"Running {script}...")
            try:
                cmd = [sys.executable, "-m", script]
                if force and script == "src.image_to_embedding":
                    cmd.append("--force")

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
            dataset_path = Path(DATASET_DIR)
            info = []

            if dataset_path.exists():
                people = [d for d in dataset_path.iterdir() if d.is_dir()]
                info.append(f"Dataset People: {len(people)}")
                for person_dir in people:
                    images_dir = person_dir / "images"
                    augmented_dir = person_dir / "augmented"
                    emb_files = list(person_dir.glob("embeddings_*.npz"))

                    n_images = len(list(images_dir.glob("*.*"))
                                   ) if images_dir.exists() else 0
                    n_aug = len(list(augmented_dir.glob("*.*"))
                                ) if augmented_dir.exists() else 0

                    info.append(
                        f"  {person_dir.name}: {n_images} orig, {n_aug} aug, {len(emb_files)} emb files")

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


def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
