"""
Script per testare sistematicamente tutti i modelli di face recognition.

Struttura cartelle test richiesta:
test_images/
├── mario_rossi/
│   ├── img1.jpg
│   └── img2.jpg
├── giulia_bianchi/
│   └── img1.jpg
└── unknown/  (opzionale, per testare Unknown detection)
    └── stranger.jpg

Output:
- Tabella con risultati (✓/✗) per ogni modello
- Confusion matrix per ogni modello
- CSV con risultati dettagliati
- Immagini con predizioni sbagliate salvate in output/
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm

from src.utils import recognize_faces, load_dataset_embeddings, draw_label
from src.inizializer import initialization_detector_recognizer
from src.config import (
    DATASET_DIR,
    DETECTOR_MODEL_PATH,
    AVAILABLE_MODELS,
    BASE_DIR,
    TEST_IMAGES_DIR
)


class ModelTestResults:
    """Classe per gestire i risultati del testing"""

    def __init__(self):
        self.results = []

    def add_result(self, image_path: str, model_name: str, true_label: str,
                   predicted_label: str, confidence: float, detection_success: bool,
                   inference_time: float):
        """Aggiunge un risultato di test"""
        self.results.append({
            'image': Path(image_path).name,
            'true_label': true_label,
            'model': model_name,
            'predicted': predicted_label,
            'confidence': confidence,
            'correct': predicted_label == true_label,
            'detection_success': detection_success,
            'inference_time_ms': inference_time * 1000
        })

    def get_dataframe(self) -> pd.DataFrame:
        """Converte risultati in DataFrame pandas"""
        return pd.DataFrame(self.results)

    def get_accuracy_per_model(self) -> Dict[str, float]:
        """Calcola accuracy per ogni modello"""
        df = self.get_dataframe()
        accuracy = {}
        for model in df['model'].unique():
            model_results = df[df['model'] == model]
            # Considera solo immagini dove la detection è riuscita
            detected = model_results[model_results['detection_success'] == True]
            if len(detected) > 0:
                accuracy[model] = (
                    detected['correct'].sum() / len(detected)) * 100
            else:
                accuracy[model] = 0.0
        return accuracy

    def print_summary(self):
        """Stampa summary dei risultati"""
        df = self.get_dataframe()

        print("\n" + "="*80)
        print("📊 SUMMARY RISULTATI TEST")
        print("="*80)

        # Accuracy per modello
        print("\n🎯 ACCURACY PER MODELLO:")
        accuracies = self.get_accuracy_per_model()
        for model, acc in sorted(accuracies.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model:50s}: {acc:6.2f}%")

        # Statistiche globali
        print(f"\n📈 STATISTICHE GLOBALI:")
        print(f"  Totale immagini testate: {len(df['image'].unique())}")
        print(f"  Totale modelli testati: {len(df['model'].unique())}")
        print(
            f"  Totale persone nel dataset: {len(df['true_label'].unique())}")

        # Detection failures
        detection_failures = df[df['detection_success'] == False]
        if len(detection_failures) > 0:
            print(f"\n⚠️  DETECTION FAILURES:")
            print(f"  Totale failures: {len(detection_failures)}")
            for model in detection_failures['model'].unique():
                model_failures = detection_failures[detection_failures['model'] == model]
                print(f"    {model}: {len(model_failures)} failures")

        # Tempi di inferenza medi
        print(f"\n⏱️  TEMPI DI INFERENZA MEDI (per immagine):")
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            avg_time = model_df['inference_time_ms'].mean()
            print(f"  {model:50s}: {avg_time:6.2f} ms")

    def save_to_csv(self, output_path: str):
        """Salva risultati in CSV"""
        df = self.get_dataframe()
        df.to_csv(output_path, index=False)
        print(f"\n💾 Risultati salvati in: {output_path}")

    def plot_results_table(self, output_path: str):
        """Crea e salva tabella visuale con ✓/✗"""
        df = self.get_dataframe()

        # Crea pivot table: righe=immagini, colonne=modelli
        pivot = df.pivot_table(
            index=['image', 'true_label'],
            columns='model',
            values='correct',
            aggfunc='first'
        )

        # Converti True/False in ✓/✗
        pivot_display = pivot.applymap(
            lambda x: '✓' if x else '✗' if pd.notna(x) else '-')

        # Crea figura
        fig, ax = plt.subplots(figsize=(16, max(8, len(pivot) * 0.3)))
        ax.axis('tight')
        ax.axis('off')

        # Colora celle
        cell_colors = []
        for idx, row in pivot.iterrows():
            row_colors = []
            for val in row:
                if pd.isna(val):
                    row_colors.append('#CCCCCC')  # Grigio per NA
                elif val:
                    row_colors.append('#90EE90')  # Verde per corretto
                else:
                    row_colors.append('#FFB6C6')  # Rosso per sbagliato
            cell_colors.append(row_colors)

        # Crea tabella
        table = ax.table(
            cellText=pivot_display.values,
            rowLabels=[f"{img}\n({label})" for img, label in pivot.index],
            colLabels=pivot.columns,
            cellColours=cell_colors,
            cellLoc='center',
            loc='center',
            colWidths=[0.15] * len(pivot.columns)
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)

        # Stile header
        for i in range(len(pivot.columns)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Stile row labels
        for i in range(1, len(pivot) + 1):
            table[(i, -1)].set_facecolor('#D9E1F2')
            table[(i, -1)].set_text_props(weight='bold')

        plt.title('Risultati Testing Modelli di Face Recognition\n(✓ = Corretto, ✗ = Sbagliato)',
                  fontsize=14, weight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 Tabella risultati salvata in: {output_path}")

    def plot_confusion_matrix(self, model_name: str, output_dir: str):
        """Crea confusion matrix per un modello specifico"""
        df = self.get_dataframe()
        model_df = df[df['model'] == model_name]
        model_df = model_df[model_df['detection_success']
                            == True]  # Solo detection riuscite

        if len(model_df) == 0:
            print(f"⚠️  Nessun dato disponibile per {model_name}")
            return

        # Get unique labels
        all_labels = sorted(set(model_df['true_label'].unique()) | set(
            model_df['predicted'].unique()))

        # Crea confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(
            model_df['true_label'],
            model_df['predicted'],
            labels=all_labels
        )

        # Plot
        plt.figure(figsize=(max(10, len(all_labels)), max(8, len(all_labels))))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=all_labels,
            yticklabels=all_labels,
            cbar_kws={'label': 'Count'}
        )
        plt.title(f'Confusion Matrix - {model_name}',
                  fontsize=14, weight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Salva
        safe_name = model_name.replace(
            ' ', '_').replace('(', '').replace(')', '')
        output_path = os.path.join(
            output_dir, f'confusion_matrix_{safe_name}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  📊 Confusion matrix salvata: {output_path}")

    def plot_accuracy_comparison(self, output_path: str):
        """Crea bar chart comparativo delle accuracy"""
        accuracies = self.get_accuracy_per_model()

        # Ordina per accuracy
        sorted_models = sorted(
            accuracies.items(), key=lambda x: x[1], reverse=True)
        models, accs = zip(*sorted_models)

        # Plot
        plt.figure(figsize=(12, 6))
        bars = plt.barh(range(len(models)), accs, color='steelblue')

        # Colora barre in base all'accuracy
        for i, (bar, acc) in enumerate(zip(bars, accs)):
            if acc >= 90:
                bar.set_color('#2E7D32')  # Verde scuro
            elif acc >= 75:
                bar.set_color('#66BB6A')  # Verde chiaro
            elif acc >= 50:
                bar.set_color('#FFA726')  # Arancione
            else:
                bar.set_color('#EF5350')  # Rosso

            # Aggiungi valore
            plt.text(acc + 1, i, f'{acc:.1f}%', va='center', fontsize=10)

        plt.yticks(range(len(models)), models)
        plt.xlabel('Accuracy (%)', fontsize=12)
        plt.title('Confronto Accuracy tra Modelli',
                  fontsize=14, weight='bold', pad=20)
        plt.xlim(0, 105)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 Grafico accuracy salvato in: {output_path}")


def get_ground_truth_from_folder(image_path: str, test_dir: str) -> str:
    """
    Estrae il ground truth label dal nome della cartella.

    Args:
        image_path: path completo all'immagine
        test_dir: directory base dei test

    Returns:
        nome della persona (nome della cartella parent)
    """
    image_path = Path(image_path)
    test_dir = Path(test_dir)

    # La cartella parent è il nome della persona
    person_folder = image_path.parent.name

    return person_folder


def load_test_images(test_dir: str) -> List[Tuple[str, str]]:
    """
    Carica tutte le immagini di test con i loro ground truth labels.

    Args:
        test_dir: directory con sottocartelle per persona

    Returns:
        lista di tuple (image_path, true_label)
    """
    test_dir = Path(test_dir)
    test_images = []

    # Estensioni supportate
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    # Scandisci tutte le sottocartelle
    for person_folder in test_dir.iterdir():
        if not person_folder.is_dir():
            continue

        person_name = person_folder.name

        # Trova tutte le immagini nella cartella
        for img_file in person_folder.iterdir():
            if img_file.suffix.lower() in image_extensions:
                test_images.append((str(img_file), person_name))

    print(
        f"\n📁 Caricate {len(test_images)} immagini di test da {len(set(t[1] for t in test_images))} persone")

    return test_images


def save_wrong_predictions(results: ModelTestResults, test_images_info: List[Tuple[str, str]],
                           detector, output_dir: str):
    """
    Salva immagini con predizioni sbagliate per analisi visuale.

    Args:
        results: oggetto con tutti i risultati
        test_images_info: lista di (image_path, true_label)
        detector: detector per ridisegnare bbox
        output_dir: directory dove salvare le immagini
    """
    df = results.get_dataframe()
    wrong_predictions = df[df['correct'] == False]

    if len(wrong_predictions) == 0:
        print("\n✅ Nessuna predizione sbagliata! Perfetto!")
        return

    wrong_dir = Path(output_dir) / "wrong_predictions"
    wrong_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📸 Salvando {len(wrong_predictions)} predizioni sbagliate...")

    for _, row in wrong_predictions.iterrows():
        # Trova il path dell'immagine
        img_path = next((path for path, label in test_images_info
                        if Path(path).name == row['image'] and label == row['true_label']), None)

        if img_path is None:
            continue

        # Carica immagine
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        # Disegna label con predizione sbagliata
        text = f"TRUE: {row['true_label']} | PRED: {row['predicted']} ({row['confidence']:.2f})"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

        # Salva
        safe_model = row['model'].replace(
            ' ', '_').replace('(', '').replace(')', '')
        out_name = f"{Path(img_path).stem}_{safe_model}_WRONG.jpg"
        out_path = wrong_dir / out_name
        cv2.imwrite(str(out_path), frame)

    print(f"  💾 Salvate in: {wrong_dir}")


def test_all_models(test_images_dir: str, dataset_dir: str, yolo_model_path: str,
                    models_to_test: List[str] = None, threshold: float = 0.60):
    """
    Testa tutti i modelli specificati.

    Args:
        test_images_dir: directory con immagini di test (struttura a sottocartelle)
        dataset_dir: directory con dataset di reference
        yolo_model_path: path al modello YOLO detector
        models_to_test: lista di nomi modelli da testare (None = tutti)
        threshold: soglia di confidence per recognition
    """
    import time

    # Setup
    if models_to_test is None:
        models_to_test = list(AVAILABLE_MODELS.keys())

    # Rimuovi modelli "not working"
    models_to_test = [
        m for m in models_to_test if "not working" not in m.lower()]

    print("\n" + "="*80)
    print("🧪 INIZIO TESTING MODELLI DI FACE RECOGNITION")
    print("="*80)
    print(f"📁 Test images dir: {test_images_dir}")
    print(f"📁 Dataset dir: {dataset_dir}")
    print(f"🎯 Modelli da testare: {len(models_to_test)}")
    print(f"🎚️  Threshold: {threshold}")

    # Carica immagini di test
    test_images = load_test_images(test_images_dir)
    if len(test_images) == 0:
        print("❌ Nessuna immagine di test trovata!")
        return

    # Inizializza risultati
    results = ModelTestResults()

    # Inizializza detector (uno solo per tutti i modelli)
    print(f"\n🔧 Inizializzo detector...")
    from src.detector import FaceDetector
    detector = FaceDetector(model_path=yolo_model_path)

    # Testa ogni modello
    for model_idx, model_name in enumerate(models_to_test, 1):
        print(f"\n{'='*80}")
        print(f"🔄 [{model_idx}/{len(models_to_test)}] Testing: {model_name}")
        print(f"{'='*80}")

        try:
            # Inizializza recognizer per questo modello
            from src.recognizer import FaceRecognizer
            from src.models_recognition import create_model

            config = AVAILABLE_MODELS[model_name]
            model_info = create_model(
                backbone_type=config["backbone_type"],
                checkpoint_path=str(config["model_path"])
            )

            recognizer = FaceRecognizer(
                model=model_info['model'],
                model_path=str(config["model_path"]),
                image_size=model_info['image_size'],
                embedding_size=model_info['embedding_size'],
                use_get_embedding=model_info['use_get_embedding']
            )

            # Carica embeddings del dataset per questo modello
            embeddings_array, labels_list = load_dataset_embeddings(
                dataset_dir, recognizer=recognizer, model_name=model_name
            )

            # Testa su tutte le immagini
            for img_path, true_label in tqdm(test_images, desc=f"  Testing {model_name}"):
                frame = cv2.imread(img_path)
                if frame is None:
                    continue

                # Misura tempo di inferenza
                start_time = time.time()

                # Recognition
                face_results = recognize_faces(
                    frame, detector, recognizer,
                    embeddings_array, labels_list,
                    threshold=threshold
                )

                inference_time = time.time() - start_time

                # Gestisci risultati
                if len(face_results) == 0:
                    # Nessuna faccia rilevata
                    results.add_result(
                        image_path=img_path,
                        model_name=model_name,
                        true_label=true_label,
                        predicted_label="NO_DETECTION",
                        confidence=0.0,
                        detection_success=False,
                        inference_time=inference_time
                    )
                else:
                    # Prendi la faccia più grande (o la prima se ce n'è una sola)
                    best_result = max(face_results,
                                      key=lambda r: (r['bbox'][2] - r['bbox'][0]) * (r['bbox'][3] - r['bbox'][1]))

                    results.add_result(
                        image_path=img_path,
                        model_name=model_name,
                        true_label=true_label,
                        predicted_label=best_result['name'],
                        confidence=best_result['confidence'],
                        detection_success=True,
                        inference_time=inference_time
                    )

            # Cleanup
            recognizer.close()

        except Exception as e:
            print(f"❌ Errore durante test di {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Cleanup detector
    detector.close()

    # Genera output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = BASE_DIR / "test_results" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("📊 GENERAZIONE OUTPUT")
    print(f"{'='*80}")

    # Summary testuale
    results.print_summary()

    # Salva CSV
    results.save_to_csv(str(output_dir / "results.csv"))

    # Tabella visuale
    results.plot_results_table(str(output_dir / "results_table.png"))

    # Grafici accuracy
    results.plot_accuracy_comparison(
        str(output_dir / "accuracy_comparison.png"))

    # Confusion matrices
    print(f"\n📊 Generando confusion matrices...")
    cm_dir = output_dir / "confusion_matrices"
    cm_dir.mkdir(exist_ok=True)
    for model_name in models_to_test:
        results.plot_confusion_matrix(model_name, str(cm_dir))

    # Salva predizioni sbagliate
    save_wrong_predictions(results, test_images, detector, str(output_dir))

    print(f"\n{'='*80}")
    print(f"✅ TEST COMPLETATO!")
    print(f"📁 Tutti i risultati salvati in: {output_dir}")
    print(f"{'='*80}\n")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Test sistematico modelli face recognition')
    parser.add_argument('--test-dir', type=str,
                        default=str(TEST_IMAGES_DIR),
                        help='Directory con immagini di test (struttura a sottocartelle)')
    parser.add_argument('--dataset-dir', type=str,
                        default=str(DATASET_DIR),
                        help='Directory dataset di reference')
    parser.add_argument('--threshold', type=float, default=0.60,
                        help='Soglia di confidence (default: 0.60)')
    parser.add_argument('--models', nargs='+',
                        help='Lista modelli da testare (default: tutti)')

    args = parser.parse_args()

    # Verifica che test_dir esista
    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        print(f"❌ Directory test non trovata: {test_dir}")
        print(f"\n💡 Crea la struttura:")
        print(f"   {test_dir}/")
        print(f"   ├── persona1/")
        print(f"   │   ├── img1.jpg")
        print(f"   │   └── img2.jpg")
        print(f"   └── persona2/")
        print(f"       └── img1.jpg")
        return

    # Run test
    test_all_models(
        test_images_dir=str(test_dir),
        dataset_dir=args.dataset_dir,
        yolo_model_path=str(DETECTOR_MODEL_PATH),
        models_to_test=args.models,
        threshold=args.threshold
    )


if __name__ == "__main__":
    main()
