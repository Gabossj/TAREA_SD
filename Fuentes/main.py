import os
import sys
import io
import re
from contextlib import redirect_stdout

# Asegurar que Python vea los scripts en el mismo directorio
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from rdim import main as main_rdim
from trn import main as main_trn
from tst import main as main_tst


# ------------------- PARSER DE SALIDA tst.py -------------------
def parse_metrics(output_text):
    """
    Extrae TP, FP, FN, TN, F1-scores y Accuracy desde la salida de tst.py.
    """

    def extract(pattern):
        match = re.search(pattern, output_text)
        return float(match.group(1)) if match else None

    return {
        "TP": extract(r"TP\): (\d+)"),
        "FP": extract(r"FP\): (\d+)"),
        "FN": extract(r"FN\): (\d+)"),
        "TN": extract(r"TN\): (\d+)"),
        "F1_1": extract(r"Clase 1.*?([0-9\.]+)"),
        "F1_2": extract(r"Clase 2.*?([0-9\.]+)"),
        "F1_avg": extract(r"Promedio.*?([0-9\.]+)"),
        "Accuracy": extract(r"Accuracy:\s*([0-9\.]+)")
    }


# -------------------- MAIN MULTI-RUN ----------------------------
def main():

    NUM_RUNS = 5
    results = []

    print("\n" + "="*80)
    print("        EJECUTANDO PIPELINE COMPLETO (5 EJECUCIONES)")
    print("="*80)

    for i in range(NUM_RUNS):
        print(f"\n>>> EJECUCIÓN {i+1}/{NUM_RUNS}")

        main_rdim()
        main_trn()

        # Capturar SOLO la salida del test
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            main_tst()

        output = buffer.getvalue()
        metrics = parse_metrics(output)
        results.append(metrics)

    # ------------------ PROMEDIOS FINALES ------------------
    print("\n" + "="*80)
    print("                PROMEDIO FINAL DE 5 EJECUCIONES")
    print("="*80)

    # Calcular promedio por cada métrica
    avg = {k: sum(r[k] for r in results) / NUM_RUNS for k in results[0]}

    print("\n--- MATRIZ DE CONFUSIÓN PROMEDIADA ---")
    print(f"TP promedio: {avg['TP']:.0f}")
    print(f"FP promedio: {avg['FP']:.0f}")
    print(f"FN promedio: {avg['FN']:.0f}")
    print(f"TN promedio: {avg['TN']:.0f}")

    print("\n--- F-SCORES PROMEDIO ---")
    print(f"Clase 1 (F1): {avg['F1_1']:.4f}")
    print(f"Clase 2 (F1): {avg['F1_2']:.4f}")
    print(f"Promedio F1:  {avg['F1_avg']:.4f}")

    print("\n--- EXACTITUD PROMEDIO ---")
    print(f"Accuracy promedio: {avg['Accuracy']:.4f} ({avg['Accuracy']*100:.2f}%)")

    print("\n" + "="*80)
    print("              PIPELINE MULTI-RUN COMPLETADO")
    print("="*80)


if __name__ == "__main__":
    main()
