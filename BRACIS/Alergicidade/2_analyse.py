# -*- coding: utf-8 -*-
"""
Script para Análise Exploratória de Dados (EDA) de arquivos de sequência de proteína.
"""
from __future__ import annotations

import re
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# --- Configuração ---
# Certifique-se que estes nomes de arquivo correspondem aos usados no seu script de treino
POS_TRAIN_FILE = Path("simalergenico.txt")
NEG_TRAIN_FILE = Path("naoalergenico.txt")
POS_TEST_FILE  = Path("test_sim.txt")
NEG_TEST_FILE  = Path("test_nao.txt")

# Diretório para salvar os gráficos
OUTPUT_DIR = Path("analise_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Aminoácidos Padrão + caracteres comuns de ambiguidade/não padrão aceitáveis (opcional)
# STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
# Ou uma definição mais ampla incluindo ambiguidades comuns:
VALID_CHARS_REGEX = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYXBZJUO*]+$", re.IGNORECASE) # Permite letras comuns + XBZJUO*
STANDARD_AA_SET = set("ACDEFGHIKLMNPQRSTVWY")

# --- Funções Auxiliares ---

def read_sequences(filepath: Path) -> list[str]:
    """Lê sequências de um arquivo, uma por linha."""
    sequences = []
    if not filepath.is_file():
        print(f"AVISO: Arquivo não encontrado: {filepath}")
        return sequences
    with open(filepath, 'r') as f:
        for line in f:
            seq = line.strip().upper() # Padroniza para maiúsculas
            if seq:
                sequences.append(seq)
    return sequences

def analyze_sequences(sequences: list[str], label: str) -> dict:
    """Realiza análises básicas em uma lista de sequências."""
    results = {"label": label}
    num_sequences = len(sequences)
    results["num_sequences"] = num_sequences

    if num_sequences == 0:
        results["lengths"] = []
        results["length_stats"] = {"min": 0, "max": 0, "mean": 0, "median": 0, "std": 0}
        results["invalid_char_sequences"] = []
        results["num_duplicates"] = 0
        results["unique_sequences"] = 0
        results["aa_composition"] = Counter()
        return results

    # Comprimentos
    lengths = [len(s) for s in sequences]
    results["lengths"] = lengths
    results["length_stats"] = {
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "std": float(np.std(lengths)),
    }

    # Qualidade: Caracteres Inválidos
    invalid_char_sequences = []
    all_chars = Counter()
    valid_aa_counter = Counter()

    for i, seq in enumerate(sequences):
        # Verifica se *todos* os caracteres são válidos segundo a regex
        if not VALID_CHARS_REGEX.match(seq):
             # Encontra os caracteres específicos que não são válidos
             invalid_chars_found = set(c for c in seq if not VALID_CHARS_REGEX.match(c))
             invalid_char_sequences.append({"index": i, "sequence_preview": seq[:30]+"...", "invalid_chars": list(invalid_chars_found)})

        # Conta todos os caracteres para composição geral
        all_chars.update(seq)
        # Conta apenas os 20 AAs padrão para composição padrão
        valid_aa_counter.update(c for c in seq if c in STANDARD_AA_SET)

    results["invalid_char_sequences"] = invalid_char_sequences
    results["all_char_counts"] = dict(all_chars.most_common()) # Composição incluindo não-padrão

    # Composição de Aminoácidos Padrão (Normalizada)
    total_standard_aas = sum(valid_aa_counter.values())
    aa_composition_norm = {
        aa: (valid_aa_counter[aa] / total_standard_aas if total_standard_aas > 0 else 0)
        for aa in sorted(STANDARD_AA_SET)
    }
    results["aa_composition_norm"] = aa_composition_norm
    results["total_standard_aas"] = total_standard_aas

    # Duplicatas
    unique_sequences = set(sequences)
    results["num_duplicates"] = num_sequences - len(unique_sequences)
    results["unique_sequences"] = len(unique_sequences)

    return results

def plot_length_distribution(analysis_results: list[dict], filename: str):
    """Plota histogramas de distribuição de comprimentos."""
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(analysis_results))) # Cores diferentes
    max_len = 0
    labels = []

    for i, result in enumerate(analysis_results):
        if result["lengths"]:
            plt.hist(result["lengths"], bins=50, alpha=0.6, label=f"{result['label']} (n={result['num_sequences']})", color=colors[i])
            max_len = max(max_len, result["length_stats"]["max"])
            labels.append(result['label'])

    if not labels: # Se nenhum dado foi carregado
        print("Nenhum dado para plotar distribuição de comprimentos.")
        plt.close()
        return

    plt.title("Distribuição do Comprimento das Sequências")
    plt.xlabel("Comprimento da Sequência")
    plt.ylabel("Frequência")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Ajusta limite x se houver sequencias muito longas, talvez log scale ou limitar
    plt.xlim(0, max_len + 50) # Ajuste conforme necessário
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    print(f"Gráfico de distribuição de comprimentos salvo em: {OUTPUT_DIR / filename}")
    plt.close()

def plot_aa_composition(analysis_results: list[dict], filename: str):
    """Plota gráfico de barras comparando a composição de aminoácidos."""
    labels = [res["label"] for res in analysis_results if res["num_sequences"] > 0]
    if len(labels) < 2:
         print("Não há dados suficientes para comparar composição de aminoácidos.")
         return

    # Pega os dados de composição normalizada dos dois primeiros resultados válidos
    comp1 = analysis_results[0]["aa_composition_norm"]
    comp2 = analysis_results[1]["aa_composition_norm"]
    label1 = analysis_results[0]["label"]
    label2 = analysis_results[1]["label"]

    aas = sorted(STANDARD_AA_SET)
    freq1 = [comp1.get(aa, 0) for aa in aas]
    freq2 = [comp2.get(aa, 0) for aa in aas]

    x = np.arange(len(aas))
    width = 0.35

    fig, ax = plt.subplots(figsize=(15, 7))
    rects1 = ax.bar(x - width/2, freq1, width, label=label1)
    rects2 = ax.bar(x + width/2, freq2, width, label=label2)

    ax.set_ylabel('Frequência Relativa')
    ax.set_title(f'Composição de Aminoácidos Padrão ({label1} vs {label2})')
    ax.set_xticks(x)
    ax.set_xticklabels(aas)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    print(f"Gráfico de composição de aminoácidos salvo em: {OUTPUT_DIR / filename}")
    plt.close()


# --- Execução Principal ---

if __name__ == "__main__":
    print("Iniciando análise exploratória dos dados...")

    # Carregar Sequências
    print("\n--- Carregando Sequências ---")
    pos_train_seqs = read_sequences(POS_TRAIN_FILE)
    neg_train_seqs = read_sequences(NEG_TRAIN_FILE)
    pos_test_seqs  = read_sequences(POS_TEST_FILE)
    neg_test_seqs  = read_sequences(NEG_TEST_FILE)

    # Analisar cada conjunto
    print("\n--- Analisando Conjuntos ---")
    analysis = []
    data_map = {
        "Treino Positivo": pos_train_seqs,
        "Treino Negativo": neg_train_seqs,
        "Teste Positivo": pos_test_seqs,
        "Teste Negativo": neg_test_seqs,
    }
    for label, seqs in data_map.items():
         print(f"\nAnalisando: {label}")
         res = analyze_sequences(seqs, label)
         analysis.append(res)

         print(f"  Número de Sequências: {res['num_sequences']}")
         if res['num_sequences'] > 0:
             print(f"  Comprimento (Min/Média/Mediana/Max/StdDev): "
                   f"{res['length_stats']['min']:.0f} / "
                   f"{res['length_stats']['mean']:.1f} / "
                   f"{res['length_stats']['median']:.0f} / "
                   f"{res['length_stats']['max']:.0f} / "
                   f"{res['length_stats']['std']:.1f}")
             print(f"  Sequências Únicas: {res['unique_sequences']} ({res['num_duplicates']} duplicatas internas)")
             print(f"  Total de AAs Padrão Contados: {res['total_standard_aas']}")
             if res['invalid_char_sequences']:
                 print(f"  AVISO: Encontradas {len(res['invalid_char_sequences'])} sequências com caracteres não mapeados pela regex:")
                 for inv in res['invalid_char_sequences'][:5]: # Mostra as 5 primeiras
                     print(f"    - Índice {inv['index']}: {inv['sequence_preview']} (Inválidos: {inv['invalid_chars']})")
             # Opcional: print da composição
             # print(f"  Composição AA Padrão (%):")
             # for aa, freq in res['aa_composition_norm'].items():
             #      print(f"    {aa}: {freq:.4f}")

    # Análise de Contaminação Treino/Teste
    print("\n--- Verificando Contaminação Treino/Teste ---")
    train_seqs_set = set(pos_train_seqs) | set(neg_train_seqs)
    test_seqs_set = set(pos_test_seqs) | set(neg_test_seqs)
    contamination = train_seqs_set.intersection(test_seqs_set)
    num_contamination = len(contamination)
    if num_contamination > 0:
        print(f"AVISO: {num_contamination} sequências estão presentes TANTO no treino QUANTO no teste!")
        # for i, seq in enumerate(list(contamination)[:10]): # Mostra as 10 primeiras
        #     print(f"  - Contaminante {i+1}: {seq[:60]}...")
    else:
        print("OK: Nenhuma sequência encontrada em comum entre treino e teste.")

    # Gerar Gráficos
    print("\n--- Gerando Gráficos ---")
    # Separa resultados de treino e teste para plots separados
    train_results = [res for res in analysis if "Treino" in res["label"]]
    test_results = [res for res in analysis if "Teste" in res["label"]]

    if any(res["num_sequences"] > 0 for res in train_results):
        plot_length_distribution(train_results, "distribuicao_comprimento_treino.png")
        plot_aa_composition(train_results, "composicao_aa_treino.png")
    else:
        print("Sem dados de treino para plotar.")

    if any(res["num_sequences"] > 0 for res in test_results):
        plot_length_distribution(test_results, "distribuicao_comprimento_teste.png")
        # Não plotamos composição para teste usualmente, mas poderíamos se quisessemos comparar teste pos/neg
        # plot_aa_composition([res for res in test_results if "Positivo" in res["label"]],
        #                     [res for res in test_results if "Negativo" in res["label"]],
        #                     "composicao_aa_teste.png")
    else:
        print("Sem dados de teste para plotar.")

    print("\nAnálise concluída.")
    print(f"Resultados e gráficos salvos em: {OUTPUT_DIR}")