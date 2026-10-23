"""
report_utils.py — Persiste resultados numéricos de scripts de diagnóstico
em arquivos JSON, dentro de Outputs/Reports/.

Evita perder métricas calculadas (ex: B_wfa_limit, ângulos de cruzamento,
erros percentuais) quando o terminal é fechado, sem precisar rodar o
script de novo só para recuperar um número.
"""

import json
from datetime import datetime, timezone

from Src.config import REPORTS_DIR


def save_report(name, data):
    """
    Salva um dicionário de resultados como JSON em Outputs/Reports/<name>.json.

    Parâmetros
    ----------
    name : str — nome do relatório (sem extensão), ex: "cel7_validacao_wfa"
    data : dict — valores a salvar (numpy floats/ints são convertidos
           automaticamente para tipos nativos do Python)
    """
    def _default(obj):
        if hasattr(obj, "item"):
            return obj.item()
        raise TypeError(f"Tipo não serializável: {type(obj)}")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }

    out_path = REPORTS_DIR / f"{name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_default)

    return out_path


def load_report(name):
    """Carrega um relatório salvo anteriormente. Retorna o dict 'data'."""
    in_path = REPORTS_DIR / f"{name}.json"
    with open(in_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["data"]