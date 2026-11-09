# Оценка любой системы протоколом CareFlow (для команды)

Одна команда: ответы вашей системы и опорной системы на одних и тех же контекстах → вердикты судьи в обоих
порядках показа, сырой win-rate и win-rate с поправкой на длину (с 95% CI), вердикт эквивалентности (TOST) и
разбивка по 6 категориям HEART. Судья — Gemma-4-26B-A4B, рубрика HEART дословно (валидация: `docs/data.md`, E2).

## 1. Взять общие контексты

```bash
uv run python scripts/run_experiment.py --config configs/experiments/export_contexts.yaml \
  data.split=test data.n_contexts=1000          # → outputs/export_contexts/contexts.jsonl
```

Строка: `{"ctx", "history", "speakers", "emotion", "human"}`. Это ровно те контексты ESConv, на которых оценены
наши 9 систем (тот же seed), поэтому ваши результаты сразу сравнимы с таблицами `docs/results.md`.
Свои диалоги (например, FailedESConv) тоже подходят — формат тот же.

## 2. Сгенерировать ответы и сложить в один JSONL

Одна строка = один ответ одной системы:

```json
{"ctx": "esconv-0789", "history": ["...", "..."], "speakers": ["seeker", "supporter"], "emotion": "anxiety",
 "system": "my_router", "text": "ответ системы"}
```

В файле должны быть ответы **обеих** систем (вашей и опорной) на одних и тех же `ctx`. Опорная по умолчанию —
`single_9b`; её ответы на наших контекстах лежат в `outputs/router_data_test/traces.jsonl` (поле `text`).

## 3. Запустить оценку

```bash
uv run python scripts/run_experiment.py --config configs/experiments/evaluate_replies.yaml \
  replies_file=my_replies.jsonl system=my_router reference=single_9b experiment.name=eval_my_router
```

Результат: `outputs/eval_my_router/report.json` + все вердикты в `judgments.jsonl`, прогон в MLflow
(эксперимент `careflow-team-eval`). Все вызовы судьи кэшируются: повторный запуск бесплатен и бит-в-бит.

## 4. Как читать отчёт

| Поле | Что значит |
|---|---|
| `raw.winrate` [CI] | доля побед над опорной (0.5 — на равных), бутстрэп по контекстам |
| `lc.winrate` [CI] | то же при равной длине ответов — **главная цифра**: судья сильно любит длинные ответы |
| `lc.verdict@0.1` | `better` / `worse` / `equivalent` (в пределах ±0.10) / `inconclusive` (данных мало) |
| `order_consistency` | доля контекстов, где вердикт не зависит от порядка показа (норма ≈ 0.8–0.9) |
| `mean_words_*`, `length_beta` | длина ответов и сила смещения на длину |
| `cat.*` | win-rate по категориям HEART |

**Правила, которые мы выучили на своих данных:** не сравнивать с ответами людей ESConv (все системы выигрывают
92–98%); всегда смотреть `lc.*`, а не `raw.*`; `inconclusive` ≠ «равны» — для ±0.10 нужно ≈ 175–300 контекстов
(`docs/results.md`, раздел «Эквивалентность»).
