# References

Статусы (последняя проверка 2026-09-28):
- ✅ **verified**: метаданные и ключевые утверждения сверены с первоисточником (arXiv/ar5iv или Crossref).
- ⚠️ **partial**: работа существует, но часть утверждений не сверена (полный текст за paywall) или есть оговорки о переносимости.
- ❌ **unresolved**: первоисточник не найден. Ссылаться нельзя.

Пометка **[польз.]** означает, что деталь взята из описания пользователя и по первоисточнику не проверена.

---

## 1. Роутинг и confidence learning

### ✅ SATER — arXiv:2510.05164
Shen, Liu, Huang, Yin, Zheng, Huang. *SATER: A Self-Aware and Token-Efficient Approach to Routing and Cascading.* 2025-10-04.
- **Идея.** Этап I, shortest-response preference optimization. Положительный пример — самый короткий верный ответ. Отрицательный — самый длинный неверный, причём длиннее положительного не менее чем в 1.5 раза. Лосс DPO (β=1) + SFT (λ=0.2).
  Этап II, confidence-aware rejection. Каждый вопрос семплируется 10 раз, считается доля верных ответов. Для порогов 0.1…1.0 в обучающую выборку идёт верный ответ, если доля выше порога, иначе шаблон отказа «Sorry, I can't answer that.».
- **Метрики.** Для pre-routing: **ToA** (площадь под кривой cost–performance между точками «только SLM» и «только LLM», пороги с шагом 0.1) и **ToGR** (ToA текущего роутинга, нормированная на golden routing). Для каскадов: **AGL** (латентность, когда ответ завершает SLM; прокси — число выходных токенов) и **AROL** (дополнительная латентность, когда SLM отказалась и пришлось звать LLM, по сравнению с прямым вызовом LLM).
- **Что даёт проекту.** AGL/AROL не зависят от железа и сети, поэтому их можно сравнивать между запусками на MacBook и по API. ToGR подходит для сравнения наших роутеров с oracle gate. Refusal training можно применить к агентам MAS: SLM-агент явно отказывается, и тогда вызывается тяжёлый агент.
- **Ограничения.** SLM: Llama-3.1-8B, Qwen2.5-7B/3B. Данные: MMLU, ARC-C/E, GSM8K, MATH-500, ReClor, то есть задачи с проверяемым ответом. В ESC золотого ответа нет, поэтому «доля верных» придётся заменить оценкой judge или HEART и порогом. Это наша адаптация, и она вносит шум judge в целевую переменную. Наши 1.5B меньше самых маленьких SLM в статье.

### ✅ GCUL — arXiv:2609.29327
Fan, Zhang, Chen. *GCUL: Ambiguity Identification in Text Emotion Classification via Cluster-Guided Learning.* 2026-09-24.
- **Идея.** Неоднозначные и ошибочно классифицированные примеры образуют компактные кластеры в пространстве представлений. Три фазы: базовый классификатор → LDA + K-Means для поиска кластера неоднозначности → переобучение с классом *uncertain*. DistilBERT: 89.37 → 94.98% accuracy при отказе менее чем на 9% примеров.
- **Что даёт проекту.** Отказ ищется геометрически, по кластерам неоднозначности, а не по порогу confidence. Selectivity score и формальные условия позволяют проверить пользу отказа до развёртывания.
- **Ограничения.** Это классификация эмоций на уровне реплики. Перенос на решение «эскалировать ли к тяжёлому агенту» — наша гипотеза.

### ⚠️ TAD-CL — *Teacher-Anchored, Disagreement-Weighted Confidence Learning*
Peng Li, Yang Pei. Есть только PDF у пользователя, в открытом поиске не находится.
- **[польз.]** Ключевая проблема каскадов — overtrust: SLM уверена, но ошибается. Confidence обучается на teacher-anchored retainability target с односторонним весом (one-sided disagreement weight) на positive overtrust gap. ECE 0.230 → 0.061, bypass rate 34 → 12.3%.
- **Нужно:** положить PDF в `docs/papers/` (в git не попадает, см. `.gitignore`), чтобы сверить цифры, площадку и год.
- **Сравнение с SATER.** Оба подхода учат SLM понимать, когда ответ ненадёжен, но с разных сторон. SATER учит модель *генерировать* отказ, и цель задаётся порогом по самосогласованности (доля верных из 10 семплов). TAD-CL калибрует *скаляр* confidence относительно учителя и штрафует только завышение. Противоречие: сигнал SATER — собственная точность SLM, которая на уверенно-ошибочных примерах может быть стабильно высокой и неверной. Это как раз тот случай overtrust, на который нацелен TAD-CL. Как разрешить: на одном сплите ESConv сравнить три варианта: SATER-отказ, TAD-CL-confidence и их комбинацию (SATER-модель плюс TAD-CL-голова поверх её hidden state). Критерии: ECE, bypass rate, ToGR.

### ✅ AIMS — DOI 10.1145/3767295.3803622
Liu, Shen, Che, Ghandi, Li. *AIMS: Cost-Efficient LLM-Based Agent Deployment in Hybrid Cloud-Edge Environments.* EuroSys 2026.
- **Идея.** Лёгкий планировщик разбивает запрос агента на подзадачи и распределяет каждую между локальной SLM и облачной LLM. На SLM уходит 83.4% подзадач при сопоставимой точности. До +27.5% accuracy и +31.4% использования SLM против HybridLLM.
- **Что даёт проекту.** Маршрутизация по агентам MAS, а не по запросу целиком. Это ровно наш гибридный профиль: анализ контекста и safety локально, генерация по API.
- **Ограничения.** Домен — агентные задачи, не ESC. Полный текст не читали: ACM DL отдаёт 403, PDF в репозитории нет. Цифры взяты из аннотации.

### ⚠️ Selection Effect — submission #2935 (PDF от пользователя; прочитаны основной текст и прил. A–C, прил. D — выборочно)
Tikhomirov, Savchenko (Central University, HSE, Sber AI Lab). *The Selection Effect: Why Compact LLM Verifiers for Emotion Classification Fail Behind Encoder Confidence Gates.* PDF `2935_The_Selection_Effect_Why_ (2).pdf`, собран 2026-08-29, 20 стр.; код: github.com/tikhomirovd/selective-emotion-classification. «2935» — номер подачи, не arXiv ID; публикация/площадка не подтверждены.
- **Задача:** закрытая классификация эмоций (Ekman-7) на GoEmotions, EmpatheticDialogues (ситуационные метки), русском CEDR; контроль — AG News. Метрика — macro-F1 по золотым меткам.
- **Метод:** encoder-first routing: RoBERTa/DeBERTa/XLM-R предсказывает, неуверенные примеры (margin / max-prob / entropy, 6 альтернативных гейтов, error-oracle) отправляются верификатору или в abstain; NSGA-II-поиск по порогам и 7 верификаторам (Qwen3.5-4B zero-shot, LoRA-Qwen, голосование, self-consistency, MultiAgentESC-style, гетерогенные роли, второй энкодер); split-conformal как альтернатива. Задержка — Apple Silicon/Ollama.
- **Вывод:** ни один компактный LLM-верификатор не улучшает фронт «качество/стоимость»; механизм — *selection effect*: гейт отправляет примеры, неоднозначные для самих людей-разметчиков, где энкодер и LLM ошибаются вместе. Масштаб 4B → 397B не помогает. Исключение — ED (энкодер перенесён zero-shot): там эскалация выигрывает (их положительный контроль).
- **Связь с нами:** тот же вопрос «окупается ли добавочное LLM-вычисление», но для **классификации**, не генерации ответа. Не пересекается с нашими RQ2 (MAS для генерации), протоколом оценки (судья, длина, потолок vs человек), sycophancy. Пересекается с отложенной частью этапа 7 (порог по confidence, conformal abstention для эмоций) — **её не делаем, ссылаемся**. Их ED-исключение согласуется с нашим E1 (классификаторы эмоций слабы на диалогах поддержки).
- **Ссылки из этой работы для нас (не проверены):** Xu et al. 2026, *Rethinking the value of multi-agent workflow: A strong single agent baseline*, arXiv:2601.12307; MultiAgentESC (Xu et al., EMNLP 2025, LLM-based multi-agent framework for ESC) — опубликованная MAS для ESC, стоит добавить как бейзлайн; RouteLLM (Ong et al., ICLR 2025); FrugalGPT (arXiv:2305.05176); DAAO (Su et al., WWW 2026, arXiv:2509.11079).

## 2. Топология MAS

### ✅ InsideOut — DOI 10.3233/FAIA241039
Mozikov, Severin, Glushanina, Baklashkin, Savchenko, Makarov. *InsideOut: Unifying Emotional LLMs to Foster Empathy.* ECAI 2024.
- **Идея.** По агенту на каждую базовую эмоцию Экмана плюс агрегатор, star-топология. GPT-4 и GigaChat на EmpatheticDialogues.
- **Что даёт проекту.** Главный baseline MAS.
- **Ограничения.** Оценка только на ED. Нет стоимости и латентности, нет sycophancy.

### ⚠️ UniESC — DOI 10.1109/TAFFC.2026.3697775
Wang, Tu, Jin, Li, Yang, Lu, Li, Xu. *UniESC: A Personality-Sensitive Multi-Agent Role-Playing for Emotional Support Conversation.* IEEE Trans. Affective Computing, 2026 (early access). Метаданные подтверждены через Crossref.
- **[польз.]** Theory-of-Mind Roundtable: агенты School Supporters разыгрывают дискуссию с учётом характеристик пользователя, супервизор выбирает ответ. Компактную UniChat файнтюнят на датасете UniConv.
- **Что даёт проекту.** Roundtable как альтернатива star-топологии InsideOut. UniChat — baseline дистиллированной модели.
- **Ограничения.** Аннотация на странице computer.org не отрисовалась (JS). Выложены ли UniConv и веса UniChat, неизвестно. Если нет, воспроизводим по описанию и так и пишем.

### ✅ PAIR — arXiv:2609.16344 (v2)
Quan, Ding, Yong, Zhang, Wang, Chin. *From Momentary Emotion Inference to Sustained Emotion Support: Evaluating a Companion Agent in a Longitudinal Study.* 2026-09-14.
- **Роутер.** Лёгкий, **rule-based**, по структуре события. **Cog** (cognitive appraisal) — для одиночных событий: отдельный инцидент, изолированный конфликт, ожидание. **Combined** (нейрокогнитивный) — для стека стрессоров, цепочек событий, связки тело–психика. Совпал с предпочтением большинства в 26/30 формативных сценариев (87%). В полевом исследовании Cog выбран в 87.5% сессий.
- **Память.** После вечерней сессии лёгкая модель сжимает день в структурированный блок до 150 слов: повторяющиеся темы, систематические ошибки оценки, коррекции пользователя, исходы стратегий. Блок подаётся в следующие вызовы appraisal. Коррекции запрашиваются при оценке понимания ≤4 и встретились в 4.0% сессий. Из 37: 15 про направление или интенсивность эмоции, 8 про контекст, 8 про ошибки памяти, 6 про несоответствие стратегии.
- **Оценка.** SAM по 9-балльной шкале (valence, arousal, dominance) до показа оценки системы (чтобы избежать якорения) и после guided conversation. 19 участников, 14 дней, 1093 сессии.
- **Что даёт проекту.** Rule-based роутер — нижняя граница, с которой сравниваем обучаемые. Схема блока памяти переносится почти один в один. Протокол SAM.
- **Ограничения.** n=19. Полевое исследование офлайн не воспроизводится, берём только протокол.

## 3. Дистилляция

### ⚠️ SpeechLLM CLS-head — arXiv:2609.20081
Watawana, Burdisso, Villatoro-Tello, K E, Hacioglu, Motlicek, Stolcke. *Reading Emotions in the Token Space: Discriminative Adaptation of SpeechLLMs for Emotion Recognition.* 2026-09-17.
- **Идея.** Линейная голова поверх hidden state последнего токена промпта. Один forward pass, нет меток вне заданного набора, классы интерпретируются как направления в пространстве токенов. Выигрыш по Macro F1 на IEMOCAP.
- **Что даёт проекту.** Шаблон компактного роутера и детектора sycophancy поверх текстовой LLM. Проекция весов головы на unembedding даёт интерпретацию через токены.
- **Ограничения.** Статья про речевые LLM. Перенос на текстовые решения роутинга — наша адаптация.

### ⚠️ MiAgent — DOI 10.1007/s00530-026-02385-9
Chu, Guo, Hong. *From disagreement to insight: multi-agent collaborative reasoning for multimodal emotional support conversations.* Multimedia Systems, 2026-06. Метаданные подтверждены через Crossref, полный текст за paywall.
- **[польз.]** Видео-, аудио- и текстовый агенты независимо анализируют эмоции. Affective reasoning agent разрешает кросс-модальные конфликты через conflict judging и reliability weighting. Supporter agent обучен через distillation-based reasoning.
- **Что даёт проекту.** Disagreement-aware reasoning: расхождение агентов анализа служит сигналом для роутера. Reliability weighting агентов. Шаблон дистилляции MAS в компактную модель.
- **Ограничения.** У нас одна модальность (текст), поэтому «конфликт модальностей» заменяем конфликтом между агентами анализа или стратегиями. Нужен PDF, чтобы сверить функцию reliability weighting.

## 4. Оценка

### ✅ HEART — arXiv:2601.19922
Iyer, Aggarwal, Koyejo, Heyman, Ong, Mukherjee. *HEART: A Unified Benchmark for Assessing Humans and LLMs in Emotional Support Dialogue.* v1 2026-01-09, v2 2026-02-25.
- **Идея.** Люди и LLM оцениваются на одних многотуровых диалогах по пяти измерениям: Human Alignment, Empathic Responsiveness, Attunement, Resonance, Task-following. Для sycophancy есть 20 adversarial cases **[польз.]**. Согласие human- и LLM-судей около 80%.
- **Что даёт проекту.** Основной протокол качества. Adversarial-кейсы для sycophancy. Методика валидации нашего judge против людей.
- **Ограничения.** 20 adversarial-кейсов мало для статистики, поэтому дополняем синтетикой с конфликтующими убеждениями.
- **Релиз (GitHub `Hippocratic-AI-Research/HEART`, коммит `c3387b43`, Apache-2.0), проверено 2026-10-20:** выложены контексты, ответы людей и ~20 моделей, judge-промпт. Человеческих попарных оценок **нет**. В опубликованном judge-промпте 6 категорий (empathic_responsiveness, emotional_handling_and_insight, personalization_and_contextual_adaptation, conversational_fluency_and_naturalness, instruction_following_and_safety, overall_eq), а не 5 измерений из текста статьи.
- **Найденные проблемы в коде HEART:** (1) judge-промпт размечает чётные реплики как Seeker, а 149/280 обычных контекстов начинаются с реплики supporter — роли перепутаны в 53% контекстов; (2) парсер при битом ответе засчитывает победу по всем категориям (это признаёт их README). Использовались ли эти версии для результатов статьи, неизвестно. Наша реализация (`src/careflow/eval/judge.py`) берёт роли из `prev_speakers` и парсит строго.
- **HEART построен на ESConv:** 240/300 контекстов пересекаются с ESConv (см. `docs/data.md`).

### ✅ Empath — arXiv:2609.29056
Gong, Huang, Liang, Deas, Subbiah, McKeown, Hirschberg. *Empath: Tracing Multi-Level Emotion Dynamics in Crisis Counseling Dialogues.* 2026-09-24.
- **Идея.** Три уровня: метки на уровне высказывания; микро-динамика (матрицы переходов, persistence, смены полярности); макро-динамика (архетипы разговоров, стратегии волонтёров).
- **Что даёт проекту.** Transition matrices, persistence, de-escalation rate. Основа детектора sycophancy через аномальную динамику.
- **Ограничения.** Узкий домен: текстовые кризисные диалоги с чернокожими пользователями о горе. Пороги «аномальности» калибруем на ESConv. Идея «sycophancy = аномальная динамика» — наша гипотеза, в статье её нет.

### ✅ Length-Controlled AlpacaEval — arXiv:2404.04475
Dubois, Galambosi, Liang, Hashimoto. *Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators.* 2024. GLM с членом длины (tanh) и общим для всех систем коэффициентом; LC win-rate = предсказание при нулевой разнице длин. **У нас:** `eval/length_control.py`.

### ⚠️ TOST / тесты эквивалентности — doi:10.1177/1948550617697177
Lakens, D. *Equivalence Tests: A Practical Primer for t Tests, Correlations, and Meta-Analyses.* Social Psychological and Personality Science, 2017. Процедура TOST (Schuirmann, 1987): эквивалентность в пределах δ, если (1−2α) CI внутри [−δ, δ]. **Цитируется по памяти — сверить DOI перед подачей.** **У нас:** `eval/equivalence.py`.

## 5. Синтетика и память

### ⚠️ Role-Guided Synthesis — DOI 10.1007/978-981-95-3352-7_46
Wang, Li, Zou, Wang, Wang. *Role-Guided Synthesis for Emotional Support: A Two-Stage Framework with Multi-agent Dialogue Synthesis.* NLPCC 2025, LNCS. Метаданные подтверждены через Crossref, полный текст за paywall.
- **[польз.]** Двухэтапный синтез для Personalized ESC: DeepSeek-V3 генерирует JSON-профили персонажей, агенты на Qwen2.5-32B ролево разыгрывают диалоги. +2.68 G-score, SOTA на NLPCC-2025 Shared Task 8.
- **Что даёт проекту.** Синтез диалогов с контролем персонажа, в том числе adversarial-сценарии эскалации и конфликтующих убеждений.

### ✅ H2HTalk — arXiv:2507.03543 / DOI 10.1007/978-981-95-3343-5_35
Wang, Wu, Guo, Li. *H2HTalk: Evaluating Large Language Models as Emotional Companion.* NLPCC 2025.
- **Идея.** 4650 сценариев (диалог, воспоминания, планирование), модуль Secure Attachment Persona, 50 LLM. Главные слабые места — long-horizon planning и memory retention.
- **Что даёт проекту.** Бенчмарк долгосрочной поддержки и памяти.

### ✅ ACERT — arXiv:2609.26422
Peuvot, Besançon, de Chalendar, Vieru, Vasilescu. *Enriching Speech Emotion Representations with Conversational Context.* 2026-09-22.
- **Идея.** Предшествующий контекст агрегируется temporal mean pooling и прибавляется к фреймам целевой реплики (FFN + residual). На MELD не помогает: эмоции там нестабильны (одна эмоция держится в среднем 5.5 с), в среднем 3.4 спикера.
- **Ключевая цитата.** «context whose emotional state differs from the target utterance provides no benefit over having no context at all».
- **Что даёт проекту.** Эмоциональная непрерывность важнее самого наличия контекста.
- **⚠️ Оговорка для нашей гипотезы.** Отсюда следует, что память полезна, когда эмоциональное состояние *сохраняется*. При резкой эскалации (наши синтетические сценарии) контекст с другим эмоциональным состоянием, по ACERT, бесполезен. Поэтому память нужно проверять отдельно на стабильных и эскалирующих диалогах, а не заявлять её пользу априори. Кроме того, это речевые данные.

## 6. Позиционирование и related work

### ✅ Motivated Emotional Mind — arXiv:2609.20437
Galus, Starzyk. *A Mathematical Model of Motivated Emotional Mind — Cognitive Embodied System.* 2026-09-17.
- Re-entrant loop (feedforward, lateral, feedback), semblions как ассоциативные структуры памяти, motivated learning через внутренние потребности, global affect как управляющий сигнал.
- Нужна для теоретического позиционирования архитектуры. Экспериментальной опоры не даёт.

### ✅ BiGraph-Diffuse — arXiv:2609.29519
Cheng, Tang, Lu, Zhang, Li, Cambria. *BiGraph-Diffuse: A Bidirectional Diffusion Language Model with Graph-Structured Retrieval For Mental Health Counseling.* 2026-08-25.
- Диффузионная (не авторегрессионная) LM: двунаправленный контекст позволяет пересматривать ранние интерпретации при progressive disclosure. Graph-RAG без LLM-извлечения связей.
- Альтернатива через архитектуру модели. Используем для позиционирования маршрутизации как другого пути.

### ✅ Transformer-GAT — arXiv:2609.29556
Qiao, Lyu, Xu. *Cross-Modal Emotion Understanding: A Transformer-GAT Approach for Dialogue Emotion Recognition.* 2026-09-02.
- Transformer ловит глобальную семантику, GAT — связи между модальностями и long/short/local контекст. Weighted F1: 72.45 (IEMOCAP), 77.37 (MELD).
- Related work по мультимодальному ERC. Графовое моделирование контекста — вариант для агента анализа контекста.
- **Ограничения.** Мультимодальная постановка. В нашем текстовом случае остаётся только графовая часть по репликам.

### ✅ BiCFlow-MER — arXiv:2609.27615
Wang, Wang, Yu. *BiCFlow-MER: Orchestrating Discriminative and Generative Multimodal Emotion Recognition via Conditional Transport.* 2026-09-23.
- Распознавание эмоций как «генеративный перенос свидетельств» в структурированном эмоциональном пространстве. Эмоция отделяется от стиля спикера и лексического содержания, есть conflict-aware условие для расхождения модальностей. Bidirectional rectified flow, adaptive prototype-cloud scoring. IEMOCAP, MELD, zero-shot CASE.
- **Что даёт проекту.** Идею conflict-aware условия для разрешения противоречий между агентами и геометрическую оценку по прототипам для валидации ответов.
- **Ограничения.** Мультимодальная и тяжёлая модель. Берём идеи, не метод целиком.

## 7. Датасеты

| Датасет | Источник | Статус | Заметки |
|---|---|---|---|
| ESConv | Liu et al., *Towards Emotional Support Dialog Systems*, ACL 2021, arXiv:2106.01144 | ✅ | 1035 диалогов, стратегии Helping Skills |
| EmpatheticDialogues | Rashkin, Smith, Li, Boureau, *Towards Empathetic Open-domain Conversation Models: a New Benchmark and Dataset*, ACL 2019, arXiv:1811.00207 | ✅ | 32 эмоции, ~25k диалогов. CC BY-NC 4.0 |
| TIDE | BN et al., EMNLP 2025, arXiv:2505.15065; HF `yenopoya/TIDE` | ⚠️ | 10k синтетических **двухтуровых** диалогов: для многотуровой динамики не годится. Проверить полноту релиза |
| EmoCare | неоднозначно | ❌ | (а) датасет «EmoCare (Psych8K)», упомянут в arXiv:2601.18630; (б) IEEE-чатбот EmoCare. Уточнить |
| KardiaBench | Kardia-R1, arXiv:2512.01282 (WWW 2026); github.com/JhCircle/Kardia-R1, MIT | ✅ ID сверен с репозиторием | 22 080 многотуровых эмпатических диалогов, 671 персона; HF, доступ по запросу; модель Kardia-R1-7B |
| TEA-Bench | arXiv:2601.18700 | ⚠️ ID из выдачи | tool-augmented ESC |
| MoodBench 1.0 | arXiv:2511.18926 | ⚠️ ID из выдачи | emotional companionship |
| HEART | arXiv:2601.19922 | ✅ | см. §4 |
| UniConv | UniESC (см. §2) | ⚠️ | доступность неизвестна |
| H2HTalk | arXiv:2507.03543 | ✅ | см. §5 |

## 7a. Инструменты

### ⚠️ Emotion English DistilRoBERTa-base — HF `j-hartmann/emotion-english-distilroberta-base`
Hartmann, J. (2022). Модель на HF Hub, ревизия `0e1cd914`; arXiv-статьи нет, лицензия в карточке не указана в тегах — проверить перед публикацией.
- 7 классов (Ekman 6 + neutral), используем для разметки эмоций реплик в `esconv_dynamics`.
- **Ограничение (наш результат):** на ESConv изменение её полярности почти не коррелирует с самоотчётом пользователей (ρ = 0.06, см. `docs/data.md`). Для оценки состояния пользователя её одной недостаточно.

## 8. Бейзлайны: специализированные подходы

### ⚠️ EmoDistill — arXiv:2605.26785
*EmoDistill: Offline Emotion Skill Distillation for Language Model Agents in Adversarial Negotiation.* 2026-05. IQL-селектор эмоции + LoRA-политика (SFT + JPO). **Домен — переговоры.** Не путать с SER-статьёй arXiv:2309.04849.

### ⚠️ EvoEmo — arXiv:2509.04310
*EvoEmo: Towards Evolved Emotional Policies for Adversarial LLM Agents in Multi-Turn Price Negotiation.* 2025-09. Эволюционный RL над переходами эмоций, представленными как MDP. **Домен — переговоры.**

> Это не ESC-бейзлайны: их цель — стратегическое выражение эмоций агентом. Берём из них техники (раздельные «выбор стратегии» и «реализация», эмоции как MDP), а не ставим строкой в таблицу. ESC-бейзлайн с дистилляцией — UniChat (UniESC).

## 9. Классика
- Chow, C. K. (1970). *On optimum recognition error and reject tradeoff.* IEEE Trans. Information Theory 16(1). Основа rejection option.
- Gross, J. J. (1998). *The emerging field of emotion regulation: An integrative review.* Review of General Psychology 2(3).
- Ekman, P. (1992). *An argument for basic emotions.* Cognition & Emotion 6(3–4). На ней основан InsideOut.

Данные классики приведены по памяти, перед BibTeX сверить.


## 10. Материалы от 2026-11-09 (прочитаны / просмотрены)

### ✅ MASCOT — EMNLP 2026 main, pp. 20293–20313 (PDF `2026.emnlp-main.878.pdf`, прочитан основной текст + прил. C)
Wang, Jin, Cabral, Hester (Georgia Tech, MIT). *MASCOT: Multi-Agent Socio-Collaborative Companion Systems.* Код: github.com/hello-diana/MASCOT.
- **Что:** MAS-«группа поддержки» из нескольких персон + Director; двухуровневая оптимизация: RLAIF для верности персоне каждого агента и групповая награда для директора (LoRA на Qwen3-8B, reward-модели на Qwen3-0.6B). Названные проблемы MAS: *persona collapse* и *social sycophancy* («yes-man» между агентами).
- **Оценка:** EmpatheticDialogues + QMSum, ESConv как held-out (прил. C.1); судьи GPT-4o и др., человеческая оценка; +14.1 persona consistency, на ESConv +10.8 по агентным критериям.
- **Связь с нами:** (1) прямое подтверждение нашей ablation-находки другим путём: без обучения MAS деградирует в шаблон; (2) их бейзлайны — только *многоагентные* при том же наборе агентов («isolates the optimization strategy»), сильного single-agent бейзлайна с равной стоимостью нет, длина контролируется только инструкцией судье (ссылаются на LC-AlpacaEval, но статистической поправки нет) — это ровно та дыра, которую закрывает наш протокол; (3) их «social sycophancy» — согласие *между агентами*, наша — согласие *с вредным убеждением пользователя*: разные явления, надо явно развести в related work; (4) MultiAgentESC у них — бейзлайн, ещё один довод добавить его к нашим 9 схемам.

### ✅ MentalHealthBench — OpenAI (PDF, прочитаны аннотация и разделы 2–4 выборочно)
Malik, Grabb, … Singhal. *MentalHealthBench: An Expert-Informed Benchmark of AI Capabilities in Realistic Mental Health Conversations.*
- **Что:** 1 215 реалистичных разговоров (уровни острóты: non-acute / high-acuity ≈ 18% / emergent), к каждому — взвешенные рубрики от 80+ лицензированных клиницистов (двое пишут вслепую, третий разрешает разногласия); 10 поведенческих осей, в т.ч. *urgency calibration*, *seeking context* (штраф и за недостаток, и за лишние вопросы), *reality-testing calibration*.
- **Связь с нами:** (1) операционализация «уместности» и «навязчивости» из нашей постановки — их оси можно взять как рубрику вместо придумывания своей; (2) «reality-testing calibration» ≈ наша сикофантия (не подтверждать искажённые убеждения) — внешняя опора для трёхаспектной схемы; (3) их вывод «пользователи ≠ эксперты» — довод, что наша ручная разметка должна иметь клиническую инструкцию. ⚠️ Доступность данных не проверена (в тексте ссылки на релиз не нашли) — уточнить.

### ⚠️ SAGE — Sentient Agent as a Judge (github.com/Tencent/digitalhuman/tree/main/SAGE)
Оценка LLM через симулированного пользователя с меняющимся эмоциональным состоянием (emotion score 0–100 по ходу диалога); лидерборд 20 моделей. Код и профили симулятора открыты. arXiv ID и лицензия на странице не указаны — сверить.
- **Связь с нами:** единственный найденный способ получить **метку изменения состояния** без людей — то, чего нам не хватает для RQ «эмоциональная динамика» (ESConv отфильтрован, классификаторы не ловят изменение). Кандидат на многотуровую оценку CareFlow v2 и на проверку детектора «аномальной траектории». Риск: состояние симулятора — тоже LLM, его валидность надо проверять.

### ✅ MentalBench-Align — arXiv:2510.19032, MIT (github.com/abeerbadawi/MentalBench-Align)
*When Can We Trust LLMs in Mental Health? Large-Scale Benchmarks for Reliable Dialogue Evaluation and Alignment.* MentalBench-100k (10k разговоров, 23 состояния, 1 человеческий + 9 LLM-ответов) и **MentalAlign-70k: 70 000 оценок от 3 клинических экспертов и 4 LLM-судей** по 7 атрибутам.
- **Вывод авторов:** LLM-судьи завышают аффективные атрибуты на 0.4–0.8; эмпатия/полезность выглядят надёжно, но с широкой неопределённостью; **safety и relevance — плохая надёжность у всех судей**.
- **Связь с нами:** готовые человеческие оценки — можно **проверить нашего судью (Gemma) на людях без собственной разметки** (то, чего не позволял HEART). Их вывод про safety — довод не доверять судье в вопросах безопасности и держать отдельную разметку.

### ⚠️ ES-MemEval (github.com/slptongji/ES-MemEval)
Бенчмарк долговременной памяти в эмоциональной поддержке (данные `data/evo_emo.json`; метрики F1/BERTScore/LLM-judge; модели Mistral, Phi-3, GPT с RAG и без). arXiv и лицензия на странице не указаны. Связь — только RQ7 (память), который мы выводим из основной статьи; держать как ссылку для future work.

### Не литература: NeuralDeep (neuraldeep.tech/en/pricing)
OpenAI-совместимый API к открытым моделям на российских GPU (Qwen 3.6 / 3.8, gpt-oss-120b; от ~5–31 ₽ за 1M входных токенов; подписка от 2 500 ₽/мес; оплата российскими картами). **Применение:** второй судья другого семейства (gpt-oss-120b) или 27B+-модели без сервера коллеги. Оговорка: API не детерминирован и меняется — только с нашим кэшем вызовов и фиксацией даты; данные (ESConv/синтетика) уходят третьей стороне — проверить лицензии датасетов (ESConv — CC BY-NC).
