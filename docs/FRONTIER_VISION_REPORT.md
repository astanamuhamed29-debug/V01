# FRONTIER VISION REPORT — SELF-OS Stage 1 → Stage 5

> **Статус:** Архитектурный аудит · Февраль 2026  
> **Роль:** Principal AI Architect / Chief Science Officer  
> **Цель:** Спроектировать эволюцию SELF-OS от пассивного парсера дневников до автономной прогностической когнитивной системы (Stage 5).

---

## Оглавление

1. [Текущая архитектура — срез реальности](#1-текущая-архитектура--срез-реальности)
2. [Cognitive Architecture & Memory — от базы данных к Памяти](#2-cognitive-architecture--memory)
3. [Multi-Agent / Multi-Model Dynamics — от пайплайна к Обществу Разума](#3-multi-agent--multi-model-dynamics)
4. [Dynamical Systems & Predictive Engine — от аналитики к Симуляции](#4-dynamical-systems--predictive-engine)
5. [The "SELF" Representation — Мета-когнитивный слой](#5-the-self-representation)
6. [Roadmap: Stage 1 → Stage 5](#6-roadmap-stage-1--stage-5)

---

## 1. Текущая архитектура — срез реальности

### Что работает сейчас

```
User Message
    │
    ▼
[Journal Storage]      ──► raw SQLite journal_entries
    │
    ▼
[Router]               ──► regex intent classification
    │
    ├──► [LLM Extractor]   (OpenRouter / Qwen)
    │         └──► parse_json_payload → map_payload_to_graph
    │
    └──► [Regex Extractors]
               ├── extractor_semantic
               ├── extractor_emotion
               └── extractor_parts
    │
    ▼
[Graph API]            ──► upsert nodes & edges → SELF-Graph (SQLite / NetworkX)
    │
    ├──► [Embedding Service]    dense vectors per node
    ├──► [MoodTracker]          PAD snapshots
    ├──► [PartsMemory]          IFS Part appearances
    ├──► [PatternAnalyzer]      Syndrome + ImplicitLink detection
    ├──► [ThresholdCalibrator]  adaptive proactive thresholds
    └──► [AgentOrchestrator]    sequential multi-agent chain
    │
    ▼
[Reply Generator]      ──► final response to user
```

### Архитектурные ограничения (что нужно взломать)

| Проблема | Модуль | Симптом |
|---|---|---|
| Граф статичен; накапливается линейно без консолидации | `core/graph/storage.py` | Со временем становится "кладбищем узлов" |
| `MessageProcessor` — монолит, делает всё последовательно | `core/pipeline/processor.py` | Невозможно параллелизировать части |
| `PatternAnalyzer` констатирует постфактум | `core/analytics/pattern_analyzer.py` | Нет предиктивного режима |
| Нет модели "Терапевтического Self" | везде | Бот реагирует, не планирует |
| SQLite не масштабируется для графовых траверсалов | `core/graph/storage.py` | O(N) при поиске путей |
| `AgentOrchestrator` — последовательная цепочка без внутреннего диалога | `agents/orchestrator.py` | Части не "спорят" до ответа |

---

## 2. Cognitive Architecture & Memory

### Проблема

Граф накапливается линейно. `GraphStorage` — это append-only хранилище узлов и рёбер без механизмов консолидации. Через 6 месяцев граф станет шумным и медленным. `ebbinghaus_retention` уже реализован в `model.py`, но никогда не применяется для активного удаления.

### Frontier-концепция: трёхуровневая архитектура памяти

```
┌─────────────────────────────────────────────────────┐
│  Working Memory (в рамках сессии)                    │
│  session_memory.py  ─  sliding window 20 messages    │
└────────────────────┬────────────────────────────────┘
                     │  Consolidation (каждые 24 ч)
                     ▼
┌─────────────────────────────────────────────────────┐
│  Episodic Memory (SQLite graph, последние 90 дней)   │
│  nodes + edges с temporally-decayed weights          │
└────────────────────┬────────────────────────────────┘
                     │  Abstraction (еженедельно)
                     ▼
┌─────────────────────────────────────────────────────┐
│  Semantic / Long-Term Memory (Neo4j / FalkorDB)      │
│  устойчивые убеждения, ценности, роли (BELIEF, NEED, │
│  VALUE, PART) с высоким retention score              │
└─────────────────────────────────────────────────────┘
```

#### Механизм Consolidation

**Интерфейс для создания:**

```python
class MemoryConsolidator(Protocol):
    async def consolidate(self, user_id: str) -> ConsolidationReport: ...
    async def abstract(self, user_id: str) -> AbstractionReport: ...
    async def forget(self, user_id: str, threshold: float = 0.05) -> int: ...
```

**Алгоритм:**
1. Каждые 24 часа запускать `consolidate()`:
   - Найти NOTE-узлы старше 3 дней с `ebbinghaus_retention < 0.3`.
   - Кластеризовать их через embedding similarity (cosine ≥ 0.82).
   - Для каждого кластера создать один BELIEF или THOUGHT узел, удалив исходные NOTE.
   - Перенести входящие/исходящие рёбра на новый узел.

2. Раз в неделю запускать `abstract()`:
   - Берём все BELIEF/THOUGHT узлы за 30 дней.
   - Через LLM-summarisation создаём "архетипные убеждения" и сохраняем в Semantic Memory.
   - Пример: 12 Note-узлов про "боюсь дедлайнов" → 1 BELIEF `key=fear_of_deadlines`.

3. Раз в 7 дней запускать `forget()`:
   - Удалять рёбра, у которых `ebbinghaus_retention < threshold`.
   - Узлы без входящих/исходящих рёбер и с `retention < 0.1` → tombstone (мягкое удаление).
   - Никогда не удалять BELIEF, NEED, VALUE с `review_count > 2`.

#### Механизм Reconsolidation

Когда пользователь прямо противоречит существующему убеждению:

```python
class ReconsolidationEngine:
    async def check_contradiction(
        self, user_id: str, new_text: str, existing_belief: Node
    ) -> ContraEvidence | None: ...

    async def update_belief(
        self, user_id: str, belief_id: str, evidence: ContraEvidence
    ) -> Node: ...
```

- Детектируется через `_cosine_similarity` нового embedding vs. старого BELIEF.
- Если similarity ∈ [0.5, 0.75] (семантически близко, но не совпадает) — сигнал реконсолидации.
- LLM-вызов: "Это противоречит или дополняет убеждение X?"
- При подтверждении: обновить `metadata["revision_count"]`, пересохранить текст.

#### Что нужно логировать уже сегодня

```python
# В Node.metadata добавить поля:
{
    "review_count": 0,           # для SM-2 / ebbinghaus
    "last_reviewed_at": None,    # ISO timestamp
    "consolidation_source": [],  # IDs исходных узлов
    "revision_history": [],      # [{text, timestamp, reason}]
    "salience_score": 1.0,       # 0–1, обновляется при каждом упоминании
    "abstraction_level": 0,      # 0=raw, 1=episodic, 2=semantic
}
```

#### Инструменты / библиотеки

- **Neo4j / FalkorDB** для Semantic Memory вместо SQLite (графовые траверсалы O(log N) вместо O(N)).
- **Qdrant** для векторного хранилища всех embeddings (замена `embedding` в SQLite).
- **Jina Embeddings v3** (1024-d, мультиязычная) или **text-embedding-3-large** (3072-d) для максимальной точности.
- Референс: *Park et al., "Generative Agents" (2023)* — memory stream + reflection tree.

---

## 3. Multi-Agent / Multi-Model Dynamics

### Проблема

`AgentOrchestrator` запускает агентов **последовательно** в одной цепочке. Нет внутреннего диалога между частями психики до формирования ответа. Текущие агенты (`SemanticExtractorAgent`, `EmotionAnalysisAgent` и др.) — безмолвные обработчики, не взаимодействующие друг с другом.

### Frontier-концепция: Society of Mind с IFS-ролями

```
                     ┌─────────────────────────────────┐
  User Message ──►   │      InnerCouncil Orchestrator   │
                     └────────────┬────────────────────┘
                                  │ параллельный запрос
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
     ┌─────────────┐   ┌────────────────┐   ┌──────────────┐
     │  CriticPart │   │ FirefighterPart│   │   SelfAgent  │
     │  (Manager)  │   │  (Protector)   │   │  (Observer)  │
     └──────┬──────┘   └───────┬────────┘   └──────┬───────┘
            │                  │                    │
            └──────────────────▼────────────────────┘
                         Internal Debate
                              │
                              ▼
                     ┌─────────────────┐
                     │  Reply Synthesizer│
                     └─────────────────┘
                              │
                              ▼
                         User Response
```

#### Новые интерфейсы

```python
class IFSPartAgent(BaseAgent):
    """Базовый агент-часть психики (IFS-модель)."""

    part_type: Literal["MANAGER", "FIREFIGHTER", "EXILE", "SELF"]
    voice_prompt: str  # system prompt, формирующий голос части

    async def deliberate(
        self, context: AgentContext, council_log: list[DebateEntry]
    ) -> DebateEntry: ...


@dataclass
class DebateEntry:
    part_type: str
    position: str        # аргумент части
    emotion: str         # эмоциональный тон
    need: str | None     # потребность, которую часть защищает
    confidence: float


class InnerCouncil:
    """Запускает параллельный внутренний диалог частей до генерации ответа."""

    async def deliberate(
        self,
        context: AgentContext,
        active_parts: list[IFSPartAgent],
        rounds: int = 2,
    ) -> CouncilVerdict: ...


@dataclass
class CouncilVerdict:
    dominant_part: str
    consensus_reply: str
    unresolved_conflict: bool
    internal_log: list[DebateEntry]
```

#### Multi-Agent Debate для терапевтического планирования

Вдохновлено *Du et al., "Improving Factuality and Reasoning in LLMs through Multi-Agent Debate" (2023)*:

1. **Round 1 (параллельно):** каждая IFS-часть получает текст пользователя + контекст графа и формирует позицию.
2. **Round 2 (дебаты):** части видят позиции друг друга и могут скорректировать или усилить позицию.
3. **SelfAgent (SELF):** финальный синтез — принимает позиции всех частей, выбирает терапевтически оптимальный ответ.

**Когда активировать:** только при высоком `session_conflict=True` или при наличии 2+ активных частей в `parts_context`.

#### Что убить / переписать

| Модуль | Действие |
|---|---|
| `agents/orchestrator.py` — `_INTENT_CHAINS` | Заменить статические цепочки на динамическую маршрутизацию через `InnerCouncil` |
| `AgentOrchestrator.run()` | Добавить параллельный `asyncio.gather` для независимых агентов |
| `MessageProcessor._extract_via_llm_all()` | Вынести в отдельный `ExtractionCoordinator` |

#### Инструменты

- **LangGraph** — граф агентов с conditional edges и state machine.
- **AutoGen** (Microsoft) — multi-agent conversation loops.
- Модели для частей: разные temperature/system_prompts через один OpenRouter endpoint.

---

## 4. Dynamical Systems & Predictive Engine

### Проблема

`PatternAnalyzer` смотрит назад: констатирует паттерны за последние 30 дней. Нет модели, способной ответить: "Что произойдёт с состоянием пользователя завтра при таких условиях?"

### Frontier-концепция: Digital Twin психики

```
┌──────────────────────────────────────────────────────────┐
│                    Digital Twin Layer                      │
│                                                            │
│  ┌──────────────┐   ┌───────────────┐   ┌─────────────┐  │
│  │ State Space   │   │  MDP over     │   │  Active     │  │
│  │ Model (SSM)   │──►│  Graph States │──►│  Inference  │  │
│  │ (Mamba/S4)   │   │  (transitions)│   │  (Free      │  │
│  └──────────────┘   └───────────────┘   │  Energy)    │  │
│                                          └─────────────┘  │
└──────────────────────────────────────────────────────────┘
```

#### Новый интерфейс: PsycheStateModel

```python
@dataclass
class PsycheState:
    """Snapshot психологического состояния пользователя."""
    timestamp: str
    valence: float          # -1..+1
    arousal: float          # -1..+1
    dominance: float        # -1..+1
    active_parts: list[str] # ключи активных IFS-частей
    dominant_need: str | None
    cognitive_load: float   # 0..1
    stressor_tags: list[str]


class PredictiveEngine(Protocol):
    async def predict_state(
        self,
        user_id: str,
        horizon_hours: int = 24,
        scenario: dict | None = None,  # {"event": "deadline", "severity": 0.8}
    ) -> list[PsycheStateForecast]: ...

    async def simulate_intervention(
        self,
        user_id: str,
        intervention_type: str,  # "CBT_reframe", "somatic", "validation"
        current_state: PsycheState,
    ) -> InterventionImpact: ...


@dataclass
class PsycheStateForecast:
    timestamp: str
    predicted_state: PsycheState
    confidence: float
    risk_flags: list[str]   # ["critic_activation_risk", "valence_drop"]


@dataclass
class InterventionImpact:
    intervention_type: str
    predicted_valence_delta: float
    predicted_arousal_delta: float
    confidence: float
    recommendation: str
```

#### Как реализовать

**Этап A — Сбор обучающих данных (сейчас):**

Добавить в каждый `MoodSnapshot`:
```python
{
    "context_events": [],        # события из журнала за +/-2 часа
    "active_parts": [],          # IFS-части в этот момент
    "stressors": [],             # тэги стрессоров
    "intervention_applied": None,# если бот применил интервенцию
    "feedback_score": None,      # пользовательский фидбек
}
```

**Этап B — Базовая модель переходов (State Space):**

Построить Марковскую цепь поверх `mood_snapshots`:
- Состояния: дискретизированные PAD-векторы (например, 8 кластеров через K-Means).
- Переходы: частоты `P(state_j | state_i, stressor_tag)`.
- Инструмент: `hmmlearn` или `pomegranate` для Hidden Markov Models.

**Этап C — Непрерывная модель (SSM / Mamba):**

Обучить State Space Model на временных рядах PAD-векторов:
- Вход: последовательность `(PAD_t, stressor_t, parts_t)` длиной 30 дней.
- Выход: `(PAD_{t+1}, risk_flags_{t+1})`.
- Модели: **Mamba** (selective SSM, O(N) complexity) или **S4** (structured SSM).
- Фреймворк: `torch` + `mamba-ssm` package.

**Этап D — Active Inference (Free Energy Principle):**

Долгосрочная цель (Stage 4+): внедрить принцип минимизации свободной энергии (*Friston, 2010*):
- Система строит генеративную модель пользователя.
- Каждое действие (интервенция) выбирается так, чтобы минимизировать surprisal.
- Библиотека: `pymdp` (Python Active Inference toolbox).

#### Что начать логировать уже сегодня

```python
# В journal_entries добавить поля:
{
    "session_id": "uuid",                  # группировать по сессиям
    "stressor_tags": ["deadline", "social"],
    "pre_mood_pad": [0.2, 0.5, 0.4],      # PAD до сообщения
    "post_mood_pad": None,                 # PAD после — заполнять через 30 мин
    "intervention_applied": None,
    "intervention_feedback": None,
    "active_parts_keys": [],
    "cognitive_load_estimate": None,       # 0–1, из LLM
}
```

---

## 5. The "SELF" Representation

### Проблема

Система не осознаёт себя как терапевтический инструмент. `generate_reply()` в `agents/reply_minimal.py` — это детерминированный шаблонный генератор. Нет долгосрочного плана терапии, нет оценки успешности интервенций, нет смены тактики.

### Frontier-концепция: Терапевтический Self-слой

```
┌─────────────────────────────────────────────────────────┐
│                 Therapeutic Self Layer                    │
│                                                           │
│  ┌──────────────────┐   ┌──────────────────────────┐     │
│  │  TherapyPlanner  │──►│  InterventionSelector    │     │
│  │  (долгосрочный   │   │  CBT / ACT / Somatic /   │     │
│  │   план терапии)  │   │  Validation / Silence    │     │
│  └──────────────────┘   └──────────────────────────┘     │
│           │                          │                    │
│           ▼                          ▼                    │
│  ┌──────────────────┐   ┌──────────────────────────┐     │
│  │  OutcomeTracker  │◄──│  EpistemicStateModel     │     │
│  │  (RLHF-сигналы) │   │  (Theory of Mind)         │     │
│  └──────────────────┘   └──────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
```

#### Новые интерфейсы

```python
class TherapyPlanner:
    """Долгосрочный план терапии пользователя.

    Хранится в отдельной таблице therapy_plans (SQLite → FalkorDB).
    Пересматривается еженедельно через PatternReport.
    """

    async def get_active_plan(self, user_id: str) -> TherapyPlan: ...
    async def revise_plan(
        self, user_id: str, report: PatternReport
    ) -> TherapyPlan: ...
    async def select_intervention(
        self, user_id: str, context: AgentContext
    ) -> Intervention: ...


@dataclass
class TherapyPlan:
    user_id: str
    primary_goal: str           # "снизить тревогу по дедлайнам"
    current_phase: str          # "psychoeducation" | "skill_building" | "integration"
    active_modality: str        # "CBT" | "ACT" | "IFS" | "somatic" | "validation"
    identified_patterns: list[str]
    contraindications: list[str]
    next_review_at: str
    revision_count: int


class InterventionSelector:
    """Выбирает тип интервенции на основе текущего состояния."""

    MODALITIES = ["CBT_reframe", "ACT_defusion", "IFS_parts_dialogue",
                  "somatic_grounding", "empathic_validation", "silence"]

    async def select(
        self,
        state: PsycheState,
        plan: TherapyPlan,
        recent_interventions: list[str],
    ) -> str: ...


class EpistemicStateModel:
    """Theory of Mind — модель убеждений пользователя о себе и о боте."""

    async def update(self, user_id: str, message: str) -> None: ...
    async def get_user_model(self, user_id: str) -> UserBeliefModel: ...


@dataclass
class UserBeliefModel:
    user_id: str
    trust_level: float          # 0–1, доверие к боту
    self_efficacy: float        # 0–1, вера в способность меняться
    insight_depth: float        # 0–1, глубина самопонимания
    resistance_level: float     # 0–1, сопротивление интервенциям
    last_updated: str


class OutcomeTracker:
    """Отслеживает эффективность интервенций (лёгкий RLHF)."""

    async def record_intervention(
        self,
        user_id: str,
        intervention_type: str,
        pre_state: PsycheState,
    ) -> str: ...  # returns tracking_id

    async def record_outcome(
        self,
        tracking_id: str,
        post_state: PsycheState,
        user_feedback: int | None,  # 1=helpful, 0=neutral, -1=harmful
    ) -> None: ...

    async def compute_effectiveness(
        self, user_id: str, intervention_type: str
    ) -> float: ...  # returns effectiveness score 0–1
```

#### RLHF в реальном времени

Вместо полноценного RLHF (требует отдельную модель-критик) — **лёгкий вариант**:

1. После каждого ответа бота логировать: `{intervention_type, pre_PAD, timestamp}`.
2. Через 15–30 минут после ответа: запросить MoodSnapshot.
3. Сравнить `post_PAD` с `pre_PAD`: `delta_valence = post_PAD.valence - pre_PAD.valence`.
4. Использовать `delta_valence` как proxy reward для `OutcomeTracker`.
5. `ThresholdCalibrator` расширяется до `InterventionCalibrator`: снижать частоту интервенций с отрицательным средним `delta_valence`.

**Явный фидбек:** после каждого ответа опционально показывать кнопки "Помогло / Нет" → записывать в `signal_feedback` (уже есть в `GraphStorage`).

---

## 6. Roadmap: Stage 1 → Stage 5

### Stage 1 ✅ — Passive Knowledge Graph (текущий)
**Что есть:** SQLite граф, regex + LLM extraction, MoodTracker, PatternAnalyzer, ThresholdCalibrator.  
**Ограничение:** реактивная система, нет памяти, нет предсказаний.

---

### Stage 2 — Consolidating Memory (3–4 месяца)

**Цель:** Граф обретает Forgetting, Consolidation и Reconsolidation.

**Что создать:**

| Интерфейс / Модуль | Действие |
|---|---|
| `core/memory/consolidator.py` | **Создать** `MemoryConsolidator` с методами `consolidate()`, `abstract()`, `forget()` |
| `core/memory/reconsolidation.py` | **Создать** `ReconsolidationEngine` |
| `core/graph/model.py` | **Расширить** `Node.metadata` полями `review_count`, `salience_score`, `abstraction_level` |
| `core/graph/storage.py` | **Добавить** `soft_delete_node()`, `merge_nodes()`, `get_nodes_by_retention()` |
| Scheduler | **Добавить** `core/scheduler/memory_scheduler.py` — APScheduler задачи |
| **SQLite → FalkorDB** | Начать миграцию Semantic Memory (BELIEF, NEED, VALUE, PART) в FalkorDB |
| **Embeddings → Qdrant** | Вынести векторы из `node.embedding` в Qdrant коллекцию `self_os_nodes` |

**Что убить:** `core/graph/storage.py` — метод сохранения `embedding` в SQLite (перенести в Qdrant).

**KPI:** Граф через 90 дней должен содержать не более 2× узлов, чем через 30 дней (consolidation ratio).

---

### Stage 3 — Society of Mind (4–6 месяцев)

**Цель:** Внутренний диалог IFS-частей до формирования ответа.

**Что создать:**

| Интерфейс / Модуль | Действие |
|---|---|
| `agents/ifs/` | **Создать** директорию с `CriticAgent`, `FirefighterAgent`, `ExileAgent`, `SelfAgent` |
| `agents/ifs/council.py` | **Создать** `InnerCouncil` с методом `deliberate()` |
| `agents/orchestrator.py` | **Переписать** `AgentOrchestrator.run()` — добавить `InnerCouncil` для конфликтных сессий |
| `core/pipeline/processor.py` | **Рефакторинг** `MessageProcessor` — вынести extraction в `ExtractionCoordinator` |
| `core/llm/prompts.py` | **Добавить** voice-промпты для каждой IFS-части |

**Что убить:** `_INTENT_CHAINS` статические словари → заменить динамической маршрутизацией.

**KPI:** На сессиях с `session_conflict=True` ответы должны упоминать конкретные части и их потребности ≥80% времени.

---

### Stage 4 — Predictive Digital Twin (6–12 месяцев)

**Цель:** Модель, способная предсказывать состояние пользователя.

**Что создать:**

| Интерфейс / Модуль | Действие |
|---|---|
| `core/prediction/state_model.py` | **Создать** `PsycheStateModel` (HMM → SSM/Mamba) |
| `core/prediction/engine.py` | **Создать** `PredictiveEngine` с `predict_state()`, `simulate_intervention()` |
| `core/mood/tracker.py` | **Расширить** `MoodSnapshot` полями `stressor_tags`, `active_parts_keys`, `context_events` |
| `core/journal/storage.py` | **Расширить** `journal_entries` полями для pre/post PAD |
| **Qdrant** | Хранить `PsycheState` snapshots как временные ряды |
| Training pipeline | **Создать** `tools/train_state_model.py` — обучение на накопленных данных |

**Что начать логировать прямо сейчас (Stage 1 → Stage 4 мост):**
```sql
ALTER TABLE mood_snapshots ADD COLUMN stressor_tags TEXT DEFAULT '[]';
ALTER TABLE mood_snapshots ADD COLUMN active_parts_keys TEXT DEFAULT '[]';
ALTER TABLE mood_snapshots ADD COLUMN intervention_applied TEXT;
ALTER TABLE mood_snapshots ADD COLUMN feedback_score INTEGER;
ALTER TABLE journal_entries ADD COLUMN session_id TEXT;
ALTER TABLE journal_entries ADD COLUMN cognitive_load REAL;
```

**KPI:** Accuracy предсказания dominant_label через 24 ч ≥ 65% (baseline random ≈ 25%).

---

### Stage 5 — Autonomous Predictive Cognitive System (12–18 месяцев)

**Цель:** Система самостоятельно ведёт долгосрочный терапевтический процесс.

**Что создать:**

| Интерфейс / Модуль | Действие |
|---|---|
| `core/therapy/planner.py` | **Создать** `TherapyPlanner` с `get_active_plan()`, `revise_plan()` |
| `core/therapy/intervention.py` | **Создать** `InterventionSelector` с полным MODALITIES реестром |
| `core/therapy/outcome.py` | **Создать** `OutcomeTracker` (лёгкий RLHF) |
| `core/theory_of_mind/user_model.py` | **Создать** `EpistemicStateModel` |
| `core/therapy/self_agent.py` | **Создать** `TherapeuticSelf` — мета-агент, координирующий всё |
| `agents/reply_minimal.py` | **Полностью переписать** — от шаблонов к `TherapeuticSelf.generate()` |
| **Neo4j / FalkorDB** | Полная миграция графа; SQLite остаётся только для journal |
| **Active Inference (pymdp)** | Интеграция Free Energy minimization для выбора интервенций |

**Финальная архитектура Stage 5:**

```
User Message
    │
    ▼
[EpistemicStateModel]  ──► обновить модель пользователя
    │
    ▼
[InnerCouncil]         ──► параллельный IFS-диалог частей
    │
    ▼
[PredictiveEngine]     ──► forecast_state + simulate_interventions
    │
    ▼
[TherapeuticSelf]      ──► выбрать модальность + интервенцию
    │                       с учётом TherapyPlan + UserBeliefModel
    ▼
[InterventionSelector] ──► CBT / ACT / IFS / somatic / validation
    │
    ▼
[Reply Synthesizer]    ──► финальный ответ с учётом voice + context
    │
    ▼
[OutcomeTracker]       ──► логировать для RLHF feedback loop
```

**KPI Stage 5:** PHQ-9 / GAD-7 прокси-снижение на 15% за 3 месяца активного использования (измеряется через PAD-тренды и self-report).

---

## Технологический стек — Frontier 2025–2026

| Слой | Текущий | Target (Stage 5) |
|---|---|---|
| Graph DB | SQLite | FalkorDB (Redis-native граф) + SQLite (journal) |
| Vector DB | SQLite blob | Qdrant (ANN search, temporal filtering) |
| Embeddings | OpenAI text-embedding | Jina v3 (1024-d) / text-embedding-3-large (3072-d) |
| LLM | OpenRouter / Qwen | Multi-model: Qwen (fast), Claude-3.7 (therapy), GPT-4.1 (synthesis) |
| Memory | Append-only graph | MemoryConsolidator + Hierarchical Memory |
| Prediction | None | Mamba SSM → Active Inference (pymdp) |
| Agent Framework | Custom orchestrator | LangGraph + AutoGen (debate loops) |
| Therapy Logic | Template replies | TherapeuticSelf + TherapyPlanner (RLHF-lite) |
| Neurosymbolic | None | KG reasoning (FalkorDB Cypher) + LLM hybrid |

---

## Что нужно сделать прямо сейчас (Sprint 0)

Приоритет — заложить фундамент логирования для будущего обучения:

1. **Расширить `Node.metadata` схему** (добавить `review_count`, `salience_score`, `abstraction_level`) — изменение обратно совместимо, `metadata` уже `dict[str, Any]`.
2. **Добавить SQL-миграцию** `mood_snapshots` и `journal_entries` с новыми колонками.
3. **Добавить `soft_delete` флаг** в таблицу `nodes` для будущего Forgetting.
4. **Запустить `ThresholdCalibrator.load()` при старте** — уже есть в коде, убедиться что вызывается.
5. **Добавить `session_id`** к каждому сообщению — группировать сессии для обучения SSM.
6. **Создать `OutcomeTracker` (минимальная версия)** — таблица `intervention_outcomes` в SQLite.

> **Принцип:** Данные, которые вы не логируете сегодня, нельзя использовать для обучения через год.

---

*Документ подготовлен как часть Project Genesis Frontier Architecture Audit.*  
*Следующий ревью: после завершения Stage 2 (Consolidating Memory).*
