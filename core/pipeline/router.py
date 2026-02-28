from __future__ import annotations
import re


INTENTS = {
    "REFLECTION",
    "EVENT_REPORT",
    "IDEA",
    "TASK_LIKE",
    "FEELING_REPORT",
    "META",
    "UNKNOWN",
}

_FEELING = re.compile(
    r"""
    боюсь|страшно|тревож|беспоко|переживаю|нервнич|
    груст|грущу|плачу|обидно|обид[еи]лся|
    злюсь|злость|раздраж|агресс|
    чувствую|ощущаю|ощущение|
    стыд|стыжусь|вина|виноват|
    устал|выгорел|нет\s+сил|без\s+сил|апати|
    подавлен|депресс|пусто|пустота|
    одиноко|одинок|
    не\s+могу|ненавижу|
    радуюсь|рад[аы]?(?!\w)|счастлив|восторг|
    гордость|горжусь|
    обеспокоен|неспокойно|не\s+по\s+себе|
    залип|зависаю|прокрастин|
    паника|панику|
    бесит|достал|задолбал|заебал|
    кайф|кайфую|кайфово|
    грузит|давит|гнетёт|
    feel(?:ing)?|felt|emotion|
    afraid|scared|fear|anxious|anxiety|
    sad|sadness|cry|crying|
    angry|anger|rage|
    tired|exhausted|burnout|
    happy|joy|excited|
    lonely|alone|empty|
    stressed|overwhelmed|depressed|
    guilty|shame|ashamed|
    proud|grateful|
    feeling\s+\w+|фил(?:инг)?
    """,
    re.IGNORECASE | re.VERBOSE,
)

_TASK = re.compile(
    r"""
    надо|нужно|необходимо|
    сделать|сделай|слеоай|выполнить|закончить|завершить|
    задача|дедлайн|срок|план(?:ировать)?|
    не\s+забыть|запомни|напомни|
    запланировал|поставил\s+цель|
    к\s+(?:завтра|понедельнику|пятнице|концу\s+недели|следующей\s+неделе)|
    до\s+(?:завтра|понедельника|конца|пятницы)|
    todo|to[\s\-]do|task|deadline|
    need\s+to|have\s+to|must|should|
    remind(?:er)?|schedule|plan(?:ned)?|
    by\s+(?:tomorrow|monday|friday|end\s+of)|
    don't\s+forget|
    таск|дедлайн|тудушка
    """,
    re.IGNORECASE | re.VERBOSE,
)

_IDEA = re.compile(
    r"""
    идея|придумал|придумала|
    хочу\s+(?:сделать|попробовать|начать|научиться|создать|запустить|написать|построить)|
    можно\s+(?:сделать|попробовать|было\s+бы)|
    было\s+бы\s+(?:здорово|круто|классно|неплохо|интересно)|
    интересно\s+(?:было\s+бы|попробовать)|
    задумался\s+о|задумалась\s+о|
    а\s+что\s+если|что\s+если\s+я|
    вот\s+бы|мечтаю|мечта\b|
    концепт|концепция|
    idea|concept|
    want\s+to\s+(?:try|start|learn|build|create|make)|
    what\s+if\s+(?:i|we)|
    would\s+be\s+(?:cool|great|nice|awesome)|
    thinking\s+(?:about|of)\s+(?:starting|building|creating)|
    dream(?:ing)?\s+(?:of|about)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_EVENT = re.compile(
    r"""
    сегодня|вчера|позавчера|
    на\s+(?:прошлой|этой)\s+неделе|
    только\s+что|недавно|
    произошло|случилось|было\b|
    встретился|встретилась|поговорил|поговорила|
    сходил|сходила|пошёл|пошла|
    прочитал|прочитала|посмотрел|посмотрела|
    узнал|узнала|понял|поняла|
    оказалось|выяснилось|оказывается|
    today|yesterday|
    just\s+(?:had|did|went|talked|saw|read|found|learned)|
    happened|occurred|
    met(?:\s+with)?|talked\s+(?:to|with)|
    went\s+to|came\s+(?:back|from)|
    turns\s+out|it\s+turned\s+out
    """,
    re.IGNORECASE | re.VERBOSE,
)

_META = re.compile(
    r"""
    зачем|нафига|
    в\s+чём\s+(?:\w+\s+){0,3}смысл|
    в\s+чем\s+(?:\w+\s+){0,3}смысл|
    в\s+ч[её]м\s+(?:\w+\s+){0,3}польза|
    какой\s+смысл|
    для\s+чего\s+(?:это|мне|всё)|
    что\s+это\s+(?:даёт|мне\s+даёт|значит)|
    что\s+ты\s+(?:умеешь|можешь|делаешь)|
    помощь\b|команды|справка|
    что\s+такое\s+self|зачем\s+писать|
    what(?:'s|\s+is)\s+the\s+point|
    why\s+(?:am\s+i|are\s+you|should\s+i)|
    what\s+(?:can\s+you|do\s+you)\s+do|
    what\s+(?:is|are)\s+you|
    help\b|commands\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_REFLECTION = re.compile(
    r"""
    думаю\s+(?:о|что|об)|думал\s+(?:о|что)|
    размышляю|размышлял|
    осознал|осознала|понимаю\s+(?:что|теперь)|
    заметил|заметила|
    (?:по\s+)?(?:моему|моей)\s+мнению|
    кажется\s+(?:мне|что)|
    вот\s+что\s+я|интересно\s+что|
    thinking\s+(?:about|that)|thought\s+about|
    realized|noticed|aware\s+(?:that|of)|
    reflecting\s+on|it\s+seems|i\s+think\s+that
    """,
    re.IGNORECASE | re.VERBOSE,
)


def classify(text: str) -> str:
    t = text.strip()

    if _META.search(t):
        return "META"
    if _FEELING.search(t):
        return "FEELING_REPORT"
    if _TASK.search(t):
        return "TASK_LIKE"
    if _IDEA.search(t):
        return "IDEA"
    if _EVENT.search(t):
        return "EVENT_REPORT"
    if _REFLECTION.search(t):
        return "REFLECTION"

    return "UNKNOWN"
