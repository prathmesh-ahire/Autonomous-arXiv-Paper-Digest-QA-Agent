from arxiv_agent.llm.null import NullLLM
from arxiv_agent.nodes.understand_query import understand_query
from arxiv_agent.state import AgentState, Intent


class _StubLLM:
    name = "stub"
    max_context_tokens = 1000

    def __init__(self, response: str) -> None:
        self._response = response
        self.calls = 0

    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        self.calls += 1
        return self._response


def test_id_input_sets_paper_lookup_without_llm() -> None:
    state = AgentState(raw_input="2301.12345")
    llm = _StubLLM("should never be used")

    result = understand_query(state, llm=llm)

    assert result == {"intent": Intent.PAPER_LOOKUP, "search_query": "2301.12345"}
    assert llm.calls == 0


def test_url_input_sets_paper_lookup_without_llm() -> None:
    state = AgentState(raw_input="https://arxiv.org/abs/2301.12345v2")
    llm = _StubLLM("should never be used")

    result = understand_query(state, llm=llm)

    assert result == {"intent": Intent.PAPER_LOOKUP, "search_query": "2301.12345"}
    assert llm.calls == 0


def test_topic_input_uses_llm_rewrite() -> None:
    state = AgentState(raw_input="kv cache compression techniques")
    llm = _StubLLM('all:"kv cache" AND all:compression')

    result = understand_query(state, llm=llm)

    assert result["intent"] == Intent.TOPIC_SEARCH
    assert result["search_query"] == 'all:"kv cache" AND all:compression'
    assert llm.calls == 1


def test_topic_input_falls_back_without_llm() -> None:
    state = AgentState(raw_input="kv cache compression techniques")

    result = understand_query(state, llm=NullLLM())

    assert result["intent"] == Intent.TOPIC_SEARCH
    assert result["search_query"]
    assert "NOT_A_REAL_WORD" not in result["search_query"]
    assert "all:" in result["search_query"]


def test_empty_input_is_unknown() -> None:
    state = AgentState(raw_input="   ")

    result = understand_query(state, llm=_StubLLM("unused"))

    assert result == {"intent": Intent.UNKNOWN}
