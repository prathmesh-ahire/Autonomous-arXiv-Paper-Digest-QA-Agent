from arxiv_agent.state import AgentState, Intent


def test_default_state_round_trips_through_json():
    state = AgentState()
    dumped = state.model_dump_json()
    restored = AgentState.model_validate_json(dumped)

    assert restored == state
    assert restored.intent == Intent.UNKNOWN
    assert restored.candidates == []
    assert restored.messages == []
