"""Model routing for the existing manager / experiment-worker architecture."""


def decision_model(agent, fallback="feedback"):
    """Keep legacy configurations working while routing decisions explicitly."""
    return getattr(agent, "orchestrator", None) or getattr(agent, fallback)


def apply_strategy(config, orchestrator=None, worker=None):
    agent = config["agent"]
    if orchestrator:
        agent["orchestrator"] = {"model": orchestrator, "temp": 0.2}
    if worker:
        if not agent.get("orchestrator"):
            agent["orchestrator"] = dict(agent["code"])
        for role in ("code", "feedback", "vlm_feedback", "summary"):
            agent.setdefault(role, {"temp": 0.2})["model"] = worker
        config["report"]["model"] = worker
    if agent.get("orchestrator"):
        agent["select_node"] = dict(agent["orchestrator"])
    return config
