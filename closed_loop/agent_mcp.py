from typing import Any, Dict, List

try:
    from .agent_bridge import bridge
except ImportError:  # pragma: no cover - script import fallback
    from agent_bridge import bridge


def freeaskworld_status() -> Dict[str, Any]:
    return bridge.get_status()


def freeaskworld_observe() -> Dict[str, Any]:
    return bridge.get_observation()


def freeaskworld_move_forward(distance_m: float = 1.0) -> Dict[str, Any]:
    return bridge.move_forward(distance_m=distance_m)


def freeaskworld_turn_left(degrees: float = 15.0) -> Dict[str, Any]:
    return bridge.turn_left(degrees=degrees)


def freeaskworld_turn_right(degrees: float = 15.0) -> Dict[str, Any]:
    return bridge.turn_right(degrees=degrees)


def freeaskworld_stop() -> Dict[str, Any]:
    return bridge.stop()


def freeaskworld_step() -> Dict[str, Any]:
    return bridge.step()


def freeaskworld_ask_human(prompt: str) -> Dict[str, Any]:
    return bridge.ask_human(prompt=prompt)


def list_tools() -> List[Dict[str, Any]]:
    return [
        {
            "name": "freeaskworld_status",
            "description": "Return current OpenClaw bridge connectivity and runtime status.",
            "callable": freeaskworld_status,
        },
        {
            "name": "freeaskworld_observe",
            "description": "Return the latest observation snapshot from closed_loop shared state.",
            "callable": freeaskworld_observe,
        },
        {
            "name": "freeaskworld_move_forward",
            "description": "Send a conservative forward NavigationCommand over the existing Unity websocket.",
            "callable": freeaskworld_move_forward,
        },
        {
            "name": "freeaskworld_turn_left",
            "description": "Send a conservative left-turn NavigationCommand compatibility mapping.",
            "callable": freeaskworld_turn_left,
        },
        {
            "name": "freeaskworld_turn_right",
            "description": "Send a conservative right-turn NavigationCommand compatibility mapping.",
            "callable": freeaskworld_turn_right,
        },
        {
            "name": "freeaskworld_stop",
            "description": "Send a stop NavigationCommand over the current Unity websocket connection.",
            "callable": freeaskworld_stop,
        },
        {
            "name": "freeaskworld_step",
            "description": "Send the existing Step message over the current Unity websocket connection.",
            "callable": freeaskworld_step,
        },
        {
            "name": "freeaskworld_ask_human",
            "description": "Record a human-help prompt through the additive bridge surface.",
            "callable": freeaskworld_ask_human,
        },
    ]
