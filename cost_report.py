"""
Standardized cost reporting for API-calling scripts.

Usage:
    from cost_report import (
        calculate_cost, format_params,
        print_full_cost_report, print_summary_cost_report
    )

This module provides:
- Cost calculation with graceful handling of unknown models
- Formatted parameter display (reasoning_effort, temperature)
- Full cost report (detailed table of all API calls)
- Summary cost report (compact table by stage)
"""

import json
from pathlib import Path

# Load pricing from external file at module level
PRICING_FILE = Path(__file__).parent / "openai_pricing.json"
_PRICING = None
_UNKNOWN_MODELS_WARNED = set()  # Track models we've already warned about


def _load_pricing():
    """Load model pricing from external JSON file (lazy loading)."""
    global _PRICING
    if _PRICING is None:
        try:
            with open(PRICING_FILE, "r", encoding="utf-8") as f:
                _PRICING = json.load(f)
        except FileNotFoundError:
            print(f"⚠ Warning: {PRICING_FILE} not found - costs will show as $0.00")
            _PRICING = {"models": {}}
        except json.JSONDecodeError as e:
            print(f"⚠ Warning: {PRICING_FILE} is invalid JSON ({e}) - costs will show as $0.00")
            _PRICING = {"models": {}}
    return _PRICING


def get_model_pricing(model_name: str) -> tuple[float, float]:
    """
    Get pricing for a specific model from the loaded pricing data.
    
    Returns (input_rate, output_rate) per token.
    Returns (0.0, 0.0) for unknown models with a warning (once per model).
    """
    pricing = _load_pricing()
    
    if model_name in pricing.get("models", {}):
        p = pricing["models"][model_name]
        return p["input_cost_per_million"] / 1_000_000, p["output_cost_per_million"] / 1_000_000
    
    # Unknown model - warn once and return zero cost
    if model_name not in _UNKNOWN_MODELS_WARNED:
        print(f"⚠ Warning: Unknown model '{model_name}' - cost will show as $0.00")
        print(f"  Add it to {PRICING_FILE} for accurate cost tracking")
        _UNKNOWN_MODELS_WARNED.add(model_name)
    
    return 0.0, 0.0


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate cost for an API call.
    
    Returns 0.0 for unknown models (with warning printed once).
    """
    input_rate, output_rate = get_model_pricing(model)
    return input_tokens * input_rate + output_tokens * output_rate


def format_params(config: dict) -> str:
    """
    Format config params for display.
    
    Shows reasoning_effort and temperature (only if non-default).
    Example: "r=high, t=1.4" or "r=medium"
    """
    parts = [f"r={config.get('reasoning_effort', 'N/A')}"]
    if 'temperature' in config and config['temperature'] != 1.0:
        parts.append(f"t={config['temperature']}")
    return ", ".join(parts)


def print_full_cost_report(api_calls: list[dict], total_cost: float) -> None:
    """
    Print detailed ASCII table of all individual API calls.
    
    Expected api_calls format:
    [
        {
            "stage": "Generate",
            "api_call": "Generate-1",
            "model": "gpt-5.2",
            "params": "r=high, t=1.4",
            "time": 12.5,
            "cost": 0.0123,
            "input_tokens": 5000,
            "output_tokens": 1000
        },
        ...
    ]
    """
    if not api_calls:
        return
    
    print("\n" + "=" * 120)
    print("                                           FULL COST REPORT (All API Calls)")
    print("=" * 120)
    
    # Column headers
    headers = ["Stage", "API-Call", "Model", "Params", "Time", "Cost", "Cost%", "In-Tokens", "Out-Tokens"]
    
    # Calculate column widths (wide enough for longest expected values)
    col_widths = [12, 14, 14, 16, 8, 10, 7, 12, 12]
    
    # Print header row
    header_row = "│"
    for header, width in zip(headers, col_widths):
        header_row += f" {header:^{width}} │"
    
    separator = "├" + "┼".join(["─" * (w + 2) for w in col_widths]) + "┤"
    top_border = "┌" + "┬".join(["─" * (w + 2) for w in col_widths]) + "┐"
    bottom_border = "└" + "┴".join(["─" * (w + 2) for w in col_widths]) + "┘"
    
    print(top_border)
    print(header_row)
    print(separator)
    
    # Print data rows
    for call in api_calls:
        cost = call.get('cost')
        cost_pct = (cost / total_cost * 100) if (total_cost > 0 and cost is not None) else 0
        
        # Format values, handling None gracefully
        cost_str = f"${cost:.4f}" if cost is not None else "N/A"
        cost_pct_str = f"{cost_pct:.1f}%" if cost is not None else "N/A"
        in_tokens = call.get('input_tokens', 0)
        out_tokens = call.get('output_tokens', 0)
        
        row = "│"
        row += f" {call.get('stage', 'N/A'):<{col_widths[0]}} │"
        row += f" {call.get('api_call', 'N/A'):<{col_widths[1]}} │"
        row += f" {call.get('model', 'N/A'):<{col_widths[2]}} │"
        row += f" {call.get('params', 'N/A'):<{col_widths[3]}} │"
        row += f" {call.get('time', 0):>{col_widths[4]}.1f}s │"
        row += f" {cost_str:>{col_widths[5]}} │"
        row += f" {cost_pct_str:>{col_widths[6]}} │"
        row += f" {in_tokens:>{col_widths[7]},} │"
        row += f" {out_tokens:>{col_widths[8]},} │"
        print(row)
    
    print(bottom_border)
    
    # Print warning if any costs were zero due to unknown models
    if _UNKNOWN_MODELS_WARNED:
        print(f"\n⚠ Note: {len(_UNKNOWN_MODELS_WARNED)} model(s) had unknown pricing. "
              f"Update {PRICING_FILE} for accurate totals.")


def print_summary_cost_report(stage_summary: dict, total_time: float, total_cost: float) -> None:
    """
    Print compact summary table of costs by stage.
    
    Expected stage_summary format:
    {
        "Generate": {"time": 12.5, "model": "gpt-5.2", "cost": 0.0123},
        "Correct": {"time": 8.2, "model": "gpt-5.2", "cost": 0.0098},
        ...
    }
    """
    if not stage_summary:
        return
    
    print("\n" + "=" * 75)
    print("                         SUMMARY COST REPORT")
    print("=" * 75)
    
    # Column headers
    headers = ["Stage", "Time", "Model", "Cost", "Cost%"]
    col_widths = [14, 10, 14, 12, 8]
    
    # Print header row
    header_row = "│"
    for header, width in zip(headers, col_widths):
        header_row += f" {header:^{width}} │"
    
    separator = "├" + "┼".join(["─" * (w + 2) for w in col_widths]) + "┤"
    top_border = "┌" + "┬".join(["─" * (w + 2) for w in col_widths]) + "┐"
    bottom_border = "└" + "┴".join(["─" * (w + 2) for w in col_widths]) + "┘"
    
    print(top_border)
    print(header_row)
    print(separator)
    
    # Print data rows
    for stage_name, data in stage_summary.items():
        cost = data.get('cost')
        cost_pct = (cost / total_cost * 100) if (total_cost > 0 and cost is not None) else 0
        
        cost_str = f"${cost:.4f}" if cost is not None else "N/A"
        cost_pct_str = f"{cost_pct:.1f}%" if cost is not None else "N/A"
        
        row = "│"
        row += f" {stage_name:<{col_widths[0]}} │"
        row += f" {data.get('time', 0):>{col_widths[1]-1}.1f}s │"
        row += f" {data.get('model', 'N/A'):<{col_widths[2]}} │"
        row += f" {cost_str:>{col_widths[3]}} │"
        row += f" {cost_pct_str:>{col_widths[4]}} │"
        print(row)
    
    print(separator)
    
    # Print totals row
    total_cost_str = f"${total_cost:.4f}" if total_cost is not None else "N/A"
    
    total_row = "│"
    total_row += f" {'TOTAL':<{col_widths[0]}} │"
    total_row += f" {total_time:>{col_widths[1]-1}.1f}s │"
    total_row += f" {'':<{col_widths[2]}} │"
    total_row += f" {total_cost_str:>{col_widths[3]}} │"
    total_row += f" {'100.0%':>{col_widths[4]}} │"
    print(total_row)
    
    print(bottom_border)

