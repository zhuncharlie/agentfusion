"""R011: lightweight Markdown leaderboard, regenerated from results/main_table.csv and
embedded into README.md between the LEADERBOARD sentinel comments. No web dashboard --
a Markdown table is enough for this stage.
"""
import datetime
import re
from pathlib import Path

import pandas as pd

RESULTS_PATH = Path("results/main_table.csv")
README_PATH = Path("README.md")
START_MARK = "<!-- LEADERBOARD:START -->"
END_MARK = "<!-- LEADERBOARD:END -->"


def build_table() -> str:
    # Reads only main_table.csv -- deliberately doesn't import the agent modules (which
    # would pull in stable_baselines3/requests) just to enumerate Registry names, so this
    # script (and the CI job that runs it on every push) only needs pandas.
    table = pd.read_csv(RESULTS_PATH)
    avg = table.groupby("system")[["sharpe", "return", "mdd", "calmar", "win_rate"]].mean()
    today = datetime.date.today().isoformat()

    lines = ["| Agent | Sharpe | Return | MDD | Calmar | Last updated |", "|---|---|---|---|---|---|"]
    for system in sorted(avg.index):
        row = avg.loc[system]
        sharpe = f"{row['sharpe']:.3f}" if pd.notna(row["sharpe"]) else "— (no trades)"
        calmar = f"{row['calmar']:.3f}" if pd.notna(row["calmar"]) else "—"
        lines.append(
            f"| `{system}` | {sharpe} | {row['return']:.2%} | {row['mdd']:.2%} | "
            f"{calmar} | {today} |"
        )
    lines.append("")
    lines.append("_Averaged across AAPL/MSFT/NVDA, 2023-01-01~2023-06-30 test period. Preliminary._")
    return "\n".join(lines)


if __name__ == "__main__":
    new_block = f"{START_MARK}\n{build_table()}\n{END_MARK}"
    readme = README_PATH.read_text()
    pattern = re.compile(re.escape(START_MARK) + r".*?" + re.escape(END_MARK), re.DOTALL)
    if not pattern.search(readme):
        raise RuntimeError(f"Could not find {START_MARK}...{END_MARK} markers in README.md")
    README_PATH.write_text(pattern.sub(new_block, readme))
    print("README.md leaderboard updated.")
