from pathlib import Path

import pandas as pd

from vectorial_eval.config import DatasetConfig
from vectorial_eval.data.build_dataset import (
    _post_records,
    assign_splits,
    load_frame,
    select_cells,
)


def _write_v2_fixture(path: Path) -> None:
    rows = []
    for platform in ("linkedin", "reddit"):
        for i in range(5):
            rows.append(
                {
                    "post_id": f"{platform}-{i}",
                    "audience_room_name": "Backend Engineer (linkedin+reddit)",
                    "platform": platform,
                    "subreddit": "r/backend" if platform == "reddit" else None,
                    "job_title": "Backend Engineer" if platform == "linkedin" else None,
                    "domain": "Software Engineering",
                    "post_text": f"A sufficiently long unique {platform} post number {i}.",
                    "post_type": "opinion",
                    "final_topic": "databases",
                }
            )
    # These rows have platform-specific tags but no common audience identity.
    # They must never become one bilateral `nan` room.
    for platform in ("linkedin", "reddit"):
        rows.append(
            {
                "post_id": f"unmapped-{platform}",
                "audience_room_name": None,
                "platform": platform,
                "subreddit": "r/random" if platform == "reddit" else None,
                "job_title": "Unmapped Role" if platform == "linkedin" else None,
                "domain": "Software Engineering",
                "post_text": f"A sufficiently long unmapped {platform} post.",
                "post_type": None,
                "final_topic": "databases",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_v2_tags_are_preserved_without_creating_nan_audience(tmp_path):
    csv_path = tmp_path / "v2.csv"
    _write_v2_fixture(csv_path)
    cfg = DatasetConfig(csv_path=csv_path, out_dir=tmp_path, heldout_cell_frac=0)

    frame = load_frame(cfg)
    assert len(frame) == 10
    assert set(frame["room"]) == {"backend_engineer"}

    cells = select_cells(frame, cfg)
    split = assign_splits(frame, cells, cfg)
    posts = _post_records(split)
    linkedin = next(p for p in posts if p.platform == "linkedin")
    reddit = next(p for p in posts if p.platform == "reddit")
    assert linkedin.job_title == "Backend Engineer"
    assert linkedin.subreddit is None
    assert reddit.subreddit == "r/backend"
    assert reddit.job_title is None
