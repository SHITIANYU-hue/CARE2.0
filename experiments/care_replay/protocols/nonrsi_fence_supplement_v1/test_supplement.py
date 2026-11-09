from dataclasses import dataclass
from pathlib import Path
import gzip
import json
import pytest
from .normalization import unwrap
from . import runner as r


def test_only_outer_wrapper_is_removed():
    body = '\n{"skills": [], "rationale": "keep  spaces"}\n'
    assert unwrap(' \n```json\n' + body + '\n```\n') == (body, True)
    assert unwrap(body) == (body, False)
    # Unwrapping does not make a semantically invalid library valid.
    assert r.original.generation.contract.strict_validate_library(json.loads(body), {})


@pytest.mark.parametrize('text', [
    'Explanation\n```json\n{}\n```', '```json\n{}\n```\nExplanation',
    '```json\n{}\n```\n```json\n{}\n```', '```python\n{}\n```',
    '```json\n{}', '```json\n```\n{}\n```'])
def test_prose_multiple_partial_or_other_language_blocks_are_not_repaired(text):
    assert unwrap(text) == (text, False)


@dataclass
class Skill:
    skill_id: str


def setup_source(tmp_path, monkeypatch):
    source = tmp_path / 'source'; output = tmp_path / 'supplement'
    monkeypatch.setattr(r.original, 'OUTPUT', source)
    monkeypatch.setattr(r.matched.semantic, 'skill_mode', lambda skill: skill.skill_id)
    monkeypatch.setattr(r.matched, 'execution_plan', lambda skills, enabled: (skills, {}))
    monkeypatch.setattr(r.matched, 'expand_metric_aliases', lambda rows, aliases: rows)
    monkeypatch.setattr(r.inputs, 'adapter', lambda target: None)
    monkeypatch.setattr(r.pipeline, 'verify_trace', lambda *args: None)
    modes = (*r.matched.BASELINES, 'existing')
    rows = [{'mode': mode, 'seed': 1, 'value': i} for i, mode in enumerate(modes)]
    traces = [{'mode': mode, 'seed': 1, 'round_index': 0} for mode in modes]
    path = source / 'target/calibration/1.json'; path.parent.mkdir(parents=True)
    r.save(path, {'binding': 'source', 'rows': rows, 'rows_sha256': r.digest(rows), 'traces_sha256': r.digest(traces)})
    path.with_suffix('.traces.json.gz').write_bytes(gzip.compress(json.dumps(traces).encode()))
    return path, output, rows, traces


def test_cached_modes_reused_without_any_replay_or_original_write(tmp_path, monkeypatch):
    path, output, rows, traces = setup_source(tmp_path, monkeypatch)
    original_bytes = path.read_bytes()
    monkeypatch.setattr(r.evaluation, 'replay_seed', lambda *args: pytest.fail('Existing mode must not be replayed'))
    result = r.merged_worker(1, target='target', split='calibration', skills=(Skill('existing'),),
        output=output, binding='supplement', source_binding='source')
    assert result == rows and path.read_bytes() == original_bytes


def test_baseline_mismatch_stops_before_publishing_supplement(tmp_path, monkeypatch):
    path, output, rows, traces = setup_source(tmp_path, monkeypatch)
    changed = [dict(row, value=-99) if row['mode'] in r.matched.BASELINES else row for row in rows]
    monkeypatch.setattr(r.evaluation, 'replay_seed', lambda *args: (changed, traces, 1.))
    with pytest.raises(r.Halt, match='baseline differs'):
        r.merged_worker(1, target='target', split='calibration', skills=(Skill('new'),),
            output=output, binding='supplement', source_binding='source')
    assert not (output / 'target/calibration/1.json').exists()
