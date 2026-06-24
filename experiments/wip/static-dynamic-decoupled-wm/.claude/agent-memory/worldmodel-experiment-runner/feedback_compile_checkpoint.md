---
name: feedback_compile_checkpoint
description: torch.compile 사용 시 checkpoint key에 _orig_mod. prefix가 붙어 로딩 실패
metadata:
  type: feedback
---

`torch.compile(model)` 사용 후 `model.state_dict()`를 저장하면 모든 key에 `_orig_mod.` prefix가 붙는다.

예: `_wm.encoder.xxx` → `_wm._orig_mod.encoder.xxx`

**Why:** dreamerv3-torch의 `Dreamer.__init__`에서 `config.compile=True`이면 `self._wm = torch.compile(self._wm)`. 이후 `agent.state_dict()`는 `_wm._orig_mod.xxx` 형태. 기존 `load_state_dict`에서 `_wm.` 만 strip하면 아무 key도 매칭되지 않아 silent random init 상태로 실행됨.

**How to apply:** checkpoint loading 시 반드시 두 패턴 처리:
```python
for k, v in state_dict.items():
    if k.startswith('_wm._orig_mod.'):
        wm_keys[k[len('_wm._orig_mod.'):]] = v
    elif k.startswith('_wm.'):
        wm_keys[k[len('_wm.'):]] = v
assert len(wm_keys) > 0, "Zero keys loaded — checkpoint loading failed"
```
loaded key count를 항상 출력하고 0이면 assertion 실패시켜 silent random init을 방지한다.
